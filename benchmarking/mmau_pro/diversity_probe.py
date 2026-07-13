"""EPF diversity probe: run Entropic Particle Filtering and measure swarm diversity vs scaling.

For each (prompt, weight-signal, budget, item) we run EPF with `return_response_only=False`,
score EVERY particle, and record SMC diagnostics the score-only runner can't:

  distinct-answer ratio = #unique predicted letters / N      (->0 = collapsed swarm)
  consensus             = plurality fraction = max class / N (->1 = total convergence)
  final ESS ratio       = (1/sum p_i^2)/N, p=softmax(log_weights)  (effective survivors)
  oracle acc            = gold predicted by ANY particle      (is the answer in the swarm?)
  selected acc          = the_one (argmax-weight) correct      (what EPF returns)
  majority acc          = plurality vote correct

Decision rule: distinct-ratio high & oracle climbs with N -> diversity real, scaling worth it;
distinct-ratio ->0 regardless of N -> swarm converges, scaling wasted; oracle >> selected ->
diversity is fine, the self-certainty WEIGHT is the bottleneck (motivates a choice-confidence reward).

Uses BOTH GPUs: pass two --endpoints; items are round-robined across them. Resumable JSONL.

    python -m benchmarking.mmau_pro.diversity_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
        --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
        --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 --limit 100 --max-inflight 64 \
        --jsonl benchmarking/mmau_pro/results/run06_epf_div/epf_div.jsonl \
        --csv   benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
        --log   benchmarking/mmau_pro/results/run06_epf_div/epf_div.log
"""

import asyncio
import csv
import json
import os
import time
from collections import Counter, defaultdict

import click
import numpy as np

from benchmarking.mmau_pro.cot_compare import _select_all, _select_items_stratified
from benchmarking.mmau_pro.loader import SUBSET_FILES, load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index
from its_hub import (
    EntropicParticleFiltering,
    OpenAICompatibleLanguageModel,
    StepGeneration,
)

CSV_FIELDS = [
    "unique_id", "category", "length_type", "num_audio", "n_choices",
    "method", "method_name", "signal", "budget", "n_particles",
    "gold_letter", "selected_letter", "majority_letter", "preds",
    "distinct_ratio", "consensus", "ess_ratio", "parsed_ratio",
    "oracle_correct", "selected_correct", "majority_correct", "error",
]


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _letter(idx):
    return LETTERS[idx] if (idx is not None and 0 <= idx < len(LETTERS)) else ""


def compute_metrics(result, rec) -> dict:
    """Per-item SMC diagnostics from a ParticleFilteringResult (return_response_only=False)."""
    contents = [p.get("content", "") for p in result.responses]
    n = len(contents)
    preds = [predicted_index(c, rec.choices) for c in contents]  # choice index or None
    parsed = [p for p in preds if p is not None]
    gold = rec.answer_index

    # effective sample size from the FINAL log-weights
    ess_ratio = None
    if result.log_weights_lst:
        probs = _softmax(result.log_weights_lst)
        ess = 1.0 / float(np.sum(probs**2))
        ess_ratio = ess / n

    distinct_ratio = len(set(parsed)) / n if n else 0.0
    parsed_ratio = len(parsed) / n if n else 0.0
    if parsed:
        majority_idx, maj_count = Counter(parsed).most_common(1)[0]
        consensus = maj_count / n
    else:
        majority_idx, consensus = None, 0.0

    sel_idx = result.selected_index
    selected_pred = preds[sel_idx] if 0 <= sel_idx < n else None

    if gold is None:  # ungradeable item
        oracle_c = selected_c = majority_c = None
    else:
        oracle_c = any(p == gold for p in parsed)
        selected_c = (selected_pred == gold)
        majority_c = (majority_idx == gold)

    return {
        "n_particles": n,
        "gold_letter": _letter(gold),
        "selected_letter": _letter(selected_pred),
        "majority_letter": _letter(majority_idx),
        "preds": ",".join(_letter(p) or "?" for p in preds),
        "distinct_ratio": round(distinct_ratio, 4),
        "consensus": round(consensus, 4),
        "ess_ratio": (round(ess_ratio, 4) if ess_ratio is not None else None),
        "parsed_ratio": round(parsed_ratio, 4),
        "oracle_correct": oracle_c,
        "selected_correct": selected_c,
        "majority_correct": majority_c,
    }


def _base_row(rec, method, signal, budget):
    return {
        "unique_id": rec.unique_id, "category": rec.category, "length_type": rec.length_type,
        "num_audio": len(rec.audio_paths), "n_choices": len(rec.choices),
        "method": method, "method_name": METHODS[method], "signal": signal, "budget": budget,
    }


def build_epf(signal, temp, max_steps, ess_threshold, early_phase, step_token="\n\n",
              stop_regex=None, stop_on_repeat=False):
    sg = StepGeneration(step_token=step_token, stop_token="Answer:", max_steps=max_steps,
                        temperature=temp, stop_regex=stop_regex, stop_on_repeat=stop_on_repeat)
    return EntropicParticleFiltering(
        sg=sg,
        resampling_method="systematic",
        temperature_method="ess",
        ess_threshold=ess_threshold,
        early_phase=early_phase,
        self_certainty_signal=signal,
        self_certainty_style="logit",
    )


def _load_resume(path):
    rows, done = [], set()
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append(r)
                if not r.get("error"):
                    done.add((r["unique_id"], int(r["method"]), r["signal"], int(r["budget"])))
    return rows, done


def _dedupe(rows):
    by = {}
    for r in rows:
        by[(r["unique_id"], int(r["method"]), r["signal"], int(r["budget"]))] = r
    return list(by.values())


def _acc(rows, key):
    g = [r for r in rows if r.get(key) in (True, False)]
    return (sum(1 for r in g if r[key]) / len(g) if g else None), len(g)


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "  — "


def build_report(rows) -> str:
    """Trend table grouped by (prompt, signal) over budgets, + a decision-rule read per cell."""
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["method"]), r["signal"], int(r["budget"]))].append(r)

    methods = sorted({int(r["method"]) for r in rows})
    signals = sorted({r["signal"] for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})

    lines = [f"rows: {len(rows)} | errors: {sum(1 for r in rows if r.get('error'))}", ""]
    hdr = (f"{'prompt':28s} {'sig':12s} {'bud':>3} {'sel_acc':>8} {'oracle':>7} {'major':>7} "
           f"{'distinct':>8} {'consen':>7} {'ess':>6} {'parsed':>7} {'n':>4}")
    for m in methods:
        for sig in signals:
            lines.append("")
            lines.append(hdr)
            for b in budgets:
                rs = cells.get((m, sig, b), [])
                if not rs:
                    continue
                sel, n = _acc(rs, "selected_correct")
                orc, _ = _acc(rs, "oracle_correct")
                maj, _ = _acc(rs, "majority_correct")
                lines.append(
                    f"{('P'+str(m)+' '+METHODS[m])[:28]:28s} {sig:12s} {b:>3} "
                    f"{_fmt(sel):>8} {_fmt(orc):>7} {_fmt(maj):>7} "
                    f"{_fmt(_mean(rs,'distinct_ratio')):>8} {_fmt(_mean(rs,'consensus')):>7} "
                    f"{_fmt(_mean(rs,'ess_ratio'),2):>6} {_fmt(_mean(rs,'parsed_ratio'),2):>7} {n:>4}"
                )
            # decision-rule read (use the largest budget vs smallest non-1 budget)
            big = max(budgets)
            rs_big = cells.get((m, sig, big), [])
            if rs_big:
                dr = _mean(rs_big, "distinct_ratio")
                orc, _ = _acc(rs_big, "oracle_correct")
                sel, _ = _acc(rs_big, "selected_correct")
                gap = (orc - sel) if (orc is not None and sel is not None) else None
                verdict = []
                if dr is not None and dr < 0.15:
                    verdict.append("swarm COLLAPSES (distinct~0) -> scaling wasted")
                elif dr is not None:
                    verdict.append(f"diversity present (distinct={dr:.2f})")
                if gap is not None and gap >= 0.10:
                    verdict.append(f"oracle>>selected (+{gap:.2f}) -> WEIGHT is the bottleneck")
                lines.append(f"    -> @{big}: " + "; ".join(verdict))
    return "\n".join(lines)


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1,http://localhost:8101/v1",
              help="comma list of vLLM endpoints; items round-robin across them (both GPUs)")
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="full",
              help="parquet subset: full = testmini (957 MCQ), test = the FULL test set (5,090 MCQ)")
@click.option("--audio-root", default=None,
              help="root for relative audio paths when they live outside --data-root "
                   "(e.g. mmau_pro_audio/ for the test subset)")
@click.option("--prompts", default="4,5,7,9", help="comma list of prompt methods")
@click.option("--signals", default="mean_logprob,entropy", help="comma list of self-certainty signals")
@click.option("--budgets", default="1,8,16,32")
@click.option("--temp", default=0.8)
@click.option("--ess-threshold", default=0.6)
@click.option("--early-phase", default=0.7)
@click.option("--max-steps", default=6)
@click.option("--step-token", default="\n\n",
              help=r"generation/resampling boundary (default blank line; pass $'\n' for "
                   "line-per-step prompts like P9 — raise --max-steps accordingly)")
@click.option("--stop-regex", default=None,
              help=r"stop a trajectory when a step matches this regex INSTEAD of the "
                   r"'Answer:' substring (e.g. 'Answer:\s*(\\boxed\{)?\(?[A-K]\b' to only "
                   "stop on letter-final answers, not prose sub-answers)")
@click.option("--stop-on-repeat", is_flag=True, default=False,
              help="kill a trajectory whose new step repeats an earlier one modulo "
                   "digits/case/whitespace (guards degenerate sub-question loops)")
@click.option("--max-tokens-per-step", default=300)
@click.option("--limit", default=100, help="# items (stratified single-audio, or first-N for --select all)")
@click.option("--select", "select_mode", type=click.Choice(["stratified", "all"]), default="stratified",
              help="stratified = single-audio stratified to --limit; all = every MCQ (incl. multi-audio)")
@click.option("--max-inflight", default=64, help="target concurrent requests PER endpoint")
@click.option("--jsonl", "jsonl_path", default=None)
@click.option("--csv", "csv_path", default=None)
@click.option("--log", "log_path", default=None)
def main(endpoints, model_name, api_key, data_root, subset, audio_root, prompts, signals,
         budgets, temp, ess_threshold, early_phase, max_steps, step_token, stop_regex,
         stop_on_repeat, max_tokens_per_step, limit, select_mode, max_inflight,
         jsonl_path, csv_path, log_path):
    eps = [e.strip() for e in endpoints.split(",") if e.strip()]
    methods = [int(m) for m in prompts.split(",")]
    sigs = [s.strip() for s in signals.split(",") if s.strip()]
    buds = [int(b) for b in budgets.split(",")]

    recs = load_mmau_mcq(data_root, subset=subset, audio_root=audio_root)
    records = _select_all(recs, limit) if select_mode == "all" else _select_items_stratified(recs, limit)
    cat_mix = Counter(r.category for r in records)
    print(f"EPF diversity probe: prompts={methods} signals={sigs} budgets={buds} "
          f"items={len(records)} endpoints={len(eps)}", flush=True)
    print("category mix: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_mix.items())), flush=True)

    lms = [OpenAICompatibleLanguageModel(
        endpoint=ep, api_key=api_key, model_name=model_name,
        max_tokens=max_tokens_per_step, max_concurrency=-1) for ep in eps]

    resumed_rows, done = _load_resume(jsonl_path)
    if resumed_rows:
        print(f"resume: {len(done)} (item,method,signal,budget) already done", flush=True)

    async def _run_one(epf, method, signal, budget, rec, lm, sem, out, lock):
        async with sem:
            try:
                msgs, _seed = build(method, rec, audio_mode="local-path")
                res = await epf.ainfer(lm, msgs, budget, return_response_only=False)
                row = {**_base_row(rec, method, signal, budget),
                       **compute_metrics(res, rec), "error": None}
            except Exception as e:
                row = {**_base_row(rec, method, signal, budget),
                       "n_particles": budget, "gold_letter": _letter(rec.answer_index),
                       "selected_letter": "", "majority_letter": "", "preds": "",
                       "distinct_ratio": None, "consensus": None, "ess_ratio": None,
                       "parsed_ratio": None, "oracle_correct": None, "selected_correct": None,
                       "majority_correct": None, "error": f"{type(e).__name__}: {e}"}
        if out is not None:
            async with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
        return row

    async def _run():
        new_rows = []
        lock = asyncio.Lock()
        out = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115 (closed in finally; spans loop)
        try:
            for method in methods:
                for signal in sigs:
                    for budget in buds:
                        epf = build_epf(signal, temp, max_steps, ess_threshold, early_phase,
                                        step_token=step_token, stop_regex=stop_regex,
                                        stop_on_repeat=stop_on_repeat)
                        per_ep = max(1, max_inflight // budget)
                        sems = [asyncio.Semaphore(per_ep) for _ in lms]
                        todo = [r for r in records
                                if (r.unique_id, method, signal, budget) not in done]
                        print(f"[P{method} {signal} b{budget}] {len(todo)} to do "
                              f"({len(records)-len(todo)} resumed) | per-endpoint conc={per_ep}", flush=True)
                        if not todo:
                            continue
                        t0 = time.time()
                        tasks = []
                        for i, rec in enumerate(todo):
                            j = i % len(lms)
                            tasks.append(_run_one(epf, method, signal, budget, rec,
                                                  lms[j], sems[j], out, lock))
                        rows = await asyncio.gather(*tasks)
                        new_rows.extend(rows)
                        errs = sum(1 for r in rows if r.get("error"))
                        print(f"    done in {time.time()-t0:.0f}s "
                              f"({(time.time()-t0)/len(todo):.2f}s/item, {errs} errors)", flush=True)
        finally:
            if out is not None:
                out.close()
            for lm in lms:
                await lm.close()
        return new_rows

    new_rows = asyncio.run(_run())
    final_rows = _dedupe(_load_resume(jsonl_path)[0] if jsonl_path else (resumed_rows + new_rows))

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(final_rows)
        print(f"\nwrote {len(final_rows)} rows -> {csv_path}", flush=True)

    report = build_report(final_rows)
    print("\n" + report)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"EPF diversity probe: prompts={methods} signals={sigs} budgets={buds} "
                    f"items={len(records)}\n")
            f.write("category mix: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_mix.items())) + "\n")
            f.write(f"config: temp={temp} ess_threshold={ess_threshold} early_phase={early_phase} "
                    f"step_token={step_token!r} max_steps={max_steps} "
                    f"systematic resampling, style=logit\n\n")
            f.write(report + "\n")
        print(f"wrote report -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
