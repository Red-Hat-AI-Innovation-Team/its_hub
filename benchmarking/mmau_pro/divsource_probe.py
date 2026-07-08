"""Run 8 — where does the (lack of) diversity come from: generation or resampling?

Run 6 showed even at budget 32 the swarm reaches ~80% consensus (~4/5 particles agree). Two
hypotheses: (a) the particles were *born similar* (low generation diversity at temp 0.8) or
(b) they started diverse but EPF's resampling *collapsed* them. We separate these by running two
arms on the SAME items, holding generation fixed and toggling only resampling:

  EPF   : EntropicParticleFiltering as deployed (systematic resampling ON).
  INDEP : same StepGeneration / temp / weight, but resampling DISABLED -> N independent
          step-chunked trajectories = the generator's intrinsic diversity (no collapse possible).

If INDEP distinct ~= EPF distinct  -> born similar  (a)  -> the lever is generation/temperature.
If INDEP distinct >> EPF distinct  -> resampling collapses (b) -> the lever is the resampling.

We also log the per-step ESS curve (free: computed inside the EPF loop each step) for both arms.

    python -m benchmarking.mmau_pro.divsource_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
        --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 \
        --jsonl benchmarking/mmau_pro/results/run08_divsource/divsource.jsonl \
        --csv benchmarking/mmau_pro/results/run08_divsource/divsource.csv --log benchmarking/mmau_pro/results/run08_divsource/divsource.log
"""

import asyncio
import csv
import json
import os
import time
from collections import defaultdict

import click
import numpy as np

from benchmarking.mmau_pro.cot_compare import _select_all, _select_items_stratified
from benchmarking.mmau_pro.diversity_probe import compute_metrics
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from its_hub import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_filtering import (
    EntropicParticleFiltering,
    _softmax,
)

ARMS = ["epf", "indep"]
CSV_FIELDS = [
    "unique_id", "category", "method", "method_name", "budget", "arm", "num_choices",
    "n_particles", "gold_letter", "distinct_ratio", "consensus", "ess_ratio_final",
    "parsed_ratio", "selected_correct", "oracle_correct", "majority_correct",
    "n_steps", "ess_trace", "error",
]


class TracedEPF(EntropicParticleFiltering):
    """EPF that records the per-step ESS ratio, and can disable resampling (INDEP arm).

    A fresh instance is built per item so `ess_trace` is per-item (no cross-item races).
    """

    def __init__(self, *args, resample_off=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ess_trace = []
        self.resample_off = resample_off

    def _weights_to_probabilities(self, log_weights, current_step, num_particles):
        probs = _softmax(np.asarray(log_weights, dtype=float))
        ess_ratio = (1.0 / float(np.sum(probs**2))) / num_particles
        self.ess_trace.append(round(ess_ratio, 4))
        return super()._weights_to_probabilities(log_weights, current_step, num_particles)

    def _resampling(self, particles, probabilities, num_particles):
        if self.resample_off:  # identity -> particles continue independently (no collapse)
            return particles
        return super()._resampling(particles, probabilities, num_particles)


def build_traced(resample_off, temp, max_steps, ess_threshold, early_phase):
    sg = StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=max_steps, temperature=temp)
    return TracedEPF(
        sg=sg, resampling_method="systematic", temperature_method="ess",
        ess_threshold=ess_threshold, early_phase=early_phase,
        self_certainty_signal="mean_logprob", self_certainty_style="logit",
        resample_off=resample_off,
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
                    done.add((r["unique_id"], int(r["method"]), int(r["budget"]), r["arm"]))
    return rows, done


def _dedupe(rows):
    by = {}
    for r in rows:
        by[(r["unique_id"], int(r["method"]), int(r["budget"]), r["arm"])] = r
    return list(by.values())


def _acc(rows, key):
    g = [r for r in rows if r.get(key) in (True, False)]
    return (sum(1 for r in g if r[key]) / len(g)) if g else None, len(g)


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "  — "


def _ess_curve(rows):
    """Mean ESS at each step index across items (ragged traces -> average what's present)."""
    traces = [r["ess_trace"] for r in rows if r.get("ess_trace")]
    if not traces:
        return []
    maxlen = max(len(t) for t in traces)
    out = []
    for i in range(maxlen):
        vals = [t[i] for t in traces if i < len(t)]
        out.append(round(sum(vals) / len(vals), 3))
    return out


def build_report(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["method"]), int(r["budget"]), r["arm"])].append(r)
    methods = sorted({int(r["method"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})
    lines = [f"rows: {len(rows)} | errors: {sum(1 for r in rows if r.get('error'))}", ""]
    lines.append("=== distinct / consensus: EPF (resampling ON) vs INDEP (resampling OFF) ===")
    hdr = (f"{'prompt':28s} {'bud':>3} {'arm':>5} {'distinct':>8} {'consensus':>9} "
           f"{'sel_acc':>7} {'oracle':>6} {'n':>4}")
    for m in methods:
        for b in budgets:
            lines.append("")
            lines.append(hdr)
            d_epf = d_ind = None
            for arm in ARMS:
                rs = cells.get((m, b, arm), [])
                if not rs:
                    continue
                dist = _mean(rs, "distinct_ratio")
                cons = _mean(rs, "consensus")
                sel, _ = _acc(rs, "selected_correct")
                orc, n = _acc(rs, "oracle_correct")
                lines.append(f"{('P'+str(m)+' '+METHODS[m])[:28]:28s} {b:>3} {arm:>5} "
                             f"{_fmt(dist):>8} {_fmt(cons):>9} {_fmt(sel):>7} {_fmt(orc):>6} {n:>4}")
                if arm == "epf":
                    d_epf = dist
                else:
                    d_ind = dist
            if d_epf is not None and d_ind is not None:
                gain = d_ind - d_epf
                if gain >= 0.05:
                    verdict = (f"resampling COLLAPSES diversity (indep {d_ind:.3f} >> epf {d_epf:.3f}, "
                               f"+{gain:.3f}) -> lever = resampling")
                else:
                    verdict = (f"BORN SIMILAR (indep {d_ind:.3f} ~= epf {d_epf:.3f}) -> generation/"
                               f"temperature is the lever, not resampling")
                lines.append(f"    -> {verdict}")
    lines.append("")
    lines.append("=== per-step ESS curve (mean across items) ===")
    for m in methods:
        for b in budgets:
            for arm in ARMS:
                rs = cells.get((m, b, arm), [])
                if rs:
                    curve = _ess_curve(rs)
                    lines.append(f"P{m} b{b} {arm:>5}: " + " ".join(f"{v:.2f}" for v in curve))
    return "\n".join(lines)


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1,http://localhost:8101/v1")
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--prompts", default="4,5,7,9")
@click.option("--budgets", default="8,16,32")
@click.option("--select", "select_mode", type=click.Choice(["stratified", "all"]), default="stratified",
              help="stratified = single-audio stratified to --limit; all = every MCQ (incl. multi-audio)")
@click.option("--temp", default=0.8)
@click.option("--ess-threshold", default=0.6)
@click.option("--early-phase", default=0.7)
@click.option("--max-steps", default=6)
@click.option("--max-tokens-per-step", default=300)
@click.option("--limit", default=100)
@click.option("--max-inflight", default=64)
@click.option("--jsonl", "jsonl_path", default=None)
@click.option("--csv", "csv_path", default=None)
@click.option("--log", "log_path", default=None)
def main(endpoints, model_name, api_key, data_root, prompts, budgets, select_mode, temp, ess_threshold,
         early_phase, max_steps, max_tokens_per_step, limit, max_inflight,
         jsonl_path, csv_path, log_path):
    eps = [e.strip() for e in endpoints.split(",") if e.strip()]
    methods = [int(m) for m in prompts.split(",")]
    buds = [int(b) for b in budgets.split(",")]
    recs = load_mmau_mcq(data_root, subset="full")
    records = _select_all(recs, limit) if select_mode == "all" else _select_items_stratified(recs, limit)
    print(f"Run 8 divsource: prompts={methods} budgets={buds} arms={ARMS} select={select_mode} "
          f"temp={temp} items={len(records)} endpoints={len(eps)}", flush=True)

    lms = [OpenAICompatibleLanguageModel(endpoint=ep, api_key=api_key, model_name=model_name,
                                         max_tokens=max_tokens_per_step, max_concurrency=-1) for ep in eps]
    resumed_rows, done = _load_resume(jsonl_path)
    if resumed_rows:
        print(f"resume: {len(done)} (item,method,budget,arm) already done", flush=True)

    async def _process(method, budget, arm, rec, lm, sem, out, lock):
        async with sem:
            try:
                msgs, _seed = build(method, rec, audio_mode="local-path")
                epf = build_traced(arm == "indep", temp, max_steps, ess_threshold, early_phase)
                res = await epf.ainfer(lm, msgs, budget, return_response_only=False)
                m = compute_metrics(res, rec)
                row = {
                    "unique_id": rec.unique_id, "category": rec.category, "method": method,
                    "method_name": METHODS[method], "budget": budget, "arm": arm,
                    "num_choices": len(rec.choices), "n_particles": m["n_particles"],
                    "gold_letter": m["gold_letter"], "distinct_ratio": m["distinct_ratio"],
                    "consensus": m["consensus"], "ess_ratio_final": m["ess_ratio"],
                    "parsed_ratio": m["parsed_ratio"], "selected_correct": m["selected_correct"],
                    "oracle_correct": m["oracle_correct"], "majority_correct": m["majority_correct"],
                    "n_steps": len(epf.ess_trace), "ess_trace": epf.ess_trace, "error": None,
                }
            except Exception as e:
                row = {"unique_id": rec.unique_id, "category": rec.category, "method": method,
                       "method_name": METHODS[method], "budget": budget, "arm": arm,
                       "num_choices": len(rec.choices), "n_particles": budget, "gold_letter": "",
                       "distinct_ratio": None, "consensus": None, "ess_ratio_final": None,
                       "parsed_ratio": None, "selected_correct": None, "oracle_correct": None,
                       "majority_correct": None, "n_steps": 0, "ess_trace": [],
                       "error": f"{type(e).__name__}: {e}"}
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
                for budget in buds:
                    per_ep = max(1, max_inflight // budget)
                    sems = [asyncio.Semaphore(per_ep) for _ in lms]
                    for arm in ARMS:
                        todo = [r for r in records
                                if (r.unique_id, method, budget, arm) not in done]
                        print(f"[P{method} b{budget} {arm}] {len(todo)} to do "
                              f"({len(records)-len(todo)} resumed) | per-endpoint conc={per_ep}", flush=True)
                        if not todo:
                            continue
                        t0 = time.time()
                        tasks = [_process(method, budget, arm, rec, lms[i % len(lms)],
                                          sems[i % len(lms)], out, lock)
                                 for i, rec in enumerate(todo)]
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
            for r in final_rows:
                w.writerow({**r, "ess_trace": json.dumps(r.get("ess_trace", []))})
        print(f"\nwrote {len(final_rows)} rows -> {csv_path}", flush=True)

    report = build_report(final_rows)
    print("\n" + report)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Run 8 divsource: prompts={methods} budgets={buds} items={len(records)} "
                    f"(temp={temp}, mean_logprob, systematic; INDEP = resampling disabled)\n\n")
            f.write(report + "\n")
        print(f"wrote report -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
