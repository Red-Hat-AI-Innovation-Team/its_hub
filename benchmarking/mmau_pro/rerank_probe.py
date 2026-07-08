"""Terminal answer-confidence re-rank experiment.

Run 6 showed EPF's failure is SELECTION, not exploration: the correct answer is in the swarm
(oracle 0.80-0.95) but EPF returns the wrong particle (selected ~0.55). This re-ranks the
finished EPF swarm by an answer-confidence signal and measures how much of that gap it recovers.

Three scorers (per finished particle, after stripping its trailing `Answer: ...` to get reasoning R):
  L-audio : letter read-out WITH audio  -> chat, P over option letters after "Answer:" (faithful)
  L-text  : letter read-out text-only   -> chat, no audio (controls for audio re-attend)
  O-text  : option likelihood text-only -> /v1/completions echo, length-normalized logP(option text)
            given question+reasoning (de-biases the letter surface form)
Two selection rules each: argmax-particle (pick the particle most confident in its own answer) and
conf-vote (sum the scorer's distribution over particles, argmax). Baselines on the same swarm:
epf-selected (status quo), majority, oracle (ceiling), rand (mean per-particle acc, floor).

Both GPUs via --endpoints (items round-robined). Resumable JSONL.

    python -m benchmarking.mmau_pro.rerank_probe \
        --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
        --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 \
        --jsonl benchmarking/mmau_pro/results/run07_rerank/rerank.jsonl \
        --csv benchmarking/mmau_pro/results/run07_rerank/rerank.csv --log benchmarking/mmau_pro/results/run07_rerank/rerank.log
"""

import asyncio
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict

import aiohttp
import click

from benchmarking.mmau_pro.cot_compare import _select_items_stratified
from benchmarking.mmau_pro.diversity_probe import build_epf
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index
from its_hub import OpenAICompatibleLanguageModel
from its_hub.api.types import ChatMessage

RULES = [  # baselines + the 6 re-rank rules; each maps the swarm -> one letter
    "epf", "majority", "laudio_argmax", "laudio_vote",
    "ltext_argmax", "ltext_vote", "otext_argmax", "otext_vote",
]
CSV_FIELDS = [
    "unique_id", "category", "method", "method_name", "budget", "num_choices",
    "n_particles", "n_unique", "gold_letter", "oracle_correct", "rand_acc",
    *[f"{r}_letter" for r in RULES], *[f"{r}_correct" for r in RULES], "error",
]


def _reasoning(content: str) -> str:
    """Drop the particle's final `Answer: ...` line -> the reasoning that precedes it."""
    return re.split(r"(?i)answer\s*[:\-]", content)[0].rstrip()


def _softmax_dict(d):
    if not d:
        return {}
    m = max(d.values())
    ex = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(ex.values())
    return {k: e / z for k, e in ex.items()}


def _letter_dist(resp, num_choices):
    """q(letter) from the first generated token's top_logprobs (softmax over valid letters)."""
    lp = (resp.get("_logprobs") or {}).get("content") or []
    if not lp:
        return {}
    valid = set(LETTERS[:num_choices])
    raw = {}
    for t in lp[0].get("top_logprobs", []):
        sym = t["token"].strip()
        if sym in valid:  # " B", "B", etc. all map to the letter
            raw[sym] = max(raw.get(sym, -1e9), t["logprob"])
    return _softmax_dict(raw)


def _option_scores_from_echo(data, prefix_len, num_choices):
    """q(letter) from /completions echo: length-normalized logP(option text) -> softmax."""
    raw = {}
    for ch in data.get("choices", []):
        lg = ch.get("logprobs") or {}
        pairs = [
            (o, lpv) for o, lpv in zip(lg.get("text_offset", []), lg.get("token_logprobs", []))
            if lpv is not None and o >= prefix_len
        ]
        score = (sum(lpv for _, lpv in pairs) / len(pairs)) if pairs else -1e9
        if ch["index"] < num_choices:
            raw[LETTERS[ch["index"]]] = score
    return _softmax_dict(raw)


def _argmax_particle(particles_committed, q):
    """Letter of the particle whose own committed letter has the highest scorer confidence."""
    best = None
    for cl in particles_committed:
        if cl is None:
            continue
        s = q.get(LETTERS[cl], 0.0)
        if best is None or s > best[0]:
            best = (s, LETTERS[cl])
    return best[1] if best else None


def _conf_vote(particles_q):
    """Sum each particle's q over letters; argmax."""
    agg = defaultdict(float)
    for q in particles_q:
        for letter, p in q.items():
            agg[letter] += p
    return max(agg, key=agg.get) if agg else None


def _base_row(rec, method, budget):
    return {"unique_id": rec.unique_id, "category": rec.category, "method": method,
            "method_name": METHODS[method], "budget": budget, "num_choices": len(rec.choices)}


async def _option_likelihood(session, comp_url, model_name, prefix, choices):
    prompts = [f"{prefix} {c}" for c in choices]
    async with session.post(comp_url, json={
        "model": model_name, "prompt": prompts, "max_tokens": 0,
        "echo": True, "logprobs": 1, "temperature": 0.0,
    }) as r:
        data = await r.json()
    return _option_scores_from_echo(data, len(prefix), len(choices))


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
                    done.add((r["unique_id"], int(r["method"]), int(r["budget"])))
    return rows, done


def _dedupe(rows):
    by = {}
    for r in rows:
        by[(r["unique_id"], int(r["method"]), int(r["budget"]))] = r
    return list(by.values())


@click.command()
@click.option("--endpoints", default="http://localhost:8100/v1,http://localhost:8101/v1")
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--prompts", default="4,5,7,9")
@click.option("--budgets", default="8,16,32")
@click.option("--temp", default=0.8)
@click.option("--ess-threshold", default=0.6)
@click.option("--early-phase", default=0.7)
@click.option("--max-steps", default=6)
@click.option("--max-tokens-per-step", default=300)
@click.option("--limit", default=100)
@click.option("--max-inflight", default=48)
@click.option("--jsonl", "jsonl_path", default=None)
@click.option("--csv", "csv_path", default=None)
@click.option("--log", "log_path", default=None)
def main(endpoints, model_name, api_key, data_root, prompts, budgets, temp, ess_threshold,
         early_phase, max_steps, max_tokens_per_step, limit, max_inflight,
         jsonl_path, csv_path, log_path):
    eps = [e.strip() for e in endpoints.split(",") if e.strip()]
    methods = [int(m) for m in prompts.split(",")]
    buds = [int(b) for b in budgets.split(",")]
    records = _select_items_stratified(load_mmau_mcq(data_root, subset="full"), limit)
    print(f"re-rank probe: prompts={methods} budgets={buds} items={len(records)} endpoints={len(eps)}", flush=True)

    lms = [OpenAICompatibleLanguageModel(endpoint=ep, api_key=api_key, model_name=model_name,
                                         max_tokens=max_tokens_per_step, max_concurrency=-1) for ep in eps]
    comp_urls = [ep.rstrip("/") + "/completions" for ep in eps]
    resumed_rows, done = _load_resume(jsonl_path)
    if resumed_rows:
        print(f"resume: {len(done)} (item,method,budget) already done", flush=True)

    async def _process(method, budget, rec, lm, comp_url, session, sem, out, lock):
        async with sem:
            try:
                msgs, _seed = build(method, rec, audio_mode="local-path")
                sys_msgs = [m for m in msgs if m.role == "system"]
                user_msg = msgs[-1]
                text_part = next(p["text"] for p in user_msg.content
                                 if isinstance(p, dict) and p.get("type") == "text")
                nch = len(rec.choices)
                gold = LETTERS[rec.answer_index] if rec.answer_index is not None else None

                epf = build_epf("mean_logprob", temp, max_steps, ess_threshold, early_phase)
                res = await epf.ainfer(lm, msgs, budget, return_response_only=False)
                contents = [p.get("content", "") for p in res.responses]
                committed = [predicted_index(c, rec.choices) for c in contents]

                # score each UNIQUE particle once
                uniq = {}
                for c in contents:
                    if c in uniq:
                        continue
                    reason = _reasoning(c)
                    asst = ChatMessage(role="assistant", content=reason + "\n\nAnswer:")
                    r_aud = await lm.agenerate_single([*msgs, asst], max_tokens=1, temperature=0.0,
                                                      logprobs=True, top_logprobs=20)
                    txt_msgs = [*sys_msgs, ChatMessage(role="user", content=text_part), asst]
                    r_txt = await lm.agenerate_single(txt_msgs, max_tokens=1, temperature=0.0,
                                                      logprobs=True, top_logprobs=20)
                    prefix = f"{text_part}\n\nReasoning: {reason}\n\nAnswer:"
                    q_opt = await _option_likelihood(session, comp_url, model_name, prefix, rec.choices)
                    uniq[c] = {
                        "q_audio": _letter_dist(r_aud, nch),
                        "q_text": _letter_dist(r_txt, nch),
                        "q_opt": q_opt,
                    }

                qa = [uniq[c]["q_audio"] for c in contents]
                qt = [uniq[c]["q_text"] for c in contents]
                qo = [uniq[c]["q_opt"] for c in contents]
                letters = {
                    "epf": (LETTERS[committed[res.selected_index]]
                            if committed[res.selected_index] is not None else None),
                    "majority": (Counter([LETTERS[c] for c in committed if c is not None]).most_common(1)[0][0]
                                 if any(c is not None for c in committed) else None),
                    "laudio_argmax": _argmax_particle(committed, _merge_max(qa)),
                    "laudio_vote": _conf_vote(qa),
                    "ltext_argmax": _argmax_particle(committed, _merge_max(qt)),
                    "ltext_vote": _conf_vote(qt),
                    "otext_argmax": _argmax_particle(committed, _merge_max(qo)),
                    "otext_vote": _conf_vote(qo),
                }
                oracle_c = (gold is not None) and any(LETTERS[c] == gold for c in committed if c is not None)
                rand_acc = (sum(1 for c in committed if c is not None and LETTERS[c] == gold) / len(committed)
                            if (gold is not None and committed) else None)
                row = {**_base_row(rec, method, budget), "n_particles": len(contents),
                       "n_unique": len(uniq), "gold_letter": gold or "",
                       "oracle_correct": oracle_c if gold is not None else None,
                       "rand_acc": (round(rand_acc, 4) if rand_acc is not None else None), "error": None}
                for rname, let in letters.items():
                    row[f"{rname}_letter"] = let or ""
                    row[f"{rname}_correct"] = (let == gold) if gold is not None else None
            except Exception as e:
                row = {**_base_row(rec, method, budget), "n_particles": budget, "n_unique": 0,
                       "gold_letter": "", "oracle_correct": None, "rand_acc": None,
                       "error": f"{type(e).__name__}: {e}",
                       **{f"{r}_letter": "" for r in RULES}, **{f"{r}_correct": None for r in RULES}}
        if out is not None:
            async with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
        return row

    async def _run():
        new_rows = []
        lock = asyncio.Lock()
        out = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115 (closed in finally; spans loop)
        async with aiohttp.ClientSession() as session:
            try:
                for method in methods:
                    for budget in buds:
                        per_ep = max(1, max_inflight // budget)
                        sems = [asyncio.Semaphore(per_ep) for _ in lms]
                        todo = [r for r in records if (r.unique_id, method, budget) not in done]
                        print(f"[P{method} b{budget}] {len(todo)} to do "
                              f"({len(records)-len(todo)} resumed) | per-endpoint conc={per_ep}", flush=True)
                        if not todo:
                            continue
                        t0 = time.time()
                        tasks = []
                        for i, rec in enumerate(todo):
                            j = i % len(lms)
                            tasks.append(_process(method, budget, rec, lms[j], comp_urls[j],
                                                  session, sems[j], out, lock))
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
            f.write(f"re-rank probe: prompts={methods} budgets={buds} items={len(records)}\n")
            f.write(f"EPF: mean_logprob, temp={temp}, systematic, ess={ess_threshold}, early={early_phase}\n\n")
            f.write(report + "\n")
        print(f"wrote report -> {log_path}", flush=True)


def _merge_max(list_of_q):
    """For argmax-particle we need a single q to look up each particle's committed-letter
    confidence; use the max confidence seen for each letter across the swarm's distributions."""
    agg = {}
    for q in list_of_q:
        for letter, p in q.items():
            agg[letter] = max(agg.get(letter, 0.0), p)
    return agg


def _acc(rows, key):
    g = [r for r in rows if r.get(key) in (True, False)]
    return (sum(1 for r in g if r[key]) / len(g)) if g else None, len(g)


def _fmt(x):
    return f"{x:.3f}" if x is not None else "  —  "


def build_report(rows) -> str:
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["method"]), int(r["budget"]))].append(r)
    methods = sorted({int(r["method"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})
    cols = ["epf", "majority", "laudio_argmax", "laudio_vote", "ltext_argmax",
            "ltext_vote", "otext_argmax", "otext_vote"]
    lines = [f"rows: {len(rows)} | errors: {sum(1 for r in rows if r.get('error'))}", ""]
    head = f"{'prompt':28s} {'bud':>3} {'oracle':>6} {'rand':>5} " + " ".join(f"{c:>13}" for c in cols)
    for m in methods:
        lines.append("")
        lines.append(head)
        for b in budgets:
            rs = cells.get((m, b), [])
            if not rs:
                continue
            orc, _ = _acc(rs, "oracle_correct")
            rand = [r["rand_acc"] for r in rs if r.get("rand_acc") is not None]
            rand_m = sum(rand) / len(rand) if rand else None
            cellvals = " ".join(f"{_fmt(_acc(rs, c+'_correct')[0]):>13}" for c in cols)
            lines.append(f"{('P'+str(m)+' '+METHODS[m])[:28]:28s} {b:>3} "
                         f"{_fmt(orc):>6} {_fmt(rand_m):>5} {cellvals}")
        # recovery rate: of oracle-right & epf-wrong items, fraction each re-rank rule recovers @max budget
        big = max(budgets)
        rs = cells.get((m, big), [])
        recov = [r for r in rs if r.get("oracle_correct") is True and r.get("epf_correct") is False]
        if recov:
            parts = []
            for c in cols[2:]:
                got = sum(1 for r in recov if r.get(c + "_correct") is True)
                parts.append(f"{c}={got}/{len(recov)}")
            lines.append(f"    recovery @b{big} (oracle-right but epf-wrong, n={len(recov)}): " + "  ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
