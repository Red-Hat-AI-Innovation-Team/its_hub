"""Compare CoT-elicitation prompts on Qwen2.5-Omni / MMAU-Pro MCQ.

For each prompt method (see prompt.METHODS), run ONE generation per item (greedy, t=0) and
measure:
  - reasoned?   (>=15 words of reasoning before the final answer marker)
  - chunks       (\\n\\n-separated non-empty segments => how well PF can chunk it)
  - accuracy     (extracted letter vs gold)

This isolates "which prompt makes the model actually reason / score" from the PF step
machinery.

Item selection (`--select`):
  - smallest    : single-audio, smallest-audio-first, take --limit (original screen).
  - stratified  : even round-robin across `category` for coverage (availability-capped).
  - all         : every MCQ record (incl. multi-audio); respects --limit if given.
  - (--ids ...) : run exactly the given unique_ids.

For large runs use `--jsonl PATH`: each completed (method,item) row is streamed to that
JSONL immediately (resumable — a re-run skips already-done rows and retries errored ones),
and every generation is wrapped so one bad/long item can't abort the sweep.

    # full 957 x 9 (background, resumable)
    python -m benchmarking.mmau_pro.cot_compare \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
        --subset full --select all --audio-mode local-path --concurrency 24 \
        --jsonl benchmarking/mmau_pro/results/run05_cot957/cot957.jsonl \
        --csv   benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
        --log   benchmarking/mmau_pro/results/run05_cot957/cot957.log
"""

import asyncio
import csv
import json
import os
import re
import time
from collections import defaultdict

import click

from benchmarking.mmau_pro.loader import SUBSET_FILES, load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import LETTERS, predicted_index
from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.utils import extract_content_from_lm_response

CSV_FIELDS = [
    "unique_id", "category", "length_type", "num_audio", "n_choices", "method", "method_name",
    "correct", "predicted_letter", "gold_letter", "n_words", "n_chunks", "reasoned",
    "question", "choices", "gold_answer", "response", "error",
]


def analyze(full_text: str, rec) -> dict:
    """Parse one response: predicted index, correctness, reasoning length, chunk count.

    Metric defs kept identical to the original screen: `reasoned` = words before the
    final answer marker ('answer:' / 'answer-') >= 15; `n_chunks` = non-empty
    '\\n\\n'-separated segments (how PF/EPF would chunk it).
    """
    pi = predicted_index(full_text, rec.choices)
    correct = (pi == rec.answer_index) if rec.answer_index is not None else None
    reasoning = re.split(r"answer\s*[:\-]", full_text, flags=re.IGNORECASE)[0]
    n_words = len(reasoning.split())
    n_chunks = len([p for p in full_text.split("\n\n") if p.strip()])
    return {
        "predicted_index": pi,
        "correct": correct,
        "reasoned": n_words >= 15,
        "n_words": n_words,
        "n_chunks": n_chunks,
    }


def _make_row(rec, method, text, a, error):
    """Assemble one per-(method,item) row; all CSV_FIELDS present in success AND error paths."""
    pi = a["predicted_index"] if a else None
    return {
        "unique_id": rec.unique_id,
        "category": rec.category,
        "length_type": rec.length_type,
        "num_audio": len(rec.audio_paths),
        "n_choices": len(rec.choices),
        "method": method,
        "method_name": METHODS[method],
        "correct": (a["correct"] if a else None),
        "predicted_letter": (LETTERS[pi] if pi is not None else ""),
        "gold_letter": (LETTERS[rec.answer_index] if rec.answer_index is not None else ""),
        "n_words": (a["n_words"] if a else 0),
        "n_chunks": (a["n_chunks"] if a else 0),
        "reasoned": (a["reasoned"] if a else False),
        "question": rec.question,
        "choices": _choices_str(rec.choices),
        "gold_answer": rec.answer,
        "response": text,
        "error": error,
    }


def _select_items_smallest(records, limit):
    single = [r for r in records if len(r.audio_paths) == 1]
    single.sort(key=lambda r: os.path.getsize(r.audio_paths[0]))  # smallest audio first = fast
    return single if limit is None else single[:limit]


def _select_items_stratified(records, limit):
    """Even round-robin across `category` (availability-capped), within-category
    smallest-audio-first. Deterministic. Returns items ordered by (category, size)."""
    single = [r for r in records if len(r.audio_paths) == 1]
    by_cat = defaultdict(list)
    for r in single:
        by_cat[r.category].append(r)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: os.path.getsize(r.audio_paths[0]))
    if limit is None:
        limit = len(single)
    # round-robin in size-desc (then name) order so larger categories absorb the remainder
    cats = sorted(by_cat, key=lambda c: (-len(by_cat[c]), c))
    chosen, idx = [], dict.fromkeys(cats, 0)
    while len(chosen) < limit:
        progressed = False
        for c in cats:
            if len(chosen) >= limit:
                break
            if idx[c] < len(by_cat[c]):
                chosen.append(by_cat[c][idx[c]])
                idx[c] += 1
                progressed = True
        if not progressed:
            break  # pool exhausted before reaching limit
    chosen.sort(key=lambda r: (r.category, os.path.getsize(r.audio_paths[0])))
    return chosen


def _select_all(records, limit):
    """Every MCQ record (incl. multi-audio), parquet order; respects limit if given."""
    return records if limit is None else records[:limit]


def _choices_str(choices: list[str]) -> str:
    return " | ".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def _load_resume(path: str):
    """Return (rows, done_keys) from an existing JSONL; errored rows are NOT 'done'."""
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
                    done.add((r["unique_id"], int(r["method"])))
    return rows, done


def _dedupe(rows):
    """Keep the last row per (unique_id, method) — a retried success overrides an old error."""
    by = {}
    for r in rows:
        by[(r["unique_id"], int(r["method"]))] = r
    return list(by.values())


def _results_from_rows(rows):
    """Per-method (acc, reasoned, avg_words, avg_chunks, n_graded) for the metrics table."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[int(r["method"])].append(r)
    results = {}
    for m in sorted(by_m):
        rs = by_m[m]
        graded = [r for r in rs if r["correct"] in (True, False)]
        acc = sum(1 for r in graded if r["correct"] is True) / max(len(graded), 1)
        reasoned = sum(1 for r in rs if r["reasoned"]) / len(rs)
        words = sum(r["n_words"] for r in rs) / len(rs)
        chunks = sum(r["n_chunks"] for r in rs) / len(rs)
        results[m] = (acc, reasoned, words, chunks, len(graded))
    return results


def _build_table(results: dict) -> str:
    """The chunkability metrics table + chunkable-BEST line (acc/reasoned/words/chunks)."""
    lines = [
        f"{'#':>2} {'method':33s} {'acc':>6} {'reasoned':>9} {'avg_words':>10} {'avg_chunks':>11}"
    ]
    for m, (acc, reasoned, words, chunks, _n) in results.items():
        lines.append(
            f"{m:>2} {METHODS[m]:33s} {acc:6.3f} {reasoned:9.2f} {words:10.1f} {chunks:11.1f}"
        )
    ok = {m: v for m, v in results.items() if v[1] >= 0.8 and v[3] >= 2.0}
    pool = ok or results
    best = max(pool, key=lambda m: (pool[m][0], pool[m][3]))
    lines.append(
        f"\nBEST (reasons + chunkable + accurate): [{best}] {METHODS[best]} "
        f"acc={results[best][0]:.3f} reasoned={results[best][1]:.2f} "
        f"chunks={results[best][3]:.1f}"
    )
    return "\n".join(lines)


def _acc_over(rows):
    g = [r for r in rows if r["correct"] in (True, False)]
    c = sum(1 for r in g if r["correct"] is True)
    return (c / len(g) if g else 0.0), c, len(g)


def _build_score_report(rows) -> str:
    """Accuracy report: per-prompt overall + excluding single-choice, plus by-category matrix."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[int(r["method"])].append(r)
    methods = sorted(by_m)
    errors = sum(1 for r in rows if r.get("error"))

    lines = [f"rows: {len(rows)} | errors: {errors}", ""]
    lines.append(f"{'#':>2} {'method':33s} {'acc_all':>8} {'acc_excl1':>10} {'graded':>7} {'g_ex1':>6}")
    best = None
    for m in methods:
        rs = by_m[m]
        a_all, _, g_all = _acc_over(rs)
        a_ex, _, g_ex = _acc_over([r for r in rs if int(r.get("n_choices", 2)) != 1])
        lines.append(f"{m:>2} {METHODS[m]:33s} {a_all:8.3f} {a_ex:10.3f} {g_all:>7} {g_ex:>6}")
        if best is None or a_ex > best[1]:
            best = (m, a_ex)
    lines.append(f"\nBEST (accuracy excl. single-choice): [{best[0]}] {METHODS[best[0]]} = {best[1]:.3f}")

    # by-category matrix (acc_all)
    cat_items = {}
    for r in rows:
        cat_items.setdefault(r["category"], set()).add(r["unique_id"])
    cats = sorted(cat_items, key=lambda c: -len(cat_items[c]))
    lines.append("\nby-category accuracy (acc_all):")
    lines.append("  " + f"{'category':22s}{'n':>5} " + "".join(f"{'P'+str(m):>6}" for m in methods))
    for c in cats:
        cells = "".join(f"{_acc_over([r for r in by_m[m] if r['category'] == c])[0]:6.2f}" for m in methods)
        lines.append("  " + f"{c:22s}{len(cat_items[c]):>5} " + cells)
    return "\n".join(lines)


@click.command()
@click.option("--endpoint", required=True)
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--subset", type=click.Choice(list(SUBSET_FILES)), default="le30s")
@click.option("--audio-root", default=None,
              help="root for relative audio paths when they live outside --data-root "
                   "(e.g. mmau_pro_audio/ for the test subsets)")
@click.option("--select", "select_mode", type=click.Choice(["smallest", "stratified", "all"]), default="smallest")
@click.option("--methods", default=None,
              help="comma list of prompt numbers to run (default: all 9), e.g. 4,5,7,9")
@click.option("--ids", default=None, help="comma list of unique_ids to run exactly (overrides --select/--limit)")
@click.option("--limit", type=int, default=None, help="cap number of items (default: all for the chosen --select)")
@click.option("--audio-mode", type=click.Choice(["local-path", "base64"]), default="base64")
@click.option("--max-tokens", default=700)
@click.option("--concurrency", default=6)
@click.option("--jsonl", "jsonl_path", default=None, help="resumable per-row stream (recommended for big runs)")
@click.option("--csv", "csv_path", default=None, help="write per-(method,item) responses to this CSV")
@click.option("--log", "log_path", default=None, help="also tee the score/metric tables to this file")
def main(endpoint, model_name, api_key, data_root, subset, audio_root, select_mode, methods, ids,
         limit, audio_mode, max_tokens, concurrency, jsonl_path, csv_path, log_path):
    recs = load_mmau_mcq(data_root, subset=subset, audio_root=audio_root)
    method_list = [int(x) for x in methods.split(",")] if methods else list(METHODS)
    unknown = [m for m in method_list if m not in METHODS]
    if unknown:
        raise SystemExit(f"--methods not in {list(METHODS)}: {unknown}")
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        by_id = {r.unique_id: r for r in recs}
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit(f"--ids not found in {subset} pool: {missing}")
        records = [by_id[w] for w in wanted]  # preserve given order
        select_mode = "ids"
    else:
        selectors = {
            "smallest": _select_items_smallest,
            "stratified": _select_items_stratified,
            "all": _select_all,
        }
        records = selectors[select_mode](recs, limit)

    cat_counts = defaultdict(int)
    for r in records:
        cat_counts[r.category] += 1
    print(f"comparing {len(method_list)} prompts over {len(records)} MCQ items "
          f"(subset={subset}, select={select_mode}, audio={audio_mode})", flush=True)
    print("category mix: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_counts.items())), flush=True)

    resumed_rows, done = _load_resume(jsonl_path)
    if resumed_rows:
        print(f"resume: {len(done)} (item,method) already done in {jsonl_path}", flush=True)

    lm = OpenAICompatibleLanguageModel(endpoint=endpoint, api_key=api_key, model_name=model_name)
    sem = asyncio.Semaphore(concurrency)

    async def _gen(method, rec, out, lock) -> dict:
        try:
            msgs, seed = build(method, rec, audio_mode=audio_mode)
            async with sem:
                resp = await lm.agenerate_single(msgs, max_tokens=max_tokens, temperature=0.0)
            text = extract_content_from_lm_response(resp)
            if seed:
                text = seed + " " + text
            row = _make_row(rec, method, text, analyze(text, rec), None)
        except Exception as e:  # record, never abort the sweep
            row = _make_row(rec, method, "", None, f"{type(e).__name__}: {e}")
        if out is not None:
            async with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
        return row

    async def _run():
        new_rows: list[dict] = []
        lock = asyncio.Lock()
        out = open(jsonl_path, "a") if jsonl_path else None  # noqa: SIM115 (closed in finally; spans the loop)
        try:
            for m in method_list:
                todo = [r for r in records if (r.unique_id, m) not in done]
                print(f"[{m}] {METHODS[m]:33s} {len(todo)} to do "
                      f"({len(records) - len(todo)} resumed)", flush=True)
                if not todo:
                    continue
                t0 = time.time()
                rows = await asyncio.gather(*(_gen(m, r, out, lock) for r in todo))
                new_rows.extend(rows)
                errs = sum(1 for r in rows if r.get("error"))
                print(f"    done in {time.time() - t0:.0f}s "
                      f"({(time.time() - t0) / len(todo):.2f}s/item, {errs} errors)", flush=True)
        finally:
            if out is not None:
                out.close()
            await lm.close()
        return new_rows

    new_rows = asyncio.run(_run())

    # authoritative set: re-read the JSONL (resumed + new) else in-memory; dedupe last-wins
    final_rows = _dedupe(_load_resume(jsonl_path)[0] if jsonl_path else (resumed_rows + new_rows))

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in final_rows:
                w.writerow({**r, "correct": ("" if r["correct"] is None else r["correct"])})
        print(f"\nwrote {len(final_rows)} rows -> {csv_path}", flush=True)

    chunk_table = _build_table(_results_from_rows(final_rows))
    score_report = _build_score_report(final_rows)
    print("\n" + score_report)
    print("\n" + chunk_table)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"{len(method_list)} prompts over {len(records)} items "
                    f"(subset={subset}, select={select_mode}, audio={audio_mode})\n")
            f.write("category mix: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_counts.items())) + "\n\n")
            f.write(score_report + "\n\n" + chunk_table + "\n")
        print(f"wrote tables -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
