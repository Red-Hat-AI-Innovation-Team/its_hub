"""Run MMAU-Pro MCQ evaluation: prompt(s) x {baseline, PF, EPF} (self-certainty weights).

Reasoning is chunked into PF/EPF steps on '\\n\\n'. Items within a (prompt, arm, budget)
cell are processed CONCURRENTLY (async ainfer + bounded gather), so vLLM batches many
requests at once (item_concurrency x budget) instead of one item at a time. Requires a
served Qwen2.5-Omni (vLLM). Example (full 957, focused config):

    python -m benchmarking.mmau_pro.run_mmau \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --subset full \
        --prompt-methods 2,4 --arms baseline,pf,epf --budgets 4 \
        --audio-mode local-path --item-concurrency 10 \
        --output /home/exx/inference-time-scaling/mmau_957_results.jsonl
"""

import asyncio
import json
import os
import time
from collections import defaultdict

import click

from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import is_correct
from its_hub import (
    EntropicParticleFiltering,
    OpenAICompatibleLanguageModel,
    ParticleFiltering,
    StepGeneration,
)
from its_hub.core.utils import extract_content_from_lm_response


def build_algorithm(arm: str, max_steps: int):
    # chunk reasoning into steps on blank lines; stop once the model writes "Answer:"
    sg = StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=max_steps)
    if arm in ("baseline", "pf"):
        return ParticleFiltering(
            sg=sg, self_certainty_signal="mean_logprob", self_certainty_style="logit",
        )
    if arm == "epf":
        return EntropicParticleFiltering(sg=sg, self_certainty_signal="entropy")
    raise ValueError(f"unknown arm {arm!r}")


def _load_done(path: str) -> set:
    """Keys already completed successfully. Errored rows are NOT counted as
    done, so resuming a run retries items that failed transiently (the final
    report dedupes on key keeping the latest row)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add((r["unique_id"], r["method"], r["arm"], r["budget"]))
    return done


def report(rows: list[dict]) -> None:
    agg = defaultdict(lambda: [0, 0, 0])  # (method,arm,budget) -> [correct, gradeable, total]
    for r in rows:
        k = (r["method"], r["arm"], r["budget"])
        agg[k][2] += 1
        if r.get("correct") is None:
            continue
        agg[k][1] += 1
        agg[k][0] += int(bool(r["correct"]))
    print(f"\n{'prompt':32s} {'arm':9s} {'bud':>3} {'acc':>6}  (correct/gradeable; total)")
    for (m, arm, b), (c, g, t) in sorted(agg.items()):
        acc = c / g if g else 0.0
        print(f"{METHODS.get(m, m):32s} {arm:9s} {b:>3} {acc:6.3f}  ({c}/{g}; {t})")


@click.command()
@click.option("--endpoint", required=True)
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--subset", type=click.Choice(["full", "le30s"]), default="full")
@click.option("--limit", type=int, default=None)
@click.option("--single-audio", is_flag=True, help="only single-audio items (smaller payloads)")
@click.option("--prompt-methods", default="4", help="comma list of CoT methods (prompt-only; see prompt.METHODS)")
@click.option("--arms", default="baseline,pf,epf")
@click.option("--budgets", default="4")
@click.option("--audio-mode", type=click.Choice(["local-path", "base64"]), default="local-path")
@click.option("--max-steps", default=6)
@click.option("--max-tokens-per-step", default=300)
@click.option("--item-concurrency", default=10, help="items processed concurrently per cell")
@click.option("--output", default="mmau_pro_results.jsonl")
def main(
    endpoint, model_name, api_key, data_root, subset, limit, single_audio,
    prompt_methods, arms, budgets, audio_mode, max_steps, max_tokens_per_step,
    item_concurrency, output,
):
    methods = [int(m) for m in prompt_methods.split(",")]
    arms = [a.strip() for a in arms.split(",") if a.strip()]
    budgets = [int(b) for b in budgets.split(",")]

    records = load_mmau_mcq(data_root, subset=subset, limit=None)
    if single_audio:
        records = [r for r in records if len(r.audio_paths) == 1]
        records.sort(key=lambda r: os.path.getsize(r.audio_paths[0]))
    if limit is not None:
        records = records[:limit]
    print(f"loaded {len(records)} MCQ records (subset={subset}, single_audio={single_audio})")
    print(f"prompts={methods} arms={arms} budgets={budgets} item_concurrency={item_concurrency}")

    lm = OpenAICompatibleLanguageModel(
        endpoint=endpoint, api_key=api_key, model_name=model_name,
        max_tokens=max_tokens_per_step, max_concurrency=-1,
    )
    done = _load_done(output)

    async def _process(alg, method, arm, budget, rec, out, sem, lock):
        msgs, seed = build(method, rec, audio_mode)
        if seed:
            raise ValueError(
                f"prompt method {method} uses an assistant seed; "
                "runner supports prompt-only CoT methods (e.g. 2,4,7)."
            )
        t0 = time.time()
        async with sem:
            try:
                resp = await alg.ainfer(lm, msgs, budget, return_response_only=True)
                content = extract_content_from_lm_response(resp)
                correct = is_correct(content, rec.choices, rec.answer_index)
                err = None
            except Exception as e:  # record, don't abort the sweep
                content, correct, err = "", None, f"{type(e).__name__}: {e}"
        row = {
            "unique_id": rec.unique_id, "method": method, "arm": arm, "budget": budget,
            "category": rec.category, "length_type": rec.length_type, "correct": correct,
            "latency_s": round(time.time() - t0, 2), "error": err, "content": content[:2000],
        }
        async with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    async def _run():
        sem = asyncio.Semaphore(item_concurrency)
        lock = asyncio.Lock()
        with open(output, "a") as out:
            for method in methods:
                for arm in arms:
                    arm_budgets = [1] if arm == "baseline" else budgets
                    for budget in arm_budgets:
                        alg = build_algorithm(arm, max_steps)
                        todo = [
                            r for r in records
                            if (r.unique_id, method, arm, budget) not in done
                        ]
                        print(
                            f"cell m{method} {arm}@{budget}: {len(todo)} to do "
                            f"({len(records) - len(todo)} already done)",
                            flush=True,
                        )
                        t0 = time.time()
                        await asyncio.gather(*(
                            _process(alg, method, arm, budget, rec, out, sem, lock)
                            for rec in todo
                        ))
                        if todo:
                            print(
                                f"  cell done in {time.time() - t0:.0f}s "
                                f"({(time.time() - t0) / len(todo):.1f}s/item)",
                                flush=True,
                            )
        await lm.close()

    asyncio.run(_run())
    # final report over the complete output (includes resumed rows); dedupe on
    # key keeping the latest row so a retried item's old errored row is ignored
    by_key = {}
    with open(output) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_key[(r["unique_id"], r["method"], r["arm"], r["budget"])] = r
    report(list(by_key.values()))


if __name__ == "__main__":
    main()
