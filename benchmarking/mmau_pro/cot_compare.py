"""Compare 8 CoT-elicitation prompts on Qwen2.5-Omni / MMAU-Pro MCQ.

For each prompt method, run ONE generation per item (greedy) and measure:
  - reasoned?   (words of reasoning before the final 'Answer:')
  - chunks       (\\n\\n-separated non-empty segments => how well PF can chunk it)
  - accuracy     (extracted letter vs gold)

This isolates "which prompt makes the model actually reason (chunkably)" from the
PF step machinery. Pick the winner, then PF/EPF chunk it via StepGeneration.

    python -m benchmarking.mmau_pro.cot_compare \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --limit 20
"""

import asyncio
import os
import re

import click

from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import METHODS, build
from benchmarking.mmau_pro.scoring import predicted_index
from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.utils import extract_content_from_lm_response


def analyze(full_text: str, rec):
    pi = predicted_index(full_text, rec.choices)
    correct = (pi == rec.answer_index) if rec.answer_index is not None else None
    reasoning = re.split(r"answer\s*[:\-]", full_text, flags=re.IGNORECASE)[0]
    n_words = len(reasoning.split())
    n_chunks = len([p for p in full_text.split("\n\n") if p.strip()])
    return correct, n_words >= 15, n_words, n_chunks


def _select_items(records, limit):
    single = [r for r in records if len(r.audio_paths) == 1]
    single.sort(key=lambda r: os.path.getsize(r.audio_paths[0]))  # smallest audio first = fast
    return single[:limit]


@click.command()
@click.option("--endpoint", required=True)
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--limit", default=20)
@click.option("--max-tokens", default=700)
@click.option("--concurrency", default=6)
def main(endpoint, model_name, api_key, data_root, limit, max_tokens, concurrency):
    records = _select_items(load_mmau_mcq(data_root, subset="le30s"), limit)
    print(f"comparing {len(METHODS)} prompts over {len(records)} single-audio le30s MCQ items\n")
    lm = OpenAICompatibleLanguageModel(endpoint=endpoint, api_key=api_key, model_name=model_name)
    sem = asyncio.Semaphore(concurrency)

    async def _gen(method, rec):
        msgs, seed = build(method, rec, audio_mode="base64")
        async with sem:
            resp = await lm.agenerate_single(msgs, max_tokens=max_tokens, temperature=0.0)
        text = extract_content_from_lm_response(resp)
        if seed:
            text = seed + " " + text
        return analyze(text, rec)

    async def _run():
        results = {}
        try:
            for m in METHODS:
                outs = await asyncio.gather(*(_gen(m, r) for r in records))
                graded = [o for o in outs if o[0] is not None]
                acc = sum(o[0] for o in graded) / max(len(graded), 1)
                reasoned = sum(o[1] for o in outs) / len(outs)
                words = sum(o[2] for o in outs) / len(outs)
                chunks = sum(o[3] for o in outs) / len(outs)
                results[m] = (acc, reasoned, words, chunks, len(graded))
                print(f"[{m}] {METHODS[m]:32s} done")
        finally:
            await lm.close()

        print(f"\n{'#':>2} {'method':33s} {'acc':>6} {'reasoned':>9} {'avg_words':>10} {'avg_chunks':>11}")
        for m, (acc, reasoned, words, chunks, _n) in results.items():
            print(f"{m:>2} {METHODS[m]:33s} {acc:6.3f} {reasoned:9.2f} {words:10.1f} {chunks:11.1f}")
        # rank: must reason (>=0.8 reasoned) & chunk (>=2), then accuracy
        ok = {m: v for m, v in results.items() if v[1] >= 0.8 and v[3] >= 2.0}
        pool = ok or results
        best = max(pool, key=lambda m: (pool[m][0], pool[m][3]))
        print(f"\nBEST (reasons + chunkable + accurate): [{best}] {METHODS[best]} "
              f"acc={results[best][0]:.3f} reasoned={results[best][1]:.2f} chunks={results[best][3]:.1f}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
