"""A/B causality check: does the model actually *hear* the audio?

For each item, do one greedy generation WITH the audio and one with the audio
parts removed (text-only). If the carry works and the model uses the audio, the
audio-present run should be more accurate and many answers should change.

    python -m benchmarking.mmau_pro.ab_causality \
        --endpoint http://localhost:8100/v1 --model-name qwen-omni \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --limit 15
"""

import asyncio

import click

from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.prompt import build_messages
from benchmarking.mmau_pro.scoring import is_correct, predicted_index
from its_hub import OpenAICompatibleLanguageModel
from its_hub.api.types import ChatMessage
from its_hub.core.utils import extract_content_from_lm_response


def _strip_audio(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return a copy with audio parts removed from the user turn (text only)."""
    out = []
    for m in messages:
        if isinstance(m.content, list):
            text_only = [p for p in m.content if p.get("type") == "text"]
            out.append(ChatMessage(role=m.role, content=text_only))
        else:
            out.append(m)
    return out


@click.command()
@click.option("--endpoint", required=True)
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--limit", default=15)
@click.option("--max-tokens", default=512)
def main(endpoint, model_name, api_key, data_root, limit, max_tokens):
    records = load_mmau_mcq(data_root, subset="le30s", limit=limit)
    lm = OpenAICompatibleLanguageModel(endpoint=endpoint, api_key=api_key, model_name=model_name)

    async def _one(messages):
        resp = await lm.agenerate_single(messages, max_tokens=max_tokens, temperature=0.0)
        return extract_content_from_lm_response(resp)

    async def _run():
        w_correct = wo_correct = changed = gradeable = 0
        try:
            for rec in records:
                msgs = build_messages(rec, audio_mode="base64")
                with_txt = await _one(msgs)
                without_txt = await _one(_strip_audio(msgs))
                cw = is_correct(with_txt, rec.choices, rec.answer_index)
                co = is_correct(without_txt, rec.choices, rec.answer_index)
                pi_w = predicted_index(with_txt, rec.choices)
                pi_o = predicted_index(without_txt, rec.choices)
                if cw is None:
                    continue
                gradeable += 1
                w_correct += int(bool(cw))
                wo_correct += int(bool(co))
                changed += int(pi_w != pi_o)
                print(
                    f"[{rec.category:12s}] with={'OK ' if cw else 'x  '}(pick {pi_w}) "
                    f"without={'OK ' if co else 'x  '}(pick {pi_o}) gold={rec.answer_index} "
                    f":: {rec.question[:60]}"
                )
        finally:
            await lm.close()
        print("\n=== A/B causality summary ===")
        print(f"  gradeable items     : {gradeable}")
        print(f"  acc WITH audio      : {w_correct}/{gradeable} = {w_correct / max(gradeable,1):.3f}")
        print(f"  acc WITHOUT audio   : {wo_correct}/{gradeable} = {wo_correct / max(gradeable,1):.3f}")
        print(f"  answers changed     : {changed}/{gradeable}")
        print("  (audio-present should be higher & many answers should change => model hears it)")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
