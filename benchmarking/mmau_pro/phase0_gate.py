"""Phase-0 go/no-go gates against a live Qwen2.5-Omni vLLM endpoint (needs GPU).

GATE 1: generated-token logprobs are returned WHEN the input includes audio.
GATE 2: continue_final_message works with an audio user turn (prefill + continue).

Run (after `vllm serve ... --allowed-local-media-path <data_root> --limit-mm-per-prompt audio=3`):

    python -m benchmarking.mmau_pro.phase0_gate \
        --endpoint http://localhost:8100/v1 --model-name <served-name> \
        --data-root /home/exx/inference-time-scaling/mmau_pro_testmini
"""

import asyncio
import json
import os

import click

from benchmarking.mmau_pro.audio import audio_content_parts
from benchmarking.mmau_pro.loader import load_mmau_mcq
from its_hub import OpenAICompatibleLanguageModel
from its_hub.api.types import ChatMessage


async def _gate1(lm, audio_parts) -> bool:
    user = ChatMessage(
        role="user",
        content=[*audio_parts, {"type": "text", "text": "In one word, what do you hear?"}],
    )
    resp = await lm.agenerate_single([user], max_tokens=32, logprobs=True, top_logprobs=20)
    lp = resp.get("_logprobs")
    ok = bool(lp and lp.get("content"))
    print(f"GATE 1 (logprobs WITH audio): {'PASS' if ok else 'FAIL'}")
    if ok:
        sample = lp["content"][0]
        print(f"  first token: {sample.get('token')!r} logprob={sample.get('logprob')} "
              f"top_logprobs={len(sample.get('top_logprobs') or [])}")
    else:
        print("  no generated-token logprobs returned -> use fallback ladder (see plan)")
    return ok


async def _gate2(lm, audio_parts) -> bool:
    user = ChatMessage(
        role="user",
        content=[*audio_parts, {"type": "text", "text": "Describe the audio step by step."}],
    )
    partial = ChatMessage(role="assistant", content="## Step 1: I first notice")
    resp = await lm.agenerate_single([user, partial], max_tokens=64)
    cont = resp.get("content") or ""
    ok = bool(cont.strip())
    print(f"GATE 2 (continue_final_message WITH audio): {'PASS' if ok else 'FAIL'}")
    print(f"  continuation: {cont[:160]!r}")
    return ok


@click.command()
@click.option("--endpoint", required=True)
@click.option("--model-name", required=True)
@click.option("--api-key", default="NO_API_KEY")
@click.option("--data-root", default="/home/exx/inference-time-scaling/mmau_pro_testmini")
@click.option("--audio-mode", type=click.Choice(["local-path", "base64"]), default="local-path")
def main(endpoint, model_name, api_key, data_root, audio_mode):
    rec = load_mmau_mcq(data_root, subset="le30s", limit=1)[0]
    print(f"using audio: {[os.path.basename(p) for p in rec.audio_paths]} (mode={audio_mode})")
    audio_parts = audio_content_parts(rec.audio_paths, mode=audio_mode)
    lm = OpenAICompatibleLanguageModel(endpoint=endpoint, api_key=api_key, model_name=model_name)

    async def _run():
        try:
            g1 = await _gate1(lm, audio_parts)
            g2 = await _gate2(lm, audio_parts)
        finally:
            await lm.close()
        print(f"\nRESULT: {json.dumps({'gate1_logprobs': g1, 'gate2_continue': g2})}")
        if not g1:
            print("GATE 1 failed -> self-certainty needs the fallback ladder before runs.")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
