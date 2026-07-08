"""Run the SAME prompt through BOTH backends and print them side by side:
  - left  column: vLLM server  (OpenAI-compatible, over HTTP)
  - right column: raw HuggingFace Transformers (local, this GPU)

Same weights, same greedy decode, same its_hub aggregation
(summarize_step_logprobs) — so any per-token difference is purely
vLLM-kernels vs HF-kernels. Rows where the two backends emit a DIFFERENT token
are flagged (that's where a high-entropy near-tie tips opposite ways).

Runs under the `epf` conda env (needs torch+transformers for HF AND aiohttp for
the vLLM client). vLLM holds GPU 0, so put HF on GPU 1:

  CUDA_VISIBLE_DEVICES=1 \
  HF_HOME=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache \
    /home/exx/miniconda3/envs/epf/bin/python test_compare_probs.py

Both sides decode greedily (temperature 0). --max-tokens caps length.
"""

import argparse
import asyncio
import math

import torch
from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration

from its_hub import OpenAICompatibleLanguageModel, ParticleFiltering, StepGeneration
from its_hub.core.utils import summarize_step_logprobs

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://localhost:8100/v1")
    ap.add_argument("--model-name", default="qwen-omni", help="vLLM served name")
    ap.add_argument("--api-key", default="NO_API_KEY")
    ap.add_argument("--model-id", default=MODEL_ID, help="HF model id")
    ap.add_argument("--prompt", default="what are the colours of a rainbow")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def entropy_of(top_logprobs: list[dict]) -> float | None:
    lps = [e["logprob"] for e in top_logprobs if e.get("logprob") is not None]
    if not lps:
        return None
    return -sum(math.exp(lp) * lp for lp in lps)


# --------------------------------------------------------------------------- #
# vLLM backend (HTTP)                                                          #
# --------------------------------------------------------------------------- #
async def run_vllm(args) -> tuple[str, dict]:
    lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint, api_key=args.api_key, model_name=args.model_name
    )
    resp = await lm.agenerate_single(
        [{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=0.0,
        logprobs=True,
        top_logprobs=args.top_logprobs,
    )
    await lm.close()
    return resp.get("content") or "", resp.get("_logprobs") or {"content": []}


# --------------------------------------------------------------------------- #
# HF backend (local, greedy)                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_hf(args) -> tuple[str, dict]:
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device)
    model.eval()
    thinker = model.thinker

    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(args.device)

    stop_ids = set()
    if tok.eos_token_id is not None:
        stop_ids.add(tok.eos_token_id)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)

    full_ids, content = input_ids, []
    for _ in range(args.max_tokens):
        logits = thinker(input_ids=full_ids).logits[:, -1, :].float().squeeze(0)
        logprobs = torch.log_softmax(logits, dim=-1)
        nid = int(torch.argmax(logprobs))
        topk = torch.topk(logprobs, k=args.top_logprobs)
        content.append({
            "token": tok.decode([nid]),
            "logprob": float(logprobs[nid]),
            "top_logprobs": [
                {"token": tok.decode([c]), "logprob": v}
                for c, v in zip(topk.indices.tolist(), topk.values.tolist())
            ],
        })
        full_ids = torch.cat([full_ids, torch.tensor([[nid]], device=args.device)], 1)
        if nid in stop_ids:
            break

    text = "".join(
        c["token"] for c in content if c["token"] not in ("<|im_end|>",)
    ).strip()
    return text, {"content": content}


# --------------------------------------------------------------------------- #
# side-by-side print                                                          #
# --------------------------------------------------------------------------- #
def fmt_side(entry) -> str:
    if entry is None:
        return f"{'—':<15} {'':>7} {'':>6}"
    h = entropy_of(entry["top_logprobs"])
    h_str = f"{h:6.2f}" if h is not None else "   n/a"
    return f"{repr(entry['token'])[:15]:<15} {entry['logprob']:7.3f} {h_str}"


def summary_line(logprobs, tag):
    s = summarize_step_logprobs(logprobs)
    sg = StepGeneration(step_token="\n\n", max_steps=1)
    w_ml = ParticleFiltering(sg=sg, self_certainty_signal="mean_logprob")._self_certainty_logweight(s)
    w_en = ParticleFiltering(sg=sg, self_certainty_signal="entropy")._self_certainty_logweight(s)
    me = f"{s['entropy']:.4f}" if s["entropy"] is not None else "None"
    return (f"  {tag:5s}  tokens={s['num_tokens']:3d}  mean_logprob={s['mean_logprob']:+.4f}  "
            f"mean_entropy={me}  |  weight(ml)={w_ml:+.3f}  weight(ent)={w_en:+.3f}")


def main():
    args = parse_args()

    print("querying vLLM server ...", flush=True)
    vllm_text, vllm_lp = asyncio.run(run_vllm(args))
    print(f"loading {args.model_id} for HF backend ...", flush=True)
    hf_text, hf_lp = run_hf(args)

    print("\n" + "=" * 100)
    print("PROMPT :", args.prompt, "   (both greedy, temp 0)")
    print("=" * 100)
    print("vLLM OUTPUT:\n  " + vllm_text.replace("\n", "\n  "))
    print("\nHF   OUTPUT:\n  " + hf_text.replace("\n", "\n  "))
    print("=" * 100)

    v, h = vllm_lp["content"], hf_lp["content"]
    print(f"PER-TOKEN  (logprob, H = top-{args.top_logprobs} entropy in nats)   "
          f"'differ' = the two backends emitted a different token here\n")
    left = f"{'vLLM token':<15} {'logp':>7} {'H':>6}"
    right = f"{'HF token':<15} {'logp':>7} {'H':>6}"
    print(f"{'idx':>3}  {left}   |   {right}")
    print("-" * 100)
    for i in range(max(len(v), len(h))):
        ve = v[i] if i < len(v) else None
        he = h[i] if i < len(h) else None
        note = ""
        if ve and he and ve["token"] != he["token"]:
            note = "   <-- differ"
        print(f"{i:>3}  {fmt_side(ve)}   |   {fmt_side(he)}{note}")

    print("\n" + "=" * 100)
    print("STEP SUMMARY + particle log-weight (its_hub, identical for both):")
    print(summary_line(vllm_lp, "vLLM"))
    print(summary_line(hf_lp, "HF"))


if __name__ == "__main__":
    main()
