"""Standalone probe: send one prompt to the served vLLM model and print, for
EVERY generated token, its log-probability and entropy — the exact quantities
PF/EPF self-certainty weighting is built on. Lets you see the signal by eye.

It reuses the real library pipeline (nothing bespoke about the numbers):
  OpenAICompatibleLanguageModel(logprobs=True, top_logprobs=K)  -> raw per-token logprobs
  its_hub.core.utils.summarize_step_logprobs(...)               -> the per-step aggregate
  ParticleFiltering._self_certainty_logweight(...)              -> the particle log-weight

The per-token entropy uses the same formula summarize_step_logprobs applies at
each position: H = -sum(p * ln p) over the returned top-k candidates (p = exp(logprob)).
It is a top-k *truncated* approximation (mass beyond the top-k is ignored).

Run (vLLM must be up on :8100 — see the launch command in
benchmarking/mmau_pro/RESULTS.md):

  uv run python test_vllm_probs.py
  uv run python test_vllm_probs.py --temperature 0.7   # sample non-top tokens:
                                                        # mean_logprob & entropy diverge
  uv run python test_vllm_probs.py --prompt "name a primary colour" --max-tokens 32
"""

import argparse
import asyncio
import math

from its_hub import OpenAICompatibleLanguageModel, ParticleFiltering, StepGeneration
from its_hub.core.utils import summarize_step_logprobs


def token_entropy(top_logprobs: list[dict]) -> float | None:
    """Top-k truncated entropy (nats) at one token position — identical to the
    per-position term inside summarize_step_logprobs: -sum(p * ln p) over the
    returned top-k candidates, with p = exp(logprob)."""
    lps = [e["logprob"] for e in top_logprobs if e.get("logprob") is not None]
    if not lps:
        return None
    return -sum(math.exp(lp) * lp for lp in lps)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://localhost:8100/v1")
    ap.add_argument("--model-name", default="qwen-omni")
    ap.add_argument("--api-key", default="NO_API_KEY")
    ap.add_argument("--prompt", default="what are the colours of a rainbow")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy (emitted token is the argmax; cleanest view). "
        "Raise it to watch the model sample non-top tokens.",
    )
    ap.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="candidate logprobs returned per position; entropy is computed over these",
    )
    return ap.parse_args()


async def main():
    args = parse_args()

    lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model_name=args.model_name,
    )
    messages = [{"role": "user", "content": args.prompt}]
    resp = await lm.agenerate_single(
        messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        logprobs=True,
        top_logprobs=args.top_logprobs,
    )
    await lm.close()

    content = resp.get("content") or ""
    logprobs = resp.get("_logprobs")

    print("=" * 78)
    print("PROMPT :", args.prompt)
    print(f"(model={args.model_name}  temp={args.temperature}  top_logprobs={args.top_logprobs})")
    print("=" * 78)
    print("OUTPUT :\n" + content)
    print("=" * 78)

    if not logprobs or not logprobs.get("content"):
        print(
            "No token logprobs returned — does the endpoint support logprobs=true? "
            "(vLLM does; some proxies strip the field.)"
        )
        return

    tokens = logprobs["content"]
    print(f"PER-TOKEN   p = exp(logprob) is prob of the EMITTED token;   "
          f"H = top-{args.top_logprobs} entropy (nats)\n")
    header = f"{'idx':>3}  {'token':<14} {'logprob':>8} {'p':>6} {'H':>7}   top-3 candidates (p)"
    print(header)
    print("-" * len(header))
    for i, t in enumerate(tokens):
        tok = t.get("token", "")
        lgp = t.get("logprob")
        p = math.exp(lgp) if lgp is not None else float("nan")
        tops = t.get("top_logprobs") or []
        h = token_entropy(tops)
        h_str = f"{h:7.3f}" if h is not None else "    n/a"
        alts = "  ".join(
            f"{repr(e.get('token', '')):>10}={math.exp(e['logprob']):.2f}"
            for e in tops[:3]
            if e.get("logprob") is not None
        )
        print(f"{i:>3}  {repr(tok)[:14]:<14} {lgp:8.3f} {p:6.3f} {h_str}   {alts}")

    # --- the exact library aggregate for this whole generation (one "step") ---
    summary = summarize_step_logprobs(logprobs)
    print("\n" + "=" * 78)
    print("STEP SUMMARY   (its_hub.core.utils.summarize_step_logprobs)")
    print(f"  num_tokens   = {summary['num_tokens']}")
    print(f"  mean_logprob = {summary['mean_logprob']:+.4f}   <- PF 'mean_logprob' signal")
    mean_entropy = summary["entropy"]
    if mean_entropy is not None:
        print(f"  mean entropy = {mean_entropy:.4f}   <- EPF 'entropy' signal")
    else:
        print("  mean entropy = None   (no top_logprobs returned)")

    # --- what particle log-weight this step would earn (logit style, the default) ---
    sg = StepGeneration(step_token="\n\n", max_steps=1)
    pf_mean = ParticleFiltering(sg=sg, self_certainty_signal="mean_logprob")
    pf_entropy = ParticleFiltering(sg=sg, self_certainty_signal="entropy")
    print("\nPARTICLE LOG-WEIGHT this step would earn (logit style):")
    print(f"  mean_logprob signal -> {pf_mean._self_certainty_logweight(summary):+.4f}")
    print(f"  entropy signal      -> {pf_entropy._self_certainty_logweight(summary):+.4f}")
    print(
        "\n(Higher = the resampler favours this particle. mean_logprob rewards "
        "high-probability\n text; entropy rewards a peaked next-token distribution "
        "regardless of what was emitted.)"
    )


if __name__ == "__main__":
    asyncio.run(main())
