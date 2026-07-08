"""Third backend: vLLM's OFFLINE (in-process) engine — `from vllm import LLM`.

This is the SAME vLLM engine the HTTP server runs, but accessed directly in
Python instead of over HTTP. It is the layer documented in
vllm/v1/engine/logprobs.py (SampleLogprobs / Logprob objects): the HTTP server
runs exactly this and then serializes the result into OpenAI JSON. So this
should match test_vllm_probs.py (HTTP) essentially bit-for-bit — same engine.

Greedy (temperature 0), same prompt, same its_hub aggregation, so it lines up
with the other two probes.

Run on a FREE gpu (the HTTP server holds GPU 0). Blackwell needs the flashinfer
sampler off, same as the server:

  CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  HF_HOME=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache \
    /home/exx/miniconda3/envs/epf/bin/python test_vllm_native_probs.py
"""

import argparse
import math

from vllm import LLM, SamplingParams

from its_hub import ParticleFiltering, StepGeneration
from its_hub.core.utils import summarize_step_logprobs

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--prompt", default="what are the colours of a rainbow")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=32768,
        gpu_memory_utilization=args.gpu_mem,
    )
    sp = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=args.max_tokens,
        logprobs=args.top_logprobs,  # top-k logprobs per generated position
    )

    out = llm.chat([{"role": "user", "content": args.prompt}], sp)
    completion = out[0].outputs[0]

    # Build the SAME OpenAI-style logprobs dict the HTTP path produces, so we can
    # reuse its_hub.core.utils.summarize_step_logprobs unchanged. vLLM gives us,
    # per generated position, a dict {token_id: Logprob(logprob, rank, decoded_token)}.
    content = []
    for pos, tid in enumerate(completion.token_ids):
        lp_dict = completion.logprobs[pos]
        sampled = lp_dict[tid]
        ranked = sorted(lp_dict.values(), key=lambda lp: lp.logprob, reverse=True)
        content.append({
            "token": sampled.decoded_token,
            "logprob": sampled.logprob,
            "top_logprobs": [
                {"token": lp.decoded_token, "logprob": lp.logprob} for lp in ranked
            ],
        })
    logprobs = {"content": content}

    print("\n" + "=" * 78)
    print("PROMPT :", args.prompt)
    print(f"(backend=vLLM OFFLINE (in-process LLM)  model={args.model_id}  "
          f"greedy  top_logprobs={args.top_logprobs})")
    print("=" * 78)
    print("OUTPUT :\n" + completion.text)
    print("=" * 78)

    print(f"PER-TOKEN   p = exp(logprob) of the EMITTED token;   "
          f"H = top-{args.top_logprobs} entropy (nats)\n")
    header = f"{'idx':>3}  {'token':<14} {'logprob':>8} {'p':>6} {'H':>7}   top-3 candidates (p)"
    print(header)
    print("-" * len(header))
    for i, entry in enumerate(content):
        lgp = entry["logprob"]
        p = math.exp(lgp)
        tops = entry["top_logprobs"]
        lps = [e["logprob"] for e in tops]
        h = -sum(math.exp(x) * x for x in lps) if lps else None
        h_str = f"{h:7.3f}" if h is not None else "    n/a"
        alts = "  ".join(
            f"{repr(e['token']):>10}={math.exp(e['logprob']):.2f}" for e in tops[:3]
        )
        print(f"{i:>3}  {repr(entry['token'])[:14]:<14} {lgp:8.3f} {p:6.3f} {h_str}   {alts}")

    summary = summarize_step_logprobs(logprobs)
    print("\n" + "=" * 78)
    print("STEP SUMMARY   (its_hub.core.utils.summarize_step_logprobs)")
    print(f"  num_tokens   = {summary['num_tokens']}")
    print(f"  mean_logprob = {summary['mean_logprob']:+.4f}   <- PF 'mean_logprob' signal")
    me = summary["entropy"]
    print(f"  mean entropy = {me:.4f}   <- EPF 'entropy' signal" if me is not None
          else "  mean entropy = None")

    sg = StepGeneration(step_token="\n\n", max_steps=1)
    pf_mean = ParticleFiltering(sg=sg, self_certainty_signal="mean_logprob")
    pf_entropy = ParticleFiltering(sg=sg, self_certainty_signal="entropy")
    print("\nPARTICLE LOG-WEIGHT this step would earn (logit style):")
    print(f"  mean_logprob signal -> {pf_mean._self_certainty_logweight(summary):+.4f}")
    print(f"  entropy signal      -> {pf_entropy._self_certainty_logweight(summary):+.4f}")


if __name__ == "__main__":
    main()
