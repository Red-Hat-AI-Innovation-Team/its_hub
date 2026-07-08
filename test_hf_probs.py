"""Companion to test_vllm_probs.py — SAME prompt, SAME weights, but run through
raw HuggingFace Transformers instead of the vLLM server, so you can diff the
per-token logprobs/entropy between the two backends.

Qwen2.5-Omni is an Omni (thinker+talker) model; text generation is the THINKER,
which is a normal causal LM. We load the full model, grab `.thinker`, and
greedy-decode one token at a time computing logprobs/entropy straight from the
logits — no reliance on the Omni-specific generate() plumbing.

The per-token entropy, the step summary, and the particle weight all go through
the SAME its_hub functions the vLLM probe uses (summarize_step_logprobs,
ParticleFiltering._self_certainty_logweight), so the ONLY difference vs the vLLM
run is the backend that produced the logits.

Must run in the `epf` conda env (has torch + transformers + the model) and on a
free GPU. The vLLM server holds GPU 0, so use GPU 1:

  CUDA_VISIBLE_DEVICES=1 \
  HF_HOME=/media/exx/68031955-bdaa-4e71-9687-916b9876dfc6/hf_cache \
    /home/exx/miniconda3/envs/epf/bin/python test_hf_probs.py

  ... --temperature 0.7   # (greedy still; temperature only rescales the reported
                          #  logprobs/entropy, mirroring the vLLM knob)
"""

import argparse
import math

import torch
from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration

from its_hub import ParticleFiltering, StepGeneration
from its_hub.core.utils import summarize_step_logprobs

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--prompt", default="what are the colours of a rainbow")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = report raw (temperature-1) logprobs, matching vLLM greedy. "
        ">0 rescales logits by 1/T before softmax (decode stays greedy).",
    )
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    print(f"loading {args.model_id} (thinker) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    thinker = model.thinker  # the text/reasoning causal LM

    # Same chat template the vLLM server applies (inserts the model's default
    # system turn, then the user turn, then the assistant generation prompt).
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(args.device)

    # Stop tokens: end the assistant turn like the server does.
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        stop_ids.add(im_end)

    scale = 1.0 / args.temperature if args.temperature and args.temperature > 0 else 1.0

    # Greedy decode, no KV cache (cheap for a short answer; keeps it robust).
    full_ids = input_ids
    records = []  # (token_id, chosen_logprob, [(cand_id, cand_logprob), ...])
    for _ in range(args.max_tokens):
        out = thinker(input_ids=full_ids)
        logits = out.logits[:, -1, :].float().squeeze(0)  # [vocab]
        logprobs = torch.log_softmax(logits * scale, dim=-1)
        next_id = int(torch.argmax(logprobs))  # greedy decode (argmax)
        chosen_lp = float(logprobs[next_id])
        topk = torch.topk(logprobs, k=args.top_logprobs)
        cands = list(zip(topk.indices.tolist(), topk.values.tolist()))
        records.append((next_id, chosen_lp, cands))
        full_ids = torch.cat(
            [full_ids, torch.tensor([[next_id]], device=args.device)], dim=1
        )
        if next_id in stop_ids:
            break

    gen_ids = [r[0] for r in records]
    output_text = tokenizer.decode(
        [t for t in gen_ids if t not in stop_ids], skip_special_tokens=True
    )

    print("=" * 78)
    print("PROMPT :", args.prompt)
    print(f"(backend=HF transformers  model={args.model_id}  temp={args.temperature}  "
          f"top_logprobs={args.top_logprobs})")
    print("=" * 78)
    print("OUTPUT :\n" + output_text)
    print("=" * 78)

    # Build the OpenAI-style logprobs dict so we can reuse the EXACT same
    # aggregation the vLLM probe uses (its_hub.core.utils.summarize_step_logprobs).
    oai_logprobs = {
        "content": [
            {
                "token": tokenizer.decode([tid]),
                "logprob": lp,
                "top_logprobs": [
                    {"token": tokenizer.decode([cid]), "logprob": clp}
                    for cid, clp in cands
                ],
            }
            for (tid, lp, cands) in records
        ]
    }

    print(f"PER-TOKEN   p = exp(logprob) of the EMITTED token;   "
          f"H = top-{args.top_logprobs} entropy (nats)\n")
    header = f"{'idx':>3}  {'token':<14} {'logprob':>8} {'p':>6} {'H':>7}   top-3 candidates (p)"
    print(header)
    print("-" * len(header))
    for i, entry in enumerate(oai_logprobs["content"]):
        tok = entry["token"]
        lgp = entry["logprob"]
        p = math.exp(lgp)
        tops = entry["top_logprobs"]
        lps = [e["logprob"] for e in tops]
        h = -sum(math.exp(lp) * lp for lp in lps) if lps else None
        h_str = f"{h:7.3f}" if h is not None else "    n/a"
        alts = "  ".join(
            f"{repr(e['token']):>10}={math.exp(e['logprob']):.2f}" for e in tops[:3]
        )
        print(f"{i:>3}  {repr(tok)[:14]:<14} {lgp:8.3f} {p:6.3f} {h_str}   {alts}")

    summary = summarize_step_logprobs(oai_logprobs)
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
