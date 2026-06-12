"""
self_certainty_demo.py — particle weights from the GENERATOR's own logprobs (no PRM).

Runnable companion to the audio inference-time-scaling experiment (Part 2 of the plan).
No GPU / server / API key — uses the real ParticleFiltering with a mock LM that emits
OpenAI-style logprobs.

    /home/exx/miniconda3/envs/epf/bin/python documentation/snippets/self_certainty_demo.py
    # or:  conda run -n epf python documentation/snippets/self_certainty_demo.py

It shows:
  PART 1 — a step's token logprobs -> one scalar (mean logprob / entropy) -> a particle
           log-weight, via the two transform styles:
             (i)  'logit': s = exp(mean_logprob) in (0,1], then _inv_sigmoid(s)
             (ii) 'raw'  : use the mean logprob directly as the log-weight
  PART 2 — the REAL ParticleFiltering running end-to-end (self-certainty is the only
           weight source), weighting/resampling particles purely from the generator's
           confidence.
"""

import math

from its_hub import AbstractLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_filtering import ParticleFiltering, _inv_sigmoid
from its_hub.core.utils import summarize_step_logprobs


def part1_signal_to_weight() -> None:
    print("=" * 72)
    print("PART 1 — step logprobs  ->  scalar signal  ->  particle log-weight")
    print("=" * 72)

    # A made-up step of 3 tokens with per-token logprobs + top-2 candidates.
    logprobs = {
        "content": [
            {"logprob": -0.10, "top_logprobs": [{"logprob": -0.10}, {"logprob": -2.0}]},
            {"logprob": -0.40, "top_logprobs": [{"logprob": -0.40}, {"logprob": -1.6}]},
            {"logprob": -0.25, "top_logprobs": [{"logprob": -0.25}, {"logprob": -1.9}]},
        ]
    }
    s = summarize_step_logprobs(logprobs)
    print(f"  mean_logprob = {s['mean_logprob']:.4f}   entropy = {s['entropy']:.4f}   "
          f"num_tokens = {s['num_tokens']}")

    mean_lp = s["mean_logprob"]
    print("\n  signal = mean_logprob:")
    print(f"    (ii) raw   -> log-weight = {mean_lp:.4f}")
    s_conf = math.exp(min(mean_lp, 0.0))
    print(f"    (i)  logit -> s = exp(mean_logprob) = {s_conf:.4f} -> "
          f"log-weight = {_inv_sigmoid(s_conf):.4f}")

    ent = s["entropy"]
    print("\n  signal = entropy (c = -entropy):")
    print(f"    (ii) raw   -> log-weight = {-ent:.4f}")
    s_conf_e = math.exp(min(-ent, 0.0))
    print(f"    (i)  logit -> s = exp(-entropy) = {s_conf_e:.4f} -> "
          f"log-weight = {_inv_sigmoid(s_conf_e):.4f}")
    print("\n  (probability s in (0,1]  vs  log-weight in R that softmax resamples on.)")


class _LogprobMockLM(AbstractLanguageModel):
    """Emits 'step k' plus per-token logprobs; cycles a few target mean logprobs."""

    def __init__(self, means=(-0.1, -0.5, -1.0, -0.2)):
        self.means = list(means)
        self.n = 0

    def _msg(self, idx, want_lp, want_top):
        base = self.means[idx % len(self.means)]
        m = {"role": "assistant", "content": f"step{idx}"}
        if want_lp:
            tok = {"token": "t", "logprob": base}
            if want_top is not None:
                tok["top_logprobs"] = [{"logprob": base}, {"logprob": base - 1.0}]
            m["_logprobs"] = {"content": [tok, dict(tok)]}
        return m

    async def agenerate(self, messages, logprobs=False, top_logprobs=None, **kwargs):
        is_batch = isinstance(messages, list) and messages and isinstance(messages[0], list)
        if is_batch:
            out = []
            for _ in messages:
                out.append(self._msg(self.n, logprobs, top_logprobs))
                self.n += 1
            return out
        m = self._msg(self.n, logprobs, top_logprobs)
        self.n += 1
        return m

    async def agenerate_single(self, messages, **kwargs):
        return await self.agenerate(messages, **kwargs)


def part2_real_particle_filtering() -> None:
    print("\n" + "=" * 72)
    print("PART 2 — real ParticleFiltering (self-certainty weights), no reward model")
    print("=" * 72)

    sg = StepGeneration(step_token="\n", max_steps=3)
    pf = ParticleFiltering(
        sg=sg,
        self_certainty_signal="mean_logprob",
        self_certainty_style="logit",
    )
    result = pf.infer(_LogprobMockLM(), "Solve it.", budget=4, return_response_only=False)

    print(f"  budget (particles)     : 4   (no reward model used)")
    print(f"  final log-weights      : {[round(float(w), 4) for w in result.log_weights_lst]}")
    print(f"  selected_index (argmax): {result.selected_index}")
    print(f"  the_one.content        : {result.the_one['content']!r}")
    print("\n  Weights came entirely from the generator's own step logprobs.")


if __name__ == "__main__":
    part1_signal_to_weight()
    part2_real_particle_filtering()
