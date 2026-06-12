"""
epf_logweights_demo.py — see exactly how a self-certainty confidence becomes a particle log-weight.

This is the runnable companion to documentation/07-particle-filtering.md. It needs NO GPU,
NO model server, and NO API key — just `numpy` and `its_hub` (the base install).

Run it with the project's conda env:

    /home/exx/miniconda3/envs/epf/bin/python documentation/snippets/epf_logweights_demo.py
    # or:  conda run -n epf python documentation/snippets/epf_logweights_demo.py

It does two things:
  PART 1 — the math, deterministically: self-certainty confidence s = exp(mean step logprob)
           -> logit(s) -> softmax over particles, and verifies that the resample probability
           of particle i is proportional to its ODDS  s/(1-s).  (This is the heart of the
           "multiply vs add" answer.)
  PART 2 — the real algorithm end-to-end with a mock LM that emits token logprobs (no GPU),
           printing the particle log-weights and the selected particle.
"""

import math
import random

import numpy as np

# The REAL helper functions used by the particle filter — we import them so the demo
# is guaranteed to match the library's behavior, not a re-implementation.
from its_hub.core.algorithms.particle_filtering import (
    ParticleFiltering,
    _inv_sigmoid,  # logit:  log(s / (1 - s))
    _softmax,
)
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.api import AbstractLanguageModel


def part1_the_math() -> None:
    print("=" * 72)
    print("PART 1 — confidence s  ->  logit (log-weight)  ->  softmax over particles")
    print("=" * 72)

    # Suppose the generator's own logprobs gave 4 particles these per-step confidences
    # (s = exp(mean step logprob), so s in (0, 1], higher = more certain):
    scores = np.array([0.20, 0.50, 0.70, 0.90])

    # Step A: each confidence becomes a LOG-WEIGHT via the logit (inverse sigmoid).
    log_weights = np.array([_inv_sigmoid(s) for s in scores])

    # Step B: softmax over particles turns log-weights into resample probabilities.
    probs = _softmax(log_weights)

    # Claim from Chapter 7: probs[i] is proportional to the ODDS  s/(1-s).
    odds = scores / (1.0 - scores)
    odds_normalized = odds / odds.sum()

    print(f"{'particle':>9} {'conf s':>9} {'logit(s)=w':>12} {'softmax p':>11} {'odds/Σodds':>12}")
    for i in range(len(scores)):
        print(f"{i:>9} {scores[i]:>9.3f} {log_weights[i]:>12.4f} {probs[i]:>11.4f} {odds_normalized[i]:>12.4f}")

    assert np.allclose(probs, odds_normalized), "softmax(logit(s)) must equal normalized odds"
    print("\n  ✓ softmax(logit(s)) == normalized odds  ->  resample prob ∝ s/(1-s)")
    print("  Across STEPS the filter does NOT sum or multiply these; it re-derives w each")
    print("  step from THAT step's logprob summary alone (Particle.log_weight is the most")
    print("  recent entry), and resampling softmaxes those latest weights across particles.")


# --- Minimal mock so the real algorithm can run with no GPU / no server --------------------

class MockLM(AbstractLanguageModel):
    """Emits a deterministic 'step k' each call, with per-token logprobs whose mean
    cycles a preset list of confidences — so particles differ."""

    def __init__(self) -> None:
        self.preset = [0.30, 0.55, 0.70, 0.85, 0.40, 0.60]  # s = exp(mean step logprob)
        self.counter = 0

    def _one(self, want_logprobs: bool, top_logprobs) -> dict:
        s = self.preset[self.counter % len(self.preset)]
        msg = {"role": "assistant", "content": f"step{self.counter}"}
        self.counter += 1
        if want_logprobs:
            tok = {"token": "t", "logprob": math.log(s)}
            if top_logprobs is not None:
                tok["top_logprobs"] = [{"logprob": math.log(s)}, {"logprob": math.log(s) - 1.0}]
            msg["_logprobs"] = {"content": [tok, dict(tok)]}
        return msg

    async def agenerate_single(self, messages, logprobs=False, top_logprobs=None, **kwargs) -> dict:
        return self._one(logprobs, top_logprobs)

    async def agenerate(self, messages, logprobs=False, top_logprobs=None, **kwargs):
        # batch path: messages is a list of conversations (list of lists)
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [self._one(logprobs, top_logprobs) for _ in messages]
        return self._one(logprobs, top_logprobs)


def part2_real_algorithm() -> None:
    print("\n" + "=" * 72)
    print("PART 2 — run the REAL ParticleFiltering with a logprob-emitting mock LM (no GPU)")
    print("=" * 72)

    random.seed(0)
    np.random.seed(0)

    sg = StepGeneration(step_token="\n", max_steps=3)
    # defaults: self_certainty_signal='mean_logprob', self_certainty_style='logit'
    # — i.e. exactly the PART-1 math, with s = exp(mean step logprob).
    pf = ParticleFiltering(sg=sg)

    budget = 4  # = number of particles
    result = pf.infer(MockLM(), "Solve: 2 + 2 = ?", budget=budget, return_response_only=False)

    print(f"  budget (num particles) : {budget}")
    print(f"  final log-weights      : {[round(float(w), 4) for w in result.log_weights_lst]}")
    print(f"  steps used per particle: {result.steps_used_lst}")
    print(f"  selected_index (argmax): {result.selected_index}")
    print(f"  the_one.content        : {result.the_one['content']!r}")
    print("\n  Note: log_weights_lst holds each surviving particle's LATEST log-weight")
    print("  (Particle.log_weight == partial_log_weights[-1]); selection is argmax over them.")


if __name__ == "__main__":
    part1_the_math()
    part2_real_algorithm()
