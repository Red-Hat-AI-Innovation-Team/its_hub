"""
epf_logweights_demo.py — see exactly how a PRM score becomes a particle log-weight.

This is the runnable companion to documentation/07-particle-filtering.md. It needs NO GPU,
NO model server, and NO API key — just `numpy` and `its_hub` (the base install).

Run it with the project's conda env:

    /home/exx/miniconda3/envs/epf/bin/python documentation/snippets/epf_logweights_demo.py
    # or:  conda run -n epf python documentation/snippets/epf_logweights_demo.py

It does two things:
  PART 1 — the math, deterministically: PRM score s -> logit(s) -> softmax over particles,
           and verifies that the resample probability of particle i is proportional to its
           ODDS  s/(1-s).  (This is the heart of the "multiply vs add" answer.)
  PART 2 — the real algorithm end-to-end with a mock LM + mock PRM (no GPU), printing the
           particle log-weights and the selected particle.
"""

import random

import numpy as np

# The REAL helper functions used by the particle filter — we import them so the demo
# is guaranteed to match the library's behavior, not a re-implementation.
from its_hub.core.algorithms.particle_gibbs import (
    ParticleFiltering,
    _inv_sigmoid,  # logit:  log(s / (1 - s))
    _softmax,
)
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.api import AbstractLanguageModel, AbstractProcessRewardModel


def part1_the_math() -> None:
    print("=" * 72)
    print("PART 1 — PRM score  ->  logit (log-weight)  ->  softmax over particles")
    print("=" * 72)

    # Suppose a PRM scored the whole partial trajectory of 4 particles like this:
    scores = np.array([0.20, 0.50, 0.70, 0.90])  # s in [0, 1], higher = better

    # Step A: each score becomes a LOG-WEIGHT via the logit (inverse sigmoid).
    log_weights = np.array([_inv_sigmoid(s) for s in scores])

    # Step B: softmax over particles turns log-weights into resample probabilities.
    probs = _softmax(log_weights)

    # Claim from Chapter 7: probs[i] is proportional to the ODDS  s/(1-s).
    odds = scores / (1.0 - scores)
    odds_normalized = odds / odds.sum()

    print(f"{'particle':>9} {'score s':>9} {'logit(s)=w':>12} {'softmax p':>11} {'odds/Σodds':>12}")
    for i in range(len(scores)):
        print(f"{i:>9} {scores[i]:>9.3f} {log_weights[i]:>12.4f} {probs[i]:>11.4f} {odds_normalized[i]:>12.4f}")

    assert np.allclose(probs, odds_normalized), "softmax(logit(s)) must equal normalized odds"
    print("\n  ✓ softmax(logit(s)) == normalized odds  ->  resample prob ∝ s/(1-s)")
    print("  Across STEPS the filter does NOT sum or multiply these; it re-derives w each")
    print("  step from the PRM's score of the WHOLE prefix. Any product of per-step rewards")
    print("  happens INSIDE the PRM (aggregation_method='prod'), not here.")


# --- Minimal mocks so the real algorithm can run with no GPU / no server -------------------

class MockLM(AbstractLanguageModel):
    """Emits a deterministic 'step k' each call; supports single + batch like the real LMs."""

    def __init__(self) -> None:
        self.counter = 0

    def _one(self) -> dict:
        step = f"step{self.counter}"
        self.counter += 1
        return {"role": "assistant", "content": step}

    async def agenerate_single(self, messages, **kwargs) -> dict:
        return self._one()

    async def agenerate(self, messages, **kwargs):
        # batch path: messages is a list of conversations (list of lists)
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [self._one() for _ in messages]
        return self._one()


class MockPRM(AbstractProcessRewardModel):
    """Returns probability-like scores by cycling a preset list, so particles differ."""

    def __init__(self) -> None:
        self.preset = [0.30, 0.55, 0.70, 0.85, 0.40, 0.60]
        self.i = 0

    def _score_one(self) -> float:
        s = self.preset[self.i % len(self.preset)]
        self.i += 1
        return s

    def score(self, prompt, response_or_responses):
        if isinstance(response_or_responses, list):
            return [self._score_one() for _ in response_or_responses]
        return self._score_one()

    async def ascore(self, prompt, response_or_responses):
        return self.score(prompt, response_or_responses)


def part2_real_algorithm() -> None:
    print("\n" + "=" * 72)
    print("PART 2 — run the REAL ParticleFiltering with a mock LM + mock PRM (no GPU)")
    print("=" * 72)

    random.seed(0)
    np.random.seed(0)

    sg = StepGeneration(step_token="\n", max_steps=3)
    pf = ParticleFiltering(sg=sg, prm=MockPRM())

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
