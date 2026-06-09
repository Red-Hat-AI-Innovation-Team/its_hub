"""
entropic_temperature_demo.py — see the three entropic-annealing temperature schedules.

Runnable companion to documentation/08-entropic-particle-filtering.md. No GPU / server / key.

    /home/exx/miniconda3/envs/epf/bin/python documentation/snippets/entropic_temperature_demo.py
    # or:  conda run -n epf python documentation/snippets/entropic_temperature_demo.py

It calls the REAL temperature methods on an EntropicParticleFiltering instance, reproduces
the exact assertions from tests/test_entropic_annealing.py, then sweeps `progress` so you can
watch each schedule anneal toward T = 1.
"""

import numpy as np

from its_hub.core.algorithms.particle_gibbs import (
    EntropicParticleFiltering,
    ResamplingMethod,
    SelectionMethod,
    TemperatureMethod,
)
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.api import AbstractProcessRewardModel


class _NoopPRM(AbstractProcessRewardModel):
    def score(self, prompt, response):
        return 0.5

    async def ascore(self, prompt, response):
        return 0.5


def make_epf() -> EntropicParticleFiltering:
    return EntropicParticleFiltering(
        sg=StepGeneration(step_token="\n", max_steps=3),
        prm=_NoopPRM(),
        final_response_selection=SelectionMethod.ARGMAX,
        resampling_method=ResamplingMethod.SYSTEMATIC,
        temperature_method=TemperatureMethod.ESS,
        ess_threshold=0.5,
        early_phase=0.5,
    )


def check_against_tests(epf: EntropicParticleFiltering) -> None:
    print("=" * 72)
    print("Reproducing assertions from tests/test_entropic_annealing.py")
    print("=" * 72)

    # ESS schedule:  T = max(1, (1/ess_ratio) * (1 - progress))
    assert epf._temperature_ess(ess_ratio=0.2, progress=0.2) == 4.0
    assert epf._temperature_ess(ess_ratio=0.5, progress=0.8) == 1.0
    # ENTROPY schedule:  beta = H + (1-H)*progress ;  T = max(1, 1/beta)
    assert epf._temperature_entropy(entropy_n=0.5, progress=0.3) == 1.0 / (0.5 + 0.5 * 0.3)
    assert epf._temperature_entropy(entropy_n=1.0, progress=0.2) == 1.0
    # BASE schedule:  T = max(1, value_max - progress)
    assert epf._temperature_base(value_max=2.0, progress=0.5) == 1.5
    assert epf._temperature_base(value_max=0.8, progress=0.5) == 1.0
    # ESS & normalized entropy
    p = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert epf._effective_sample_size(p) == 1.0 / sum(x * x for x in p)
    print("  ✓ all temperature / ESS assertions match the library\n")


def sweep(epf: EntropicParticleFiltering) -> None:
    print("=" * 72)
    print("Temperature vs progress  (T = 1 means 'no flattening', i.e. ordinary PF)")
    print("=" * 72)
    print(f"{'progress':>9} {'ESS(r=0.2)':>11} {'ENTROPY(H=0.3)':>15} {'BASE(vmax=2)':>13}")
    for progress in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        t_ess = epf._temperature_ess(ess_ratio=0.2, progress=progress)
        t_ent = epf._temperature_entropy(entropy_n=0.3, progress=progress)
        t_base = epf._temperature_base(value_max=2.0, progress=progress)
        print(f"{progress:>9.2f} {t_ess:>11.3f} {t_ent:>15.3f} {t_base:>13.3f}")
    print("\n  All schedules start high (flatten the resample distribution while the swarm")
    print("  is collapsed & early) and anneal toward T = 1 as progress -> 1.")

    print("\n  Effect of T on a collapsed weight distribution (one dominant particle):")
    log_w = np.array([3.0, 0.0, 0.0, 0.0])  # particle 0 dominates
    for T in [1.0, 2.0, 4.0]:
        probs = np.exp((log_w / T) - np.max(log_w / T))
        probs = probs / probs.sum()
        print(f"    T={T:<3}  resample probs = {np.round(probs, 3).tolist()}")
    print("    Higher T -> flatter -> weak particles keep a fighting chance (diversity).")


if __name__ == "__main__":
    epf = make_epf()
    check_against_tests(epf)
    sweep(epf)
