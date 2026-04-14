"""Algorithm factory for e2e tests.

Builds all available scaling algorithms.  Algorithms that require a process
reward model (beam-search, particle-filtering, entropic-particle-filtering)
are only included when ``--rm_name`` is provided **and** ``reward_hub`` is
installed.
"""

from its_hub import BestOfN, LLMJudge, SelfConsistency, StepGeneration
from its_hub.api import AbstractOrchestrator
from its_hub.core.algorithms.beam_search import BeamSearch
from its_hub.core.algorithms.particle_gibbs import (
    EntropicParticleFiltering,
    ParticleFiltering,
)

from tests.e2e.utils.evaluation import extract_boxed

# Optional: reward_hub for PRM-dependent algorithms
HAS_REWARD_HUB = False
try:
    from reward_hub.base import AggregationMethod

    from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

    HAS_REWARD_HUB = True
except ImportError:
    pass


# ------------------------------------------------------------------
# All recognised algorithm names
# ------------------------------------------------------------------
ALL_ALGORITHM_NAMES = [
    "self-consistency",
    "best-of-n",
    "beam-search",
    "particle-filtering",
    "entropic-particle-filtering",
]

PRM_ALGORITHMS = {"beam-search", "particle-filtering", "entropic-particle-filtering"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _build_step_generation(
    model_name: str, tokens_per_step: int | None
) -> StepGeneration:
    if tokens_per_step is not None:
        return StepGeneration(
            max_steps=50, tokens_per_step=tokens_per_step, stop_token="\\boxed"
        )
    step_token = "\n\n##" if "llama" in model_name.lower() else "\n\n"
    return StepGeneration(
        step_token=step_token, max_steps=50, stop_token="\\boxed"
    )


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def build_algorithms(
    lm,
    model_name: str,
    rm_name: str | None,
    tokens_per_step: int | None,
    orchestrator: AbstractOrchestrator | None = None,
) -> dict:
    """Return ``{name: algorithm}`` for every algorithm we can construct.

    * Self-Consistency and Best-of-N always work (LM only).
    * Beam-Search, Particle-Filtering, and Entropic-PF require
      ``--rm_name`` and ``reward_hub``.

    If *orchestrator* is provided, it is shared by SelfConsistency and BestOfN.
    """
    algs: dict = {}

    # --- algorithms that need only an LM --------------------------------
    algs["self-consistency"] = SelfConsistency(extract_boxed, orchestrator=orchestrator)

    judge = LLMJudge(lm=lm, fallback_score=5.0)
    algs["best-of-n"] = BestOfN(orm=judge, orchestrator=orchestrator)

    # --- algorithms that also need a PRM --------------------------------
    if rm_name and HAS_REWARD_HUB:
        sg = _build_step_generation(model_name, tokens_per_step)
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name,
            aggregation_method=AggregationMethod("model"),
        )
        algs["beam-search"] = BeamSearch(sg, prm, beam_width=2)
        algs["particle-filtering"] = ParticleFiltering(sg, prm)
        algs["entropic-particle-filtering"] = EntropicParticleFiltering(sg, prm)
        print(f"  PRM: LocalVllmProcessRewardModel({rm_name})")

    return algs
