"""Algorithm factory for e2e tests.

Builds the particle-filtering algorithms (PF and EPF). Both weight particles
from the generator's own token logprobs (self-certainty), so the only
requirement is an OpenAI-compatible endpoint that supports ``logprobs``.
"""

from its_hub import EntropicParticleFiltering, ParticleFiltering, StepGeneration

# ------------------------------------------------------------------
# All recognised algorithm names
# ------------------------------------------------------------------
ALL_ALGORITHM_NAMES = [
    "particle-filtering",
    "entropic-particle-filtering",
]


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
    model_name: str,
    tokens_per_step: int | None,
) -> dict:
    """Return ``{name: algorithm}`` for every algorithm we can construct."""
    sg = _build_step_generation(model_name, tokens_per_step)
    return {
        "particle-filtering": ParticleFiltering(sg),
        "entropic-particle-filtering": EntropicParticleFiltering(sg),
    }
