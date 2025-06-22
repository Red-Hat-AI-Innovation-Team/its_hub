from .beam_search import BeamSearch, BeamSearchResult
from .bon import BestOfN, BestOfNResult
from .particle_gibbs import ParticleFiltering, ParticleGibbs, ParticleGibbsResult
from .self_consistency import SelfConsistency, SelfConsistencyResult

__all__ = [
    "BeamSearch",
    "BeamSearchResult",
    "BestOfN",
    "BestOfNResult",
    "MetropolisHastings",
    "MetropolisHastingsResult",
    "ParticleFiltering",
    "ParticleGibbs",
    "ParticleGibbsResult",
    "SelfConsistency",
    "SelfConsistencyResult",
]

###

from typing import Union

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.lms import StepGeneration


class MetropolisHastingsResult(AbstractScalingResult):
    pass


class MetropolisHastings(AbstractScalingAlgorithm):
    def __init__(
        self, step_generation: StepGeneration, orm: AbstractOutcomeRewardModel
    ):
        self.step_generation = step_generation
        self.orm = orm

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        show_progress: bool = False,
        return_response_only: bool = True,
    ) -> str | MetropolisHastingsResult:
        # TODO: Implement Metropolis-Hastings algorithm
        raise NotImplementedError("Metropolis-Hastings algorithm not yet implemented")
