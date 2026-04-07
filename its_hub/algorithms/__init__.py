from .beam_search import BeamSearch, BeamSearchResult
from .bon import BestOfN, BestOfNResult
from .particle_gibbs import (
    EntropicParticleFiltering,
    ParticleFiltering,
    ParticleFilteringResult,
    ParticleGibbs,
    ParticleGibbsResult,
)
from .self_consistency import ProjectionPreset, SelfConsistency, SelfConsistencyResult

__all__ = [
    "BeamSearch",
    "BeamSearchResult",
    "BestOfN",
    "BestOfNResult",
    "EntropicParticleFiltering",
    "MetropolisHastings",
    "MetropolisHastingsResult",
    "ParticleFiltering",
    "ParticleFilteringResult",
    "ParticleGibbs",
    "ParticleGibbsResult",
    "ProjectionPreset",
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
from its_hub.types import ChatMessage, ChatMessages


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
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        show_progress: bool = False,
        return_response_only: bool = True,
    ) -> str | MetropolisHastingsResult:
        # TODO: Implement Metropolis-Hastings algorithm
        # Will need to convert prompt_or_messages to ChatMessages format when implemented
        raise NotImplementedError("Metropolis-Hastings algorithm not yet implemented")
