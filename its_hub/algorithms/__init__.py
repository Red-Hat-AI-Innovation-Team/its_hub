from .self_consistency import SelfConsistency, SelfConsistencyResult
from .bon import BestOfN, BestOfNResult
from .beam_search import BeamSearch, BeamSearchResult
from .particle_gibbs import ParticleGibbs, ParticleGibbsResult, ParticleFiltering

__all__ = [
    "SelfConsistency", "SelfConsistencyResult",
    "BestOfN", "BestOfNResult", 
    "BeamSearch", "BeamSearchResult",
    "ParticleGibbs", "ParticleGibbsResult", "ParticleFiltering",
    "MetropolisHastings", "MetropolisHastingsResult"
]

###

from typing import Union

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractOutcomeRewardModel
from ..lms import StepGeneration


class MetropolisHastingsResult(AbstractScalingResult):
    pass

class MetropolisHastings(AbstractScalingAlgorithm):
    def __init__(self, step_generation: StepGeneration, orm: AbstractOutcomeRewardModel):
        self.step_generation = step_generation
        self.orm = orm

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, MetropolisHastingsResult]:
        # TODO: Implement Metropolis-Hastings algorithm
        raise NotImplementedError("Metropolis-Hastings algorithm not yet implemented")
