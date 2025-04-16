
from .self_consistency import SelfConsistency, SelfConsistencyResult
from .bon import BestOfN, BestOfNResult
from .beam_search import BeamSearch, BeamSearchResult

###

from typing import Union

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractOutcomeRewardModel, AbstractProcessRewardModel
from ..lms import StepGeneration


class ParticleFilteringResult(AbstractScalingResult):
    pass

class ParticleFiltering(AbstractScalingAlgorithm):
    def __init__(self, step_generation: StepGeneration, prm: AbstractProcessRewardModel):
        self.step_generation = step_generation
        self.prm = prm

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, ParticleFilteringResult]:
        pass

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
        pass

class PGASResult(AbstractScalingResult):
    pass

class PGAS(AbstractScalingAlgorithm):
    def __init__(self, step_generation: StepGeneration, prm: AbstractProcessRewardModel):
        self.step_generation = step_generation
        self.prm = prm

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, PGASResult]:
        pass
