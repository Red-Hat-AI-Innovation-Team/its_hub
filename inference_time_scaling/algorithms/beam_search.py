from typing import Union

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractProcessRewardModel
from ..lms import StepGeneration


# TODO(GX) implement BeamSearch
class BeamSearchResult(AbstractScalingResult):
    pass

class BeamSearch(AbstractScalingAlgorithm):
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
    ) -> Union[str, BeamSearchResult]:
        pass
