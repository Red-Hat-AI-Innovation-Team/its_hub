from typing import Union, List
from dataclasses import dataclass
from tqdm import tqdm

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractOutcomeRewardModel


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: List[str]
    scores: List[float]
    selected_index: int

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]

class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, orm: AbstractOutcomeRewardModel):
        self.orm = orm

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, BestOfNResult]:
        # generate responses
        responses = lm.generate([prompt] * budget)

        # score responses
        scores = [] 
        for r in tqdm(responses, desc="Scoring", disable=(not show_progress)):
            scores.append(self.orm.score(prompt, r))

        # select the best response
        selected_index = scores.index(max(scores))

        # return the result
        result = BestOfNResult(
            responses=responses, 
            scores=scores, 
            selected_index=selected_index, 
        )
        return result.the_one if return_response_only else result
