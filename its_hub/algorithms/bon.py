from typing import Union, List
from pydantic.dataclasses import dataclass

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractOutcomeRewardModel
from ..types import ChatMessage


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
        return_response_only: bool = True, 
    ) -> Union[str, BestOfNResult]:
        # generate responses
        message_lists = [[ChatMessage(role="user", content=prompt)] for _ in range(budget)]
        responses = lm.generate(message_lists)
        if isinstance(responses, str):
            responses = [responses]

        # score responses
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        batched = True
        if batched:
            scores = self.orm.score(prompt, responses)
        else:
            scores = [] 
            for r in responses:
                scores.append(self.orm.score(prompt, r))

        # select the best response
        if isinstance(scores, list) and len(scores) > 0:
            selected_index = scores.index(max(scores))
        else:
            selected_index = 0

        # return the result
        result = BestOfNResult(
            responses=responses if isinstance(responses, list) else [responses], 
            scores=scores if isinstance(scores, list) else [scores], 
            selected_index=selected_index, 
        )
        return result.the_one if return_response_only else result
