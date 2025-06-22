from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: list[str]
    scores: list[float]
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
    ) -> str | BestOfNResult:
        # generate responses
        responses = lm.generate(
            [[ChatMessage(role="user", content=prompt)] for _ in range(budget)]
        )

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
        selected_index = scores.index(max(scores))

        # return the result
        result = BestOfNResult(
            responses=responses,
            scores=scores,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result
