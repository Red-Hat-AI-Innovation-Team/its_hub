from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage
from its_hub.utils import extract_response_content


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: list[str] | list[dict]
    scores: list[float]
    selected_index: int

    @property
    def the_one(self) -> str | dict:
        return self.responses[self.selected_index]


class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, orm: AbstractOutcomeRewardModel):
        self.orm = orm

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str | list[ChatMessage],
        budget: int,
        return_response_only: bool = True,
        messages_output: bool = False,
    ) -> str | dict | BestOfNResult:
        # Handle both string prompts and conversation history
        if isinstance(prompt, str):
            # Legacy string prompt
            messages_list = [[ChatMessage(role="user", content=prompt)] for _ in range(budget)]
        else:
            # Full conversation history
            messages_list = [prompt for _ in range(budget)]
            # TODO: Update ORM interface to natively support conversation history
            # Currently using a simple flattening approach as a temporary workaround
            import warnings
            warnings.warn(
                "BestOfN with conversation history uses simplified prompt flattening for ORM scoring. "
                "This may not preserve full conversation context. ORM interface should be updated to "
                "support message format natively.",
                UserWarning,
                stacklevel=2
            )

        # Generate responses in user's preferred format
        responses = lm.generate(messages_list, messages_output=messages_output)

        # Prepare prompt for ORM scoring
        if isinstance(prompt, str):
            scoring_prompt = prompt
        else:
            # Flatten conversation for current ORM interface
            scoring_prompt = "\n".join(f"{msg.role}: {msg.content}" for msg in prompt)

        # Extract content for scoring (ORM expects string content)
        response_contents = [extract_response_content(r) for r in responses]

        # score responses
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        batched = True
        if batched:
            scores = self.orm.score(scoring_prompt, response_contents)
        else:
            scores = []
            for r in response_contents:
                scores.append(self.orm.score(scoring_prompt, r))

        # select the best response
        selected_index = scores.index(max(scores))

        # return the result
        result = BestOfNResult(
            responses=responses,
            scores=scores,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result
