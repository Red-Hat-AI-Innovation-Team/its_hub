from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage, ChatMessages
from its_hub.utils import extract_content_from_lm_response


def _dedupe_with_inverse(seq: list[str]) -> tuple[list[str], list[int]]:
    """
    Deduplicate a sequence while preserving order and tracking original indices.

    Returns (uniques, inverse_idx) where:
    - uniques: list of unique items in order of first appearance
    - inverse_idx: for each item in seq, its index in the uniques list

    Example:
        seq = ["a", "b", "a", "c", "b"]
        returns (["a", "b", "c"], [0, 1, 0, 2, 1])
    """
    uniques: list[str] = []
    index_of: dict[str, int] = {}
    inverse_idx: list[int] = []

    for item in seq:
        j = index_of.get(item)
        if j is None:
            j = len(uniques)
            index_of[item] = j
            uniques.append(item)
        inverse_idx.append(j)

    return uniques, inverse_idx


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    scores: list[float]
    selected_index: int

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, orm: AbstractOutcomeRewardModel):
        self.orm = orm

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> dict | BestOfNResult:
        """run inference asynchronously with best-of-n"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        generation_kwargs: dict = {
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if response_format is not None:
            generation_kwargs["response_format"] = response_format

        # generate responses
        responses = await lm.agenerate(
            chat_messages.to_batch(budget),
            **generation_kwargs,
        )

        # extract content from message dict responses
        response_contents = [extract_content_from_lm_response(r) for r in responses]

        # deduplicate responses to avoid redundant scoring
        unique_responses, inverse_idx = _dedupe_with_inverse(response_contents)

        # early return if all responses are identical - no need to score
        if len(unique_responses) == 1:
            scores = [1] * len(responses)
            result = BestOfNResult(
                responses=responses,
                scores=scores,
                selected_index=0,
            )
            return result.the_one if return_response_only else result

        # score only unique responses
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        batched = True
        if batched:
            unique_scores = await self.orm.ascore(chat_messages, unique_responses)
        else:
            unique_scores = []
            for r in unique_responses:
                unique_scores.append(await self.orm.ascore(chat_messages, r))

        # map scores back to original response indices
        scores = [unique_scores[idx] for idx in inverse_idx]

        # select the best response
        selected_index = scores.index(max(scores))

        # return the result - preserve original message format with tool calls
        result = BestOfNResult(
            responses=responses,  # Keep original dict format with tool calls
            scores=scores,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result
