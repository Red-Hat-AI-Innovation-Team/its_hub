import json
from dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage, ChatMessages


def _response_to_hashable_key(response: dict) -> str:
    """
    Convert a response dict to a hashable key for deduplication.

    Handles:
    - content: None, str, or list[dict] (multi-modal)
    - tool_calls: Optional list of tool call dicts

    Returns canonical string representation for use as dict key.
    """
    # Handle content (can be None, str, or list[dict] for multi-modal)
    raw_content = response.get("content")
    if raw_content is None:
        content_str = ""
    elif isinstance(raw_content, str):
        content_str = raw_content
    elif isinstance(raw_content, list):
        # Multi-modal: extract text parts
        text_parts = [
            item.get("text", "")
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content_str = " ".join(text_parts)
    else:
        content_str = str(raw_content)

    # Handle tool_calls (optional, may not exist)
    tool_calls_str = ""
    if response.get("tool_calls"):
        tool_parts = []
        for tc in response.get("tool_calls", []):
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "")
                # Use json.dumps with sort_keys to make arguments hashable
                func_args = json.dumps(func.get("arguments", {}), sort_keys=True)
                tool_parts.append(f"{func_name}:{func_args}")
        tool_calls_str = "|".join(tool_parts)

    # Combine into canonical key (|| separator to avoid content/tool_calls collision)
    return f"{content_str}||{tool_calls_str}"


def _dedupe_responses_with_inverse(responses: list[dict]) -> tuple[list[dict], list[int]]:
    """
    Deduplicate response dicts while preserving order and tracking original indices.

    Returns (uniques, inverse_idx) where:
    - uniques: list of unique response dicts in order of first appearance
    - inverse_idx: for each response in original list, its index in uniques

    Deduplication considers both content and tool_calls for semantic equality.

    Example:
        responses = [r1, r2, r1, r3, r2]  # where r1, r2, r3 are response dicts
        returns ([r1, r2, r3], [0, 1, 0, 2, 1])
    """
    uniques: list[dict] = []
    index_of: dict[str, int] = {}
    inverse_idx: list[int] = []

    for response in responses:
        key = _response_to_hashable_key(response)
        j = index_of.get(key)
        if j is None:
            j = len(uniques)
            index_of[key] = j
            uniques.append(response)  # Keep original dict with all fields
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
    ) -> dict | BestOfNResult:
        """run inference asynchronously with best-of-n"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # generate responses
        responses = await lm.agenerate(
            chat_messages.to_batch(budget), tools=tools, tool_choice=tool_choice
        )

        # deduplicate responses to avoid redundant scoring
        # Deduplication considers both content and tool_calls
        unique_responses, inverse_idx = _dedupe_responses_with_inverse(responses)

        # early return if all responses are identical - no need to score
        if len(unique_responses) == 1:
            scores = [1.0] * len(responses)
            result = BestOfNResult(
                responses=responses,
                scores=scores,
                selected_index=0,
            )
            return result.the_one if return_response_only else result

        # score only unique responses
        # Construct full conversations (context + response) for each unique response
        context_dicts = [msg.to_dict() for msg in chat_messages.to_chat_messages()]

        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        batched = True
        if batched:
            # Batch scoring: list of conversations
            conversations = [context_dicts + [response] for response in unique_responses]
            unique_scores = await self.orm.ascore(conversations)
        else:
            # Single scoring: one conversation at a time
            unique_scores = []
            for response in unique_responses:
                conversation = context_dicts + [response]
                unique_scores.append(await self.orm.ascore(conversation))

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
