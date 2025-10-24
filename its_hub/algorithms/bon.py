from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage, ChatMessages
from its_hub.utils import extract_content_from_lm_response
import logging

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
    usage: dict | None = None  # Cumulative usage across all N responses
    reward_usage: dict | None = None  # Usage from reward model scoring

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, orm: AbstractOutcomeRewardModel):
        self.orm = orm

    def _aggregate_usage(self, usages: list[dict]) -> dict:
        """Aggregate usage information from multiple responses.

        Args:
            usages: List of usage dictionaries from each response

        Returns:
            Dictionary with cumulative usage totals
        """
        total_usage = {}
        for usage in usages:
            if not usage:
                continue
            for key, value in usage.items():
                if value is None:
                    continue
                if isinstance(value, dict):
                    # Handle nested dicts (e.g., prompt_tokens_details, completion_tokens_details)
                    if key not in total_usage:
                        total_usage[key] = {}
                    for k, v in value.items():
                        if v is None:
                            continue
                        total_usage[key][k] = total_usage[key].get(k, 0) + v
                else:
                    # Handle simple numeric values
                    total_usage[key] = total_usage.get(key, 0) + value
        return total_usage

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

        # generate responses - returns list of (message, usage) tuples
        response_tuples = await lm.agenerate(
            chat_messages.to_batch(budget), tools=tools, tool_choice=tool_choice
        )

        # unpack responses and usage information
        responses = [msg for msg, _ in response_tuples]
        usages = [usage for _, usage in response_tuples]

        # extract content from message dict responses
        response_contents = [extract_content_from_lm_response(r) for r in responses]

        # deduplicate responses to avoid redundant scoring
        unique_responses, inverse_idx = _dedupe_with_inverse(response_contents)

        # early return if all responses are identical - no need to score
        if len(unique_responses) == 1:
            scores = [1] * len(responses)
            # Calculate cumulative usage
            total_usage = self._aggregate_usage(usages)
            result = BestOfNResult(
                responses=responses,
                scores=scores,
                selected_index=0,
                usage=total_usage,
                reward_usage=None,
            )
            return result.the_one if return_response_only else result

        # score only unique responses with usage tracking
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        reward_usage = None
        batched = True
        if batched:
            # Try to get usage information from scoring
            # reward_hub's ascore may return (scores, usage) tuple or just scores
            result = await self.orm.ascore(chat_messages, unique_responses)

            # Check if result is a tuple (scores/JudgeResult, usage)
            if isinstance(result, tuple) and len(result) == 2:
                scores_or_judge_result, judge_usage = result

                # Check if first element is a JudgeResult object (has .scores attribute)
                if hasattr(scores_or_judge_result, 'scores'):
                    # JudgeResult with reasoning
                    unique_scores = scores_or_judge_result.scores
                else:
                    # Just scores (list or float)
                    unique_scores = scores_or_judge_result

                reward_usage = judge_usage.model_dump()
                logging.info(f"Judge usage: {reward_usage}")
            else:
                # Old interface - just scores or JudgeResult
                if hasattr(result, 'scores'):
                    unique_scores = result.scores
                else:
                    unique_scores = result
                logging.info("Reward model does not return usage information")
        else:
            unique_scores = []
            for r in unique_responses:
                result = await self.orm.ascore(chat_messages, r)

                # Check if result is a tuple (score/JudgeResult, usage)
                if isinstance(result, tuple) and len(result) == 2:
                    score_or_judge_result, judge_usage = result

                    # Check if first element is a JudgeResult object
                    if hasattr(score_or_judge_result, 'scores'):
                        score = score_or_judge_result.scores
                    else:
                        score = score_or_judge_result

                    unique_scores.append(score)
                    if reward_usage is None:
                        reward_usage = judge_usage.model_dump()
                else:
                    # Old interface - just score or JudgeResult
                    if hasattr(result, 'scores'):
                        unique_scores.append(result.scores)
                    else:
                        unique_scores.append(result)

        # map scores back to original response indices
        scores = [unique_scores[idx] for idx in inverse_idx]

        # select the best response
        selected_index = scores.index(max(scores))

        # Calculate cumulative usage across all N responses
        total_usage = self._aggregate_usage(usages)

        # return the result - preserve original message format with tool calls
        result = BestOfNResult(
            responses=responses,  # Keep original dict format with tool calls
            scores=scores,
            selected_index=selected_index,
            usage=total_usage,
            reward_usage=reward_usage,
        )
        return result.the_one if return_response_only else result
