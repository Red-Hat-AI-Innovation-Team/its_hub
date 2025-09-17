import random
from collections import Counter
from collections.abc import Callable

from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage
from its_hub.utils import extract_response_content


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: list[str] | list[dict]
    response_counts: Counter[str]
    selected_index: int

    @property
    def the_one(self) -> str | dict:
        return self.responses[self.selected_index]


def _select_most_common_or_random(list_to_select_from: list[str]) -> int:
    # count occurrences of each element
    counts = Counter(list_to_select_from)

    # find the element with maximum occurrences
    max_count = max(counts.values())

    # find indices of the most common elements
    most_common_indices = [
        i for i, r in enumerate(list_to_select_from) if counts[r] == max_count
    ]

    # select a random index from the most common ones
    # note above implementation ensures that if there are multiple
    #      elements with the same count, a random one is selected
    selected_index = random.choice(most_common_indices)

    return counts, selected_index


class SelfConsistency(AbstractScalingAlgorithm):
    def __init__(self, consistency_space_projection_func: Callable):
        self.consistency_space_projection_func = consistency_space_projection_func

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str | list[ChatMessage],
        budget: int,
        return_response_only: bool = True,
        messages_output: bool = False,
    ) -> str | dict | SelfConsistencyResult:
        # Handle both string prompts and conversation history
        if isinstance(prompt, str):
            # Legacy string prompt
            messages_list = [[ChatMessage(role="user", content=prompt)] for _ in range(budget)]
        else:
            # Full conversation history
            messages_list = [prompt for _ in range(budget)]
        
        # Always generate as message objects internally for consistency
        responses = lm.generate(messages_list, messages_output=True)

        # Single extraction function that works regardless of format
        def extract_text_for_projection(response) -> str:
            return extract_response_content(response)

        # project responses into consistency space
        responses_projected = [
            self.consistency_space_projection_func(extract_text_for_projection(r)) for r in responses
        ]

        # select the most common or random response
        response_counts, selected_index = _select_most_common_or_random(
            responses_projected
        )

        # Format responses according to user preference
        if messages_output:
            formatted_responses = responses  # Keep as message objects
        else:
            formatted_responses = [extract_response_content(resp) for resp in responses]

        # Return result based on format preference
        if return_response_only:
            return formatted_responses[selected_index]
        else:
            return SelfConsistencyResult(
                responses=formatted_responses,
                response_counts=response_counts,
                selected_index=selected_index,
            )
