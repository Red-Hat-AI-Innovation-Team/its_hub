import random
import re
from collections import Counter
from collections.abc import Callable

from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: list[str]
    response_counts: Counter[str] | Counter[tuple]
    selected_index: int

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]


def _select_most_common_or_random(list_to_select_from: list[str]) -> tuple[Counter, int]:
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


def _select_hierarchical_most_common_or_random(list_to_select_from: list[tuple]) -> tuple[Counter, int]:
    if not list_to_select_from:
        raise ValueError("Cannot select from empty list")

    # If all elements are single-element tuples, fall back to flat behavior
    if all(len(item) == 1 for item in list_to_select_from):
        flat_list = [item[0] for item in list_to_select_from]
        _, selected_index = _select_most_common_or_random(flat_list)
        # Convert back to tuple format for consistency
        tuple_counts = Counter(list_to_select_from)
        return tuple_counts, selected_index

    # Find the maximum hierarchy depth
    max_depth = max(len(item) for item in list_to_select_from)

    # Start with all indices as candidates
    candidate_indices = list(range(len(list_to_select_from)))

    # Process each level of the hierarchy
    for level in range(max_depth):
        # Get the values at this level for current candidates
        level_values = []
        valid_indices = []

        for idx in candidate_indices:
            item = list_to_select_from[idx]
            if level < len(item):
                level_values.append(item[level])
                valid_indices.append(idx)

        if not level_values:
            break

        # Count occurrences at this level
        level_counts = Counter(level_values)
        max_count = max(level_counts.values())

        # Filter candidates to only those with the most common value at this level
        new_candidates = []
        for i, idx in enumerate(valid_indices):
            if level_counts[level_values[i]] == max_count:
                new_candidates.append(idx)

        candidate_indices = new_candidates

        # If we have a unique winner, we can stop
        if len(candidate_indices) == 1:
            break

    # Randomly select from remaining candidates
    selected_index = random.choice(candidate_indices)

    # Count all original tuples for the result
    tuple_counts = Counter(list_to_select_from)

    return tuple_counts, selected_index


class SelfConsistency(AbstractScalingAlgorithm):
    def __init__(self, consistency_space_projection_func: Callable):
        self.consistency_space_projection_func = consistency_space_projection_func

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | SelfConsistencyResult:
        # generate responses
        responses = lm.generate(
            [[ChatMessage(role="user", content=prompt)] for _ in range(budget)]
        )

        # project responses into consistency space
        responses_projected = [
            self.consistency_space_projection_func(r) for r in responses
        ]

        # determine if we're dealing with hierarchical (tuple) or flat (string) projections
        if responses_projected and isinstance(responses_projected[0], tuple):
            # hierarchical consistency space
            response_counts, selected_index = _select_hierarchical_most_common_or_random(
                responses_projected
            )
        else:
            # flat consistency space (backward compatibility)
            response_counts, selected_index = _select_most_common_or_random(
                responses_projected
            )

        # return the result
        result = SelfConsistencyResult(
            responses=responses,
            response_counts=response_counts,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result


def create_regex_projection_function(patterns: str | list[str]) -> Callable[[str], tuple]:
    """Create a hierarchical projection function from regex pattern(s).

    Args:
        patterns: Single regex pattern string or list of regex patterns.
                 Each pattern should contain capturing groups to extract features.
                 For hierarchical consistency, earlier patterns in the list represent
                 higher hierarchy levels.

    Returns:
        A projection function that takes a response string and returns a tuple
        where each element corresponds to the first match from each pattern.
        If no match is found for a pattern, None is used for that position.

    Example:
        # Single pattern for extracting final answer
        pattern = r'\\\\boxed\\{([^}]+)\\}'
        proj_func = create_regex_projection_function(pattern)
        proj_func("The answer is \\\\boxed{42}") -> ("42",)

        # Multiple patterns for hierarchical consistency
        patterns = [r'Method:\\s*(\\w+)', r'\\\\boxed\\{([^}]+)\\}']
        proj_func = create_regex_projection_function(patterns)
        proj_func("Method: algebra\\n...\\nAnswer: \\\\boxed{42}") -> ("algebra", "42")
    """
    # Ensure patterns is a list
    if isinstance(patterns, str):
        patterns = [patterns]

    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in patterns]

    def projection_function(response: str) -> tuple:
        """Extract features from response using compiled regex patterns."""
        results = []

        for pattern in compiled_patterns:
            match = pattern.search(response)
            if match:
                # If pattern has capturing groups, use the first group
                if match.groups():
                    results.append(match.group(1).strip())
                else:
                    # If no capturing groups, use the entire match
                    results.append(match.group(0).strip())
            else:
                # No match found, use None
                results.append(None)

        return tuple(results)

    return projection_function
