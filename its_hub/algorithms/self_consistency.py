import logging
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
from its_hub.types import ChatMessage, ChatMessages


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    response_counts: Counter[str] | Counter[tuple] | Counter
    selected_index: int

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


def _select_most_common_or_random(
    list_to_select_from: list[str],
) -> tuple[Counter, int]:
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


def _select_hierarchical_most_common_or_random(
    list_to_select_from: list[tuple],
) -> tuple[Counter, int]:
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
    def __init__(
        self,
        consistency_space_projection_func: Callable,
        tool_vote: str | None = None,
        exclude_args: list[str] | None = None,
    ):
        """Initialize SelfConsistency algorithm with optional tool-vote capability.

        Args:
            consistency_space_projection_func: Function that maps response content (str)
                to a comparable value for voting. Used when tool_vote is None or when
                responses don't contain tool calls. Can return str, tuple, or any hashable type.

            tool_vote: Tool voting strategy when responses contain tool calls. Options:
                - None (default): Vote on message content using consistency_space_projection_func
                - "tool_name": Vote on tool function names only
                - "tool_args": Vote on tool function arguments only (as dicts)
                - "tool_hierarchical": Vote on tool name first, then arguments (hierarchical)
                When tool calls exist and tool_vote is set, this takes priority over content voting.

            exclude_args: List of argument names to exclude from tool voting when
                tool_vote is "tool_args" or "tool_hierarchical". Useful for filtering out
                non-semantic arguments like timestamps, request IDs, etc.

        Raises:
            ValueError: If tool_vote is not one of the supported options.
        """
        # Validate tool_vote parameter - only validation needed since typing handles the rest
        valid_tool_vote_options = {None, "tool_name", "tool_args", "tool_hierarchical"}
        if tool_vote not in valid_tool_vote_options:
            raise ValueError(
                f"tool_vote must be one of {valid_tool_vote_options}, got: {tool_vote}"
            )

        self.consistency_space_projection_func = consistency_space_projection_func
        self.tool_vote = tool_vote
        self.exclude_args = exclude_args or []

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | SelfConsistencyResult:
        # Convert to uniform ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # generate responses using async method
        responses = await lm.agenerate(
            chat_messages.to_batch(budget), tools=tools, tool_choice=tool_choice
        )

        # Check if majority of responses have tool calls to decide voting method
        tool_call_count = sum(1 for r in responses if r.get("tool_calls"))
        has_majority_tool_calls = tool_call_count >= len(responses) // 2

        if has_majority_tool_calls and self.tool_vote:
            logging.info("Tool voting is invoked")
            # mutate responses variable to be only responses with tool calls;
            responses = [r for r in responses if r.get("tool_calls")]

            # Vote on tool calls directly
            responses_projected = [
                self._extract_tool_call_features(r) for r in responses
            ]
        else:
            # Vote on message content (existing behavior)
            response_contents = [r.get("content", "") or "" for r in responses]
            responses_projected = [
                self.consistency_space_projection_func(r) for r in response_contents
            ]

        # determine if we're dealing with hierarchical (tuple) or flat (string) projections
        if responses_projected and isinstance(responses_projected[0], tuple):
            # hierarchical consistency space
            response_counts, selected_index = (
                _select_hierarchical_most_common_or_random(responses_projected)
            )
        else:
            # flat consistency space (backward compatibility)
            response_counts, selected_index = _select_most_common_or_random(
                responses_projected
            )

        # return the result - preserve original message format with tool calls
        result = SelfConsistencyResult(
            responses=responses,  # Keep original dict format with tool calls
            response_counts=response_counts,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | SelfConsistencyResult:
        # Convert to uniform ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # generate responses
        responses = lm.generate(
            chat_messages.to_batch(budget), tools=tools, tool_choice=tool_choice
        )

        # Check if majority of responses have tool calls to decide voting method
        tool_call_count = sum(1 for r in responses if r.get("tool_calls"))
        has_majority_tool_calls = tool_call_count >= len(responses) // 2

        if has_majority_tool_calls and self.tool_vote:
            logging.info("Tool voting is invoked")
            # mutate responses variable to be only responses with tool calls;
            responses = [r for r in responses if r.get("tool_calls")]

            # Vote on tool calls directly
            responses_projected = [
                self._extract_tool_call_features(r) for r in responses
            ]
        else:
            # Vote on message content (existing behavior)
            response_contents = [r.get("content", "") or "" for r in responses]
            responses_projected = [
                self.consistency_space_projection_func(r) for r in response_contents
            ]

        # determine if we're dealing with hierarchical (tuple) or flat (string) projections
        if responses_projected and isinstance(responses_projected[0], tuple):
            # hierarchical consistency space
            response_counts, selected_index = (
                _select_hierarchical_most_common_or_random(responses_projected)
            )
        else:
            # flat consistency space (backward compatibility)
            response_counts, selected_index = _select_most_common_or_random(
                responses_projected
            )

        # return the result - preserve original message format with tool calls
        result = SelfConsistencyResult(
            responses=responses,  # Keep original dict format with tool calls
            response_counts=response_counts,
            selected_index=selected_index,
        )
        return result.the_one if return_response_only else result

    def _extract_tool_call_features(self, message_obj: dict):
        """Extract tool call features for voting based on tool_vote type."""
        tool_calls = message_obj.get("tool_calls", [])
        if not tool_calls:
            return None if self.tool_vote == "tool_name" else (None, None)

        first_tc = tool_calls[0]
        function_name = first_tc.get("function", {}).get("name")
        function_args = first_tc.get("function", {}).get("arguments", {})

        # Handle case where arguments might be a JSON string instead of dict
        if isinstance(function_args, str):
            try:
                import json

                function_args = json.loads(function_args)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, treat as empty dict
                function_args = {}

        # Ensure function_args is a dict
        if not isinstance(function_args, dict):
            function_args = {}

        # Filter arguments if specified
        if self.exclude_args:
            function_args = {
                k: v for k, v in function_args.items() if k not in self.exclude_args
            }

        # Convert dict to hashable tuple for Counter compatibility
        # handles nested structures
        def make_hashable(obj):
            """Recursively convert nested structures to hashable types."""
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, set):
                return tuple(sorted(make_hashable(item) for item in obj))
            else:
                return obj

        args_tuple = make_hashable(function_args) if function_args else ()

        if self.tool_vote == "tool_name":
            return function_name
        elif self.tool_vote == "tool_args":
            return args_tuple  # Use tuple instead of dict
        elif self.tool_vote == "tool_hierarchical":
            return (function_name, args_tuple)  # Use tuple instead of dict
        else:
            raise ValueError(f"Unknown tool_vote type: {self.tool_vote}")


def create_regex_projection_function(
    patterns: str | list[str],
) -> Callable[[str], tuple]:
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
    compiled_patterns = [
        re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in patterns
    ]

    def projection_function(response: str | None) -> tuple:
        """Extract features from response using compiled regex patterns."""
        results = []

        # Handle None or empty response
        if response is None:
            response = ""

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
