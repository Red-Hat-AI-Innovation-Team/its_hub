"""Field-aware tool-call self-consistency scorer.

Given N sampled tool calls for one task:
  (a) Vote on tool name (majority).
  (b) Among samples that agree on the winning tool name, score each
      argument field by exact match (majority vote per field).
  (c) Tag the result as "high_confidence" or "forced" based on a
      configurable field-agreement threshold (default 75%).

Always returns a selection — never abstains.

Argument parsing follows the same conventions as
its_hub.core.algorithms.self_consistency._parse_tool_args and _make_hashable
for consistency with the existing library.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field


def parse_tool_args(raw_args: str | dict | None) -> dict:
    """Parse tool call arguments into a dict.

    Mirrors its_hub.core.algorithms.self_consistency._parse_tool_args
    but returns a dict instead of a hashable tuple, since field-aware
    scoring needs to iterate over individual fields.
    """
    if raw_args is None:
        return {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw_args, dict):
        return {}
    return raw_args


def make_hashable(obj: object) -> object:
    """Recursively convert nested structures to hashable types.

    Mirrors its_hub.core.algorithms.self_consistency.SelfConsistency._make_hashable.
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(item) for item in obj))
    else:
        return obj


@dataclass
class FieldVote:
    """Result of majority voting on a single argument field."""

    field_name: str
    winning_value: object
    vote_count: int
    total_votes: int
    agreement: float


@dataclass
class ScoredToolCall:
    """Result of field-aware scoring over N sampled tool calls."""

    tool_name: str
    tool_name_vote_count: int
    tool_name_total: int
    merged_args: dict
    field_votes: list[FieldVote]
    confidence: str  # "high_confidence" or "forced"
    selected_index: int
    num_samples: int
    raw_tool_calls: list[dict] = field(default_factory=list)


def _majority_vote(values: list) -> tuple[object, int]:
    """Return (winning_value, count) with random tiebreak."""
    counts = Counter(values)
    max_count = max(counts.values())
    winners = [v for v, c in counts.items() if c == max_count]
    winner = random.choice(winners)
    return winner, max_count


def score_tool_calls(
    tool_calls: list[dict],
    threshold: float = 0.75,
) -> ScoredToolCall:
    """Score N sampled tool calls with field-aware majority voting.

    Args:
        tool_calls: List of tool call dicts, each with structure:
            {"function": {"name": str, "arguments": str | dict}}
        threshold: Minimum agreement ratio (across all fields) to tag
            as "high_confidence". Must be in (0, 1]. Default 0.75.

    Returns:
        ScoredToolCall with the merged result and confidence tag.

    Raises:
        ValueError: If tool_calls is empty or threshold is out of range.
    """
    if not tool_calls:
        raise ValueError("Cannot score empty tool_calls list")
    if not 0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1.0], got: {threshold}")

    # Step (a): vote on tool name
    names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
    winning_name, name_vote_count = _majority_vote(names)

    # Filter to samples that agree on the winning tool name
    agreeing_indices = [i for i, n in enumerate(names) if n == winning_name]
    agreeing_calls = [tool_calls[i] for i in agreeing_indices]

    # Parse arguments for agreeing calls
    all_args = [
        parse_tool_args(tc.get("function", {}).get("arguments"))
        for tc in agreeing_calls
    ]

    # Collect all field names across agreeing samples
    all_field_names: set[str] = set()
    for args in all_args:
        all_field_names.update(args.keys())

    # Step (b): vote on each argument field independently
    field_votes: list[FieldVote] = []
    merged_args: dict = {}

    for field_name in sorted(all_field_names):
        values = []
        for args in all_args:
            if field_name in args:
                values.append(make_hashable(args[field_name]))

        if not values:
            continue

        winning_value, vote_count = _majority_vote(values)
        total_votes = len(values)
        agreement = vote_count / total_votes

        field_votes.append(
            FieldVote(
                field_name=field_name,
                winning_value=winning_value,
                vote_count=vote_count,
                total_votes=total_votes,
                agreement=agreement,
            )
        )
        # Unhash for the merged args dict — use the raw value from the
        # first agreeing sample that matches the winning hashable value
        raw_value = winning_value
        for args in all_args:
            if field_name in args and make_hashable(args[field_name]) == winning_value:
                raw_value = args[field_name]
                break
        merged_args[field_name] = raw_value

    # Step (c): tag confidence based on field agreement
    if field_votes:
        min_agreement = min(fv.agreement for fv in field_votes)
    else:
        min_agreement = 1.0

    confidence = "high_confidence" if min_agreement >= threshold else "forced"

    # Select the original sample closest to the merged result (prefer
    # an agreeing sample whose args exactly match the merged args)
    selected_index = agreeing_indices[0]
    for i in agreeing_indices:
        tc_args = parse_tool_args(tool_calls[i].get("function", {}).get("arguments"))
        if all(
            make_hashable(tc_args.get(fv.field_name)) == fv.winning_value
            for fv in field_votes
            if fv.field_name in tc_args
        ):
            selected_index = i
            break

    return ScoredToolCall(
        tool_name=winning_name,
        tool_name_vote_count=name_vote_count,
        tool_name_total=len(tool_calls),
        merged_args=merged_args,
        field_votes=field_votes,
        confidence=confidence,
        selected_index=selected_index,
        num_samples=len(tool_calls),
        raw_tool_calls=tool_calls,
    )
