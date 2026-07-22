"""Field-aware tool-call self-consistency scorer.

Given N sampled tool calls for one task:
  (a) Vote on tool name (majority, after normalization).
  (b) Among samples that agree on the winning tool name, score each
      argument field by exact match (majority vote per field).
  (c) Tag the result as "high_confidence" or "forced" based on a
      configurable field-agreement threshold (default 75%).
  (d) Optionally abstain (return None) on forced picks when
      allow_abstain=True — available for research analysis but off
      by default to match production behavior.

Tool-name ties (no strict majority) are always tagged "forced"
regardless of per-field agreement.

Argument parsing follows the same conventions as
its_hub.core.algorithms.self_consistency._parse_tool_args and _make_hashable
for consistency with the existing library.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field


def normalize_tool_name(name: str) -> str:
    """Normalize tool/function name for comparison.

    Lowercases, strips whitespace, and collapses hyphens/underscores
    so that e.g. 'get_Weather', 'get-weather', 'GET_WEATHER' all match.
    """
    name = name.strip().lower()
    name = re.sub(r"[-_]+", "_", name)
    return name


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
    tool_name_is_tie: bool
    merged_args: dict
    field_votes: list[FieldVote]
    confidence: str  # "high_confidence" or "forced"
    selected_index: int
    num_samples: int
    raw_tool_calls: list[dict] = field(default_factory=list)


def _majority_vote(values: list) -> tuple[object, int, bool]:
    """Return (winning_value, count, is_tie) with random tiebreak."""
    counts = Counter(values)
    max_count = max(counts.values())
    winners = [v for v, c in counts.items() if c == max_count]
    is_tie = len(winners) > 1
    winner = random.choice(winners)
    return winner, max_count, is_tie


def score_tool_calls(
    tool_calls: list[dict],
    threshold: float = 0.75,
    allow_abstain: bool = False,
) -> ScoredToolCall | None:
    """Score N sampled tool calls with field-aware majority voting.

    Args:
        tool_calls: List of tool call dicts, each with structure:
            {"function": {"name": str, "arguments": str | dict}}
        threshold: Minimum agreement ratio (across all fields) to tag
            as "high_confidence". Must be in (0, 1]. Default 0.75.
        allow_abstain: If True, return None instead of a forced selection
            when confidence is below threshold. Off by default to match
            production behavior (always return a selection).

    Returns:
        ScoredToolCall with the merged result and confidence tag,
        or None if allow_abstain=True and confidence would be "forced".

    Raises:
        ValueError: If tool_calls is empty or threshold is out of range.
    """
    if not tool_calls:
        raise ValueError("Cannot score empty tool_calls list")
    if not 0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1.0], got: {threshold}")

    # Step (a): vote on tool name (after normalization)
    raw_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
    normalized_names = [normalize_tool_name(n) for n in raw_names]
    winning_normalized, name_vote_count, name_is_tie = _majority_vote(normalized_names)

    # Map back to original name from the first sample that matches
    winning_name = raw_names[normalized_names.index(winning_normalized)]

    # Filter to samples that agree on the winning tool name
    agreeing_indices = [
        i for i, n in enumerate(normalized_names) if n == winning_normalized
    ]
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

        winning_value, vote_count, _ = _majority_vote(values)
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

    # Step (c): tag confidence based on field agreement AND tool-name tie
    if name_is_tie:
        confidence = "forced"
    elif field_votes:
        min_agreement = min(fv.agreement for fv in field_votes)
        confidence = "high_confidence" if min_agreement >= threshold else "forced"
    else:
        confidence = "high_confidence"

    # Step (d): optionally abstain on forced picks
    if allow_abstain and confidence == "forced":
        return None

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
        tool_name_is_tie=name_is_tie,
        merged_args=merged_args,
        field_votes=field_votes,
        confidence=confidence,
        selected_index=selected_index,
        num_samples=len(tool_calls),
        raw_tool_calls=tool_calls,
    )
