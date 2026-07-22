"""Tests for roll_up_scorer — field agreement, confidence tagging, tie handling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from roll_up_scorer import (
    FieldVote,
    ScoredToolCall,
    make_hashable,
    normalize_tool_name,
    parse_tool_args,
    score_tool_calls,
)


class TestParseToolArgs:
    def test_dict_passthrough(self):
        assert parse_tool_args({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert parse_tool_args('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_empty(self):
        assert parse_tool_args("not json") == {}

    def test_none_returns_empty(self):
        assert parse_tool_args(None) == {}

    def test_non_dict_returns_empty(self):
        assert parse_tool_args([1, 2, 3]) == {}


class TestMakeHashable:
    def test_dict(self):
        result = make_hashable({"b": 2, "a": 1})
        assert result == (("a", 1), ("b", 2))

    def test_list(self):
        assert make_hashable([1, 2]) == (1, 2)

    def test_nested(self):
        result = make_hashable({"a": [1, {"b": 2}]})
        assert result == (("a", (1, (("b", 2),))),)

    def test_primitive(self):
        assert make_hashable(42) == 42
        assert make_hashable("hello") == "hello"


class TestScoreToolCallsBasic:
    def test_unanimous_agreement(self, make_tool_call):
        calls = [
            make_tool_call("get_weather", {"city": "NYC", "unit": "celsius"}),
            make_tool_call("get_weather", {"city": "NYC", "unit": "celsius"}),
            make_tool_call("get_weather", {"city": "NYC", "unit": "celsius"}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name == "get_weather"
        assert result.confidence == "high_confidence"
        assert result.merged_args == {"city": "NYC", "unit": "celsius"}
        assert all(fv.agreement == 1.0 for fv in result.field_votes)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            score_tool_calls([])

    def test_invalid_threshold_raises(self, make_tool_call):
        calls = [make_tool_call("f", {"a": 1})]
        with pytest.raises(ValueError, match="threshold"):
            score_tool_calls(calls, threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            score_tool_calls(calls, threshold=1.5)


class TestToolNameVoting:
    def test_majority_name_wins(self, make_tool_call):
        calls = [
            make_tool_call("search", {"q": "cats"}),
            make_tool_call("search", {"q": "dogs"}),
            make_tool_call("lookup", {"q": "cats"}),
            make_tool_call("search", {"q": "cats"}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name == "search"
        assert result.tool_name_vote_count == 3
        assert result.tool_name_total == 4

    def test_dissenting_name_excluded_from_field_voting(self, make_tool_call):
        calls = [
            make_tool_call("search", {"q": "cats"}),
            make_tool_call("search", {"q": "cats"}),
            make_tool_call("lookup", {"q": "dogs"}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name == "search"
        assert result.merged_args == {"q": "cats"}
        # Only 2 agreeing samples, both say "cats" → 100%
        q_vote = next(fv for fv in result.field_votes if fv.field_name == "q")
        assert q_vote.agreement == 1.0


class TestFieldAgreement:
    def test_mixed_field_agreement(self, make_tool_call):
        calls = [
            make_tool_call("f", {"city": "NYC", "unit": "celsius"}),
            make_tool_call("f", {"city": "NYC", "unit": "fahrenheit"}),
            make_tool_call("f", {"city": "NYC", "unit": "celsius"}),
            make_tool_call("f", {"city": "NYC", "unit": "celsius"}),
        ]
        result = score_tool_calls(calls, threshold=0.75)

        city_vote = next(fv for fv in result.field_votes if fv.field_name == "city")
        unit_vote = next(fv for fv in result.field_votes if fv.field_name == "unit")

        assert city_vote.agreement == 1.0
        assert unit_vote.agreement == 0.75
        assert result.confidence == "high_confidence"

    def test_below_threshold_forced(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "c"}),
        ]
        # x agreement = 2/4 = 50% < 75%
        result = score_tool_calls(calls, threshold=0.75)
        assert result.confidence == "forced"

    def test_threshold_at_boundary(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
        ]
        # x agreement = 3/4 = 75% == 75% → high_confidence
        result = score_tool_calls(calls, threshold=0.75)
        assert result.confidence == "high_confidence"

    def test_custom_threshold(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
        ]
        # x agreement = 2/3 ≈ 66.7%
        assert score_tool_calls(calls, threshold=0.5).confidence == "high_confidence"
        assert score_tool_calls(calls, threshold=0.7).confidence == "forced"


class TestTieHandling:
    def test_name_tie_selects_one(self, make_tool_call):
        calls = [
            make_tool_call("alpha", {"x": 1}),
            make_tool_call("beta", {"x": 2}),
        ]
        result = score_tool_calls(calls)
        assert result.tool_name in ("alpha", "beta")
        assert result.tool_name_vote_count == 1

    def test_field_tie_selects_one(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
        ]
        result = score_tool_calls(calls)
        x_vote = next(fv for fv in result.field_votes if fv.field_name == "x")
        assert x_vote.winning_value in ("a", "b")
        assert x_vote.agreement == 0.5


class TestNestedArgs:
    def test_nested_dict_args(self, make_tool_call):
        calls = [
            make_tool_call("f", {"config": {"nested": True, "count": 5}}),
            make_tool_call("f", {"config": {"nested": True, "count": 5}}),
            make_tool_call("f", {"config": {"nested": False, "count": 5}}),
        ]
        result = score_tool_calls(calls)

        # 2/3 ≈ 66.7% < 75% default threshold → forced
        assert result.confidence == "forced"
        config_vote = next(fv for fv in result.field_votes if fv.field_name == "config")
        assert config_vote.agreement == pytest.approx(2 / 3)

    def test_list_args(self, make_tool_call):
        calls = [
            make_tool_call("f", {"tags": ["a", "b"]}),
            make_tool_call("f", {"tags": ["a", "b"]}),
            make_tool_call("f", {"tags": ["a", "c"]}),
        ]
        result = score_tool_calls(calls)
        tags_vote = next(fv for fv in result.field_votes if fv.field_name == "tags")
        assert tags_vote.vote_count == 2


class TestMissingFields:
    def test_sparse_args(self, make_tool_call):
        calls = [
            make_tool_call("f", {"a": 1, "b": 2}),
            make_tool_call("f", {"a": 1}),
            make_tool_call("f", {"a": 1, "b": 2}),
        ]
        result = score_tool_calls(calls)

        a_vote = next(fv for fv in result.field_votes if fv.field_name == "a")
        b_vote = next(fv for fv in result.field_votes if fv.field_name == "b")

        assert a_vote.total_votes == 3
        assert b_vote.total_votes == 2  # only 2 samples had "b"

    def test_no_args(self, make_tool_call):
        calls = [
            make_tool_call("f", {}),
            make_tool_call("f", {}),
        ]
        result = score_tool_calls(calls)
        assert result.field_votes == []
        assert result.merged_args == {}
        assert result.confidence == "high_confidence"


class TestSingleSample:
    def test_single_sample_high_confidence(self, make_tool_call):
        calls = [make_tool_call("f", {"x": 42})]
        result = score_tool_calls(calls, threshold=0.75)

        assert result.tool_name == "f"
        assert result.confidence == "high_confidence"
        assert result.num_samples == 1

    def test_single_sample_selected_index(self, make_tool_call):
        calls = [make_tool_call("f", {"x": 42})]
        result = score_tool_calls(calls)
        assert result.selected_index == 0


class TestSelectedIndex:
    def test_selected_index_matches_merged(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a", "y": "1"}),
            make_tool_call("f", {"x": "a", "y": "2"}),
            make_tool_call("f", {"x": "a", "y": "1"}),
        ]
        result = score_tool_calls(calls)

        # Merged should be x=a, y=1 → index 0 or 2
        assert result.selected_index in (0, 2)


class TestToolNameNormalization:
    def test_normalize_basic(self):
        assert normalize_tool_name("get_weather") == "get_weather"

    def test_normalize_case(self):
        assert normalize_tool_name("GET_WEATHER") == "get_weather"
        assert normalize_tool_name("Get_Weather") == "get_weather"

    def test_normalize_hyphens(self):
        assert normalize_tool_name("get-weather") == "get_weather"

    def test_normalize_mixed(self):
        assert normalize_tool_name("Get-Weather") == "get_weather"
        assert normalize_tool_name("GET--WEATHER") == "get_weather"

    def test_normalize_whitespace(self):
        assert normalize_tool_name("  get_weather  ") == "get_weather"

    def test_normalized_names_match_in_voting(self, make_tool_call):
        calls = [
            make_tool_call("get_weather", {"city": "NYC"}),
            make_tool_call("Get_Weather", {"city": "NYC"}),
            make_tool_call("get-weather", {"city": "NYC"}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name_vote_count == 3
        assert not result.tool_name_is_tie
        assert result.confidence == "high_confidence"

    def test_normalized_names_exclude_dissenter(self, make_tool_call):
        calls = [
            make_tool_call("get_weather", {"city": "NYC"}),
            make_tool_call("GET_WEATHER", {"city": "NYC"}),
            make_tool_call("search_api", {"city": "LA"}),
        ]
        result = score_tool_calls(calls)

        assert normalize_tool_name(result.tool_name) == "get_weather"
        assert result.tool_name_vote_count == 2
        assert result.merged_args == {"city": "NYC"}


class TestToolNameTieForced:
    def test_tie_always_forced(self, make_tool_call):
        """Tool-name tie → forced regardless of per-field agreement."""
        calls = [
            make_tool_call("alpha", {"x": "same"}),
            make_tool_call("beta", {"x": "same"}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name_is_tie
        assert result.confidence == "forced"

    def test_three_way_tie_forced(self, make_tool_call):
        calls = [
            make_tool_call("a", {"x": 1}),
            make_tool_call("b", {"x": 1}),
            make_tool_call("c", {"x": 1}),
        ]
        result = score_tool_calls(calls)

        assert result.tool_name_is_tie
        assert result.confidence == "forced"

    def test_no_tie_with_clear_majority(self, make_tool_call):
        calls = [
            make_tool_call("search", {"q": "a"}),
            make_tool_call("search", {"q": "a"}),
            make_tool_call("lookup", {"q": "a"}),
        ]
        result = score_tool_calls(calls)

        assert not result.tool_name_is_tie
        assert result.confidence == "high_confidence"


class TestAbstainFlag:
    def test_abstain_returns_none_on_forced(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
        ]
        result = score_tool_calls(calls, threshold=0.75, allow_abstain=True)
        assert result is None

    def test_abstain_returns_result_on_high_confidence(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "a"}),
        ]
        result = score_tool_calls(calls, threshold=0.75, allow_abstain=True)
        assert result is not None
        assert result.confidence == "high_confidence"

    def test_no_abstain_by_default(self, make_tool_call):
        calls = [
            make_tool_call("f", {"x": "a"}),
            make_tool_call("f", {"x": "b"}),
        ]
        result = score_tool_calls(calls, threshold=0.75)
        assert result is not None
        assert result.confidence == "forced"

    def test_abstain_on_name_tie(self, make_tool_call):
        calls = [
            make_tool_call("alpha", {"x": "same"}),
            make_tool_call("beta", {"x": "same"}),
        ]
        result = score_tool_calls(calls, allow_abstain=True)
        assert result is None
