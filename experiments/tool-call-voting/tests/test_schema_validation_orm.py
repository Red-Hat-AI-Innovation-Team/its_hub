"""Tests for SchemaValidationORM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema_validation_orm import SchemaValidationORM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                    "days": {"type": "integer", "description": "Forecast days"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]


def _make_conversation(tool_calls: list[dict] | None = None, content: str | None = None) -> list:
    msgs = [{"role": "user", "content": "What's the weather?"}]
    assistant = {"role": "assistant", "content": content}
    if tool_calls is not None:
        assistant["tool_calls"] = tool_calls
    msgs.append(assistant)
    return msgs


def _tc(name: str, args: dict) -> dict:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class TestCorrectToolCall:
    def test_perfect_score(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("get_weather", {"location": "NYC"})])
        score = orm.score(conv)
        assert score == 1.0

    def test_all_params(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(
            [_tc("get_weather", {"location": "NYC", "unit": "celsius", "days": 3})]
        )
        assert orm.score(conv) == 1.0


class TestWrongFunctionName:
    def test_unknown_function(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("nonexistent_func", {"location": "NYC"})])
        score = orm.score(conv)
        assert score == 0.0


class TestMissingRequired:
    def test_missing_required_param(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("get_weather", {"unit": "celsius"})])
        score = orm.score(conv)
        assert 0.0 < score < 1.0

    def test_missing_all_required(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("search_flights", {})])
        score = orm.score(conv)
        assert score < 1.0


class TestWrongType:
    def test_string_instead_of_int(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(
            [_tc("get_weather", {"location": "NYC", "days": "five"})]
        )
        score = orm.score(conv)
        assert score < 1.0

    def test_int_instead_of_string(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("get_weather", {"location": 42})])
        score = orm.score(conv)
        assert score < 1.0


class TestInvalidEnum:
    def test_invalid_enum_value(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(
            [_tc("get_weather", {"location": "NYC", "unit": "kelvin"})]
        )
        score = orm.score(conv)
        assert score < 1.0

    def test_valid_enum_value(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(
            [_tc("get_weather", {"location": "NYC", "unit": "celsius"})]
        )
        assert orm.score(conv) == 1.0


class TestUnexpectedParams:
    def test_extra_parameter(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(
            [_tc("get_weather", {"location": "NYC", "color": "blue"})]
        )
        score = orm.score(conv)
        assert score < 1.0


class TestBatchScoring:
    def test_batch(self):
        orm = SchemaValidationORM(TOOLS)
        good = _make_conversation([_tc("get_weather", {"location": "NYC"})])
        bad = _make_conversation([_tc("nonexistent", {})])
        scores = orm.score([good, bad])
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert scores[0] == 1.0
        assert scores[1] == 0.0


class TestNoToolCalls:
    def test_no_tool_calls(self):
        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation(content="Just some text")
        score = orm.score(conv)
        assert score == 0.0

    def test_empty_conversation(self):
        orm = SchemaValidationORM(TOOLS)
        assert orm.score([]) == 0.0


class TestAsyncScore:
    def test_async_delegates_to_sync(self):
        import asyncio

        orm = SchemaValidationORM(TOOLS)
        conv = _make_conversation([_tc("get_weather", {"location": "NYC"})])
        score = asyncio.new_event_loop().run_until_complete(orm.ascore(conv))
        assert score == 1.0


class TestBfclTypeNormalization:
    """BFCL schemas use Python type names (dict, int, str) instead of JSON schema types."""

    def test_dict_type(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "func",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "val": {"type": "int"},
                        },
                        "required": ["val"],
                    },
                },
            }
        ]
        orm = SchemaValidationORM(tools)
        conv = _make_conversation([_tc("func", {"val": 5})])
        assert orm.score(conv) == 1.0

    def test_args_as_dict(self):
        """Test that arguments provided as dict (not JSON string) also work."""
        orm = SchemaValidationORM(TOOLS)
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"location": "NYC"}},
        }
        conv = _make_conversation([tc])
        assert orm.score(conv) == 1.0
