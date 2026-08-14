"""Tests for tool call handling in scaling algorithms."""

import pytest

from its_hub.core.algorithms.self_consistency import (
    SelfConsistency,
    SelfConsistencyResult,
)


class ToolCallMockLanguageModel:
    """Mock language model that returns alternating responses with/without tool calls."""

    def __init__(self):
        self.call_count = 0

    def _make_response(self, idx):
        if idx % 2 == 0:
            return {
                "role": "assistant",
                "content": "I need to calculate this. The answer is 42.",
                "tool_calls": [
                    {
                        "id": f"call_{idx}",
                        "type": "function",
                        "function": {"name": "calculate"},
                    }
                ],
            }
        else:
            return {"role": "assistant", "content": "The answer is 42."}

    async def agenerate_single(self, messages, **kwargs):
        idx = self.call_count
        self.call_count += 1
        return self._make_response(idx)


def _make_fixed_list_mock(responses: list[dict]):
    """Create a mock LM that cycles through a fixed list of responses."""

    class FixedListMock:
        def __init__(self):
            self.call_count = 0

        async def agenerate_single(self, messages, **kwargs):
            idx = self.call_count % len(responses)
            self.call_count += 1
            return responses[idx]

    return FixedListMock()


class TestToolCallHandling:
    """Test tool call handling across scaling algorithms."""

    def test_self_consistency_with_tool_calls(self):
        """Test that self-consistency handles tool calls without breaking."""
        mock_lm = ToolCallMockLanguageModel()

        # Simple projection function that extracts numbers
        def extract_number(text):
            import re

            numbers = re.findall(r"\d+", text)
            return numbers[0] if numbers else ""

        sc = SelfConsistency(extract_number)

        # This should not throw a NotImplementedError and should handle tool calls gracefully
        result = sc.infer(
            mock_lm, "What is 6 * 7?", budget=4, return_response_only=False
        )

        assert isinstance(result, SelfConsistencyResult)
        # Should vote on content only, ignoring tool calls
        assert result.the_one["content"] in [
            "I need to calculate this. The answer is 42.",
            "The answer is 42.",
        ]
        assert len(result.responses) == 4

        # Verify that responses contain the full message dicts
        for response in result.responses:
            assert isinstance(response, dict)
            assert "role" in response
            assert "content" in response
            assert "42" in response["content"]

    def test_self_consistency_tool_calls_content_voting(self):
        """Test that self-consistency votes on content, not tool calls."""
        mock_lm = ToolCallMockLanguageModel()

        # Identity projection function - just returns the content as-is
        def identity_projection(text):
            return text

        sc = SelfConsistency(identity_projection)
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should have 2 responses of each type
        response_counts = {}
        for response in result.responses:
            content = response["content"]
            response_counts[content] = response_counts.get(content, 0) + 1

        # Both response types should appear
        assert "I need to calculate this. The answer is 42." in response_counts
        assert "The answer is 42." in response_counts

        # The selected response should be one of the content strings
        assert result.the_one["content"] in [
            "I need to calculate this. The answer is 42.",
            "The answer is 42.",
        ]

    def test_self_consistency_with_empty_content_and_tool_calls(self):
        """Content-only voting (tool_vote=None) raises when every response is a tool call."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "search"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "search"},
                        },
                    ],
                },
            ]
        )

        def identity_projection(text):
            return text

        # tool_vote=None forces content-only voting; with every response a tool
        # call there is nothing to vote on.
        sc = SelfConsistency(identity_projection, tool_vote=None)

        with pytest.raises(
            ValueError, match="No eligible responses found after filtering"
        ):
            sc.infer(
                mock_lm, "Search for information", budget=3, return_response_only=False
            )

    def test_default_tool_vote_handles_all_tool_calls(self):
        """With the default tool_vote, all-tool-call responses vote instead of raising."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "search"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "lookup"},
                        },
                    ],
                },
            ]
        )

        # No tool_vote passed -> DEFAULT_TOOL_VOTE ("tool_hierarchical") applies.
        sc = SelfConsistency()
        assert sc.tool_vote == "tool_hierarchical"

        result = sc.infer(
            mock_lm, "Search for information", budget=3, return_response_only=False
        )

        # "search" wins 2-1 over "lookup"; the winner is a search tool call.
        assert result.the_one["tool_calls"][0]["function"]["name"] == "search"


class TestToolCallVotingBackwardCompatibility:
    """Test that tool vote features maintain backward compatibility."""

    def test_existing_behavior_unchanged_without_tool_vote(self):
        """Positional-arg construction still works; the '42' answer is selected."""
        mock_lm = ToolCallMockLanguageModel()

        # Old way of initializing (projection func only) still works. Note the
        # default tool_vote now routes tool-majority batches to tool voting, so
        # the selected response may be a tool call rather than a text message.
        def extract_number(text):
            import re

            numbers = re.findall(r"\d+", text)
            return numbers[0] if numbers else ""

        sc = SelfConsistency(extract_number)  # No tool_vote parameter
        result = sc.infer(
            mock_lm, "What is 6 * 7?", budget=4, return_response_only=False
        )

        assert isinstance(result, SelfConsistencyResult)
        assert result.the_one["content"] in [
            "I need to calculate this. The answer is 42.",
            "The answer is 42.",
        ]
        assert all("42" in response["content"] for response in result.responses)


class TestToolCallVoting:
    """Test tool call voting functionality."""

    def test_tool_name_voting(self):
        """Test voting on tool function names only."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Calculating...",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 1}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Computing...",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 2}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Processing...",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {"name": "search", "arguments": {"q": "test"}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Working...",
                    "tool_calls": [
                        {
                            "id": "4",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 3}},
                        }
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote on tool names - "calculate" appears 3 times, "search" 1 time
        assert isinstance(result, SelfConsistencyResult)
        assert "calculate" in str(result.response_counts)
        assert result.response_counts["calculate"] == 3
        assert result.response_counts["search"] == 1

    def test_tool_args_voting(self):
        """Test voting on tool arguments only."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "func1",
                                "arguments": {"x": 1, "y": 2},
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {
                                "name": "func2",
                                "arguments": {"x": 1, "y": 2},
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {
                                "name": "func3",
                                "arguments": {"a": 3, "b": 4},
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test4",
                    "tool_calls": [
                        {
                            "id": "4",
                            "type": "function",
                            "function": {
                                "name": "func4",
                                "arguments": {"x": 1, "y": 2},
                            },
                        }
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_args")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote on arguments - (("x", 1), ("y", 2)) appears 3 times (converted to tuple)
        expected_args = (("x", 1), ("y", 2))
        assert result.response_counts[expected_args] == 3

    def test_tool_hierarchical_voting(self):
        """Test hierarchical voting: tool name first, then arguments."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 1}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 2}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {"name": "search", "arguments": {"q": "test"}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test4",
                    "tool_calls": [
                        {
                            "id": "4",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 1}},
                        }
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_hierarchical")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote hierarchically - ("calculate", (("x", 1),)) appears twice and should win
        assert isinstance(result, SelfConsistencyResult)
        expected_winner = ("calculate", (("x", 1),))
        assert result.response_counts[expected_winner] == 2

    def test_exclude_args_functionality(self):
        """Test excluding certain argument names from voting."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "func",
                                "arguments": {"x": 1, "timestamp": "123", "id": "abc"},
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {
                                "name": "func",
                                "arguments": {"x": 1, "timestamp": "456", "id": "def"},
                            },
                        }
                    ],
                },
            ]
        )

        sc = SelfConsistency(
            lambda x: x, tool_vote="tool_args", exclude_args=["timestamp", "id"]
        )
        result = sc.infer(mock_lm, "Test prompt", budget=2, return_response_only=False)

        # After excluding timestamp and id, both should have (("x", 1),) and be considered identical
        expected_args = (("x", 1),)
        assert result.response_counts[expected_args] == 2

    def test_fallback_to_content_voting_when_no_tool_calls(self):
        """Test that it falls back to content voting when responses have no tool calls."""
        mock_lm = _make_fixed_list_mock(
            [
                {"role": "assistant", "content": "Answer: 42"},
                {"role": "assistant", "content": "Answer: 42"},
                {"role": "assistant", "content": "Answer: 24"},
            ]
        )

        def extract_answer(text):
            import re

            match = re.search(r"Answer: (\d+)", text)
            return match.group(1) if match else None

        sc = SelfConsistency(extract_answer, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # Should fall back to content voting since no tool calls
        assert result.response_counts["42"] == 2
        assert result.response_counts["24"] == 1
        assert (
            result.the_one["content"] == "Answer: 42"
        )  # Returns full content, not extracted value

    def test_tool_flat_all_single_tool(self):
        """Test tool_flat_all with single tool calls behaves like tool_hierarchical."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "pods_log",
                                "arguments": {"namespace": "demo-oom"},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "events_list", "arguments": {}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {
                                "name": "pods_log",
                                "arguments": {"namespace": "demo-oom"},
                            },
                        },
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        winner = result.the_one
        assert winner["tool_calls"][0]["function"]["name"] == "pods_log"

    def test_tool_flat_all_multiple_tools_order_independent(self):
        """Test tool_flat_all treats [A, B] and [B, A] as the same signature."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1a",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "x"}},
                        },
                        {
                            "id": "1b",
                            "type": "function",
                            "function": {"name": "events_list", "arguments": {}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2a",
                            "type": "function",
                            "function": {"name": "events_list", "arguments": {}},
                        },
                        {
                            "id": "2b",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "x"}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3a",
                            "type": "function",
                            "function": {"name": "resources_get", "arguments": {}},
                        },
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        winner = result.the_one
        assert len(winner["tool_calls"]) == 2
        tool_names = {tc["function"]["name"] for tc in winner["tool_calls"]}
        assert tool_names == {"pods_log", "events_list"}

    def test_tool_flat_all_different_tool_sets(self):
        """Test tool_flat_all distinguishes [A] from [A, B]."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "x"}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2a",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "x"}},
                        },
                        {
                            "id": "2b",
                            "type": "function",
                            "function": {"name": "events_list", "arguments": {}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3a",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "x"}},
                        },
                        {
                            "id": "3b",
                            "type": "function",
                            "function": {"name": "events_list", "arguments": {}},
                        },
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        winner = result.the_one
        assert len(winner["tool_calls"]) == 2

    def test_tool_flat_all_exclude_args(self):
        """Test exclude_args works with tool_flat_all across all tool calls."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "pods_log",
                                "arguments": {"ns": "x", "timestamp": "111"},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {
                                "name": "pods_log",
                                "arguments": {"ns": "x", "timestamp": "222"},
                            },
                        },
                    ],
                },
            ]
        )

        sc = SelfConsistency(
            lambda x: x, tool_vote="tool_flat_all", exclude_args=["timestamp"]
        )
        result = sc.infer(mock_lm, "Test prompt", budget=2, return_response_only=False)

        assert len(result.response_counts) == 1

    def test_tool_flat_all_uses_flat_voting(self):
        """Test that tool_flat_all uses flat Counter voting, not hierarchical."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Test1",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "a"}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test2",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "a"}},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Test3",
                    "tool_calls": [
                        {
                            "id": "3",
                            "type": "function",
                            "function": {"name": "pods_log", "arguments": {"ns": "b"}},
                        },
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        for key in result.response_counts:
            assert isinstance(key, frozenset), (
                f"Expected frozenset key, got {type(key)}"
            )

    def test_mixed_responses_with_and_without_tool_calls(self):
        """Test behavior when some responses have tool calls and others don't."""
        mock_lm = _make_fixed_list_mock(
            [
                {
                    "role": "assistant",
                    "content": "Using calculator",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 1}},
                        }
                    ],
                },
                {"role": "assistant", "content": "Direct answer: 42"},
                {
                    "role": "assistant",
                    "content": "Let me compute",
                    "tool_calls": [
                        {
                            "id": "2",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": {"x": 1}},
                        }
                    ],
                },
            ]
        )

        sc = SelfConsistency(lambda x: x, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # Should use tool voting since at least one response has tool calls
        # Responses without tool calls are filtered out entirely
        assert "calculate" in result.response_counts
        assert result.response_counts["calculate"] == 2
        assert len(result.response_counts) == 1  # Only tool responses are counted
