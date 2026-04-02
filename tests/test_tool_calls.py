"""Tests for tool call handling in scaling algorithms."""

import pytest

from its_hub.algorithms.self_consistency import SelfConsistency, SelfConsistencyResult
from tests.mocks.language_models import SimpleMockLanguageModel


class ToolCallMockLanguageModel(SimpleMockLanguageModel):
    """Mock language model that returns responses with tool calls."""

    def __init__(self):
        super().__init__([])  # No predefined responses
        self.call_count = 0

    def generate(self, messages, **kwargs):
        if isinstance(messages[0], list):
            # Batch generation - return responses with tool calls and content
            responses = []
            for i in range(len(messages)):
                if i % 2 == 0:
                    # Response with tool calls and content
                    responses.append(
                        {
                            "role": "assistant",
                            "content": "I need to calculate this. The answer is 42.",
                            "tool_calls": [
                                {
                                    "id": f"call_{i}",
                                    "type": "function",
                                    "function": {"name": "calculate"},
                                }
                            ],
                        }
                    )
                else:
                    # Response with only content
                    responses.append({
                        "role": "assistant",
                        "content": "The answer is 42."
                    })
            return responses
        else:
            # Single generation
            return {"role": "assistant", "content": "The answer is 42."}


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
        result = sc.infer(mock_lm, "What is 6 * 7?", budget=4, return_response_only=False)

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
        assert result.the_one["content"] in ["I need to calculate this. The answer is 42.", "The answer is 42."]

    def test_self_consistency_with_empty_content_and_tool_calls(self):
        """Test self-consistency raises error when all responses have tool calls but tool_vote is not set."""

        class EmptyContentToolCallMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    # Return responses with tool calls but empty/no content
                    responses = []
                    for i in range(len(messages)):
                        responses.append(
                            {
                                "role": "assistant",
                                "content": "",  # Empty content
                                "tool_calls": [
                                    {
                                        "id": f"call_{i}",
                                        "type": "function",
                                        "function": {"name": "search"},
                                    }
                                ],
                            }
                        )
                    return responses
                else:
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_single",
                                "type": "function",
                                "function": {"name": "search"},
                            }
                        ],
                    }

        mock_lm = EmptyContentToolCallMock()

        def identity_projection(text):
            return text

        sc = SelfConsistency(identity_projection)

        # Should raise ValueError when all responses have tool calls but tool_vote is not set
        with pytest.raises(
            ValueError, match="No eligible responses found after filtering"
        ):
            sc.infer(
                mock_lm, "Search for information", budget=3, return_response_only=False
            )


class TestToolCallVotingBackwardCompatibility:
    """Test that tool vote features maintain backward compatibility."""

    def test_existing_behavior_unchanged_without_tool_vote(self):
        """Test that existing code without tool_vote parameter works unchanged."""
        mock_lm = ToolCallMockLanguageModel()

        # Old way of initializing - should work exactly as before
        def extract_number(text):
            import re

            numbers = re.findall(r"\d+", text)
            return numbers[0] if numbers else ""

        sc = SelfConsistency(extract_number)  # No tool_vote parameter
        result = sc.infer(mock_lm, "What is 6 * 7?", budget=4, return_response_only=False)

        # Should behave exactly as before - vote on content only
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

        class ToolNameMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])  # No predefined responses needed

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Calculating...",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 1},
                                    },
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
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 2},
                                    },
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
                                    "function": {
                                        "name": "search",
                                        "arguments": {"q": "test"},
                                    },
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
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 3},
                                    },
                                }
                            ],
                        },
                    ]

        mock_lm = ToolNameMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote on tool names - "calculate" appears 3 times, "search" 1 time
        # So "calculate" should win
        assert isinstance(result, SelfConsistencyResult)
        assert "calculate" in str(result.response_counts)
        assert result.response_counts["calculate"] == 3
        assert result.response_counts["search"] == 1

    def test_tool_args_voting(self):
        """Test voting on tool arguments only."""

        class ToolArgsMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
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

        mock_lm = ToolArgsMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_args")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote on arguments - (("x", 1), ("y", 2)) appears 3 times (converted to tuple)
        expected_args = (("x", 1), ("y", 2))
        assert result.response_counts[expected_args] == 3

    def test_tool_hierarchical_voting(self):
        """Test hierarchical voting: tool name first, then arguments."""

        class HierarchicalMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 1},
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
                                        "name": "calculate",
                                        "arguments": {"x": 2},
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
                                        "name": "search",
                                        "arguments": {"q": "test"},
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
                                        "name": "calculate",
                                        "arguments": {"x": 1},
                                    },
                                }
                            ],
                        },
                    ]

        mock_lm = HierarchicalMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_hierarchical")
        result = sc.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Should vote hierarchically - ("calculate", (("x", 1),)) appears twice and should win
        assert isinstance(result, SelfConsistencyResult)
        expected_winner = ("calculate", (("x", 1),))
        assert result.response_counts[expected_winner] == 2

    def test_exclude_args_functionality(self):
        """Test excluding certain argument names from voting."""

        class ExcludeArgsMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "type": "function",
                                    "function": {
                                        "name": "func",
                                        "arguments": {
                                            "x": 1,
                                            "timestamp": "123",
                                            "id": "abc",
                                        },
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
                                        "arguments": {
                                            "x": 1,
                                            "timestamp": "456",
                                            "id": "def",
                                        },
                                    },
                                }
                            ],
                        },
                    ]

        mock_lm = ExcludeArgsMock()
        sc = SelfConsistency(
            lambda x: x, tool_vote="tool_args", exclude_args=["timestamp", "id"]
        )
        result = sc.infer(mock_lm, "Test prompt", budget=2, return_response_only=False)

        # After excluding timestamp and id, both should have (("x", 1),) and be considered identical
        expected_args = (("x", 1),)
        assert result.response_counts[expected_args] == 2

    def test_fallback_to_content_voting_when_no_tool_calls(self):
        """Test that it falls back to content voting when responses have no tool calls."""

        class NoToolCallsMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {"role": "assistant", "content": "Answer: 42"},
                        {"role": "assistant", "content": "Answer: 42"},
                        {"role": "assistant", "content": "Answer: 24"},
                    ]

        def extract_answer(text):
            import re

            match = re.search(r"Answer: (\d+)", text)
            return match.group(1) if match else None

        mock_lm = NoToolCallsMock()
        sc = SelfConsistency(extract_answer, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # Should fall back to content voting since no tool calls
        assert result.response_counts["42"] == 2
        assert result.response_counts["24"] == 1
        assert result.the_one["content"] == "Answer: 42"  # Returns full content, not extracted value

    def test_tool_flat_all_single_tool(self):
        """Test tool_flat_all with single tool calls behaves like tool_hierarchical."""

        class SingleToolMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
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
                                        "name": "events_list",
                                        "arguments": {},
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
                                        "name": "pods_log",
                                        "arguments": {"namespace": "demo-oom"},
                                    },
                                }
                            ],
                        },
                    ]

        mock_lm = SingleToolMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        # pods_log(namespace=demo-oom) appears 2/3 times, should win
        winner = result.the_one
        assert winner["tool_calls"][0]["function"]["name"] == "pods_log"

    def test_tool_flat_all_multiple_tools_order_independent(self):
        """Test tool_flat_all treats [A, B] and [B, A] as the same signature."""

        class MultiToolOrderMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {"id": "1a", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x"}}},
                                {"id": "1b", "type": "function", "function": {"name": "events_list", "arguments": {}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test2",
                            "tool_calls": [
                                {"id": "2a", "type": "function", "function": {"name": "events_list", "arguments": {}}},
                                {"id": "2b", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x"}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test3",
                            "tool_calls": [
                                {"id": "3a", "type": "function", "function": {"name": "resources_get", "arguments": {}}},
                            ],
                        },
                    ]

        mock_lm = MultiToolOrderMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # [pods_log, events_list] and [events_list, pods_log] should be the same signature
        # That signature appears 2/3, should win over resources_get (1/3)
        winner = result.the_one
        assert len(winner["tool_calls"]) == 2
        tool_names = {tc["function"]["name"] for tc in winner["tool_calls"]}
        assert tool_names == {"pods_log", "events_list"}

    def test_tool_flat_all_different_tool_sets(self):
        """Test tool_flat_all distinguishes [A] from [A, B]."""

        class DifferentSetsMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {"id": "1", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x"}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test2",
                            "tool_calls": [
                                {"id": "2a", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x"}}},
                                {"id": "2b", "type": "function", "function": {"name": "events_list", "arguments": {}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test3",
                            "tool_calls": [
                                {"id": "3a", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x"}}},
                                {"id": "3b", "type": "function", "function": {"name": "events_list", "arguments": {}}},
                            ],
                        },
                    ]

        mock_lm = DifferentSetsMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # [pods_log + events_list] appears 2/3, [pods_log alone] appears 1/3
        winner = result.the_one
        assert len(winner["tool_calls"]) == 2

    def test_tool_flat_all_exclude_args(self):
        """Test exclude_args works with tool_flat_all across all tool calls."""

        class ExcludeArgsFlatMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {"id": "1", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x", "timestamp": "111"}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test2",
                            "tool_calls": [
                                {"id": "2", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "x", "timestamp": "222"}}},
                            ],
                        },
                    ]

        mock_lm = ExcludeArgsFlatMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all", exclude_args=["timestamp"])
        result = sc.infer(mock_lm, "Test prompt", budget=2, return_response_only=False)

        # After excluding timestamp, both should have the same signature
        # All values in the Counter should be the same frozenset
        assert len(result.response_counts) == 1

    def test_tool_flat_all_uses_flat_voting(self):
        """Test that tool_flat_all uses flat Counter voting, not hierarchical."""

        class FlatVoteMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Test1",
                            "tool_calls": [
                                {"id": "1", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "a"}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test2",
                            "tool_calls": [
                                {"id": "2", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "a"}}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "Test3",
                            "tool_calls": [
                                {"id": "3", "type": "function", "function": {"name": "pods_log", "arguments": {"ns": "b"}}},
                            ],
                        },
                    ]

        mock_lm = FlatVoteMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_flat_all")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # response_counts should use frozenset keys (flat), not tuple keys (hierarchical)
        for key in result.response_counts:
            assert isinstance(key, frozenset), f"Expected frozenset key, got {type(key)}"

    def test_mixed_responses_with_and_without_tool_calls(self):
        """Test behavior when some responses have tool calls and others don't."""

        class MixedMock(SimpleMockLanguageModel):
            def __init__(self):
                super().__init__([])

            def generate(self, messages, **kwargs):
                if isinstance(messages[0], list):
                    return [
                        {
                            "role": "assistant",
                            "content": "Using calculator",
                            "tool_calls": [
                                {
                                    "id": "1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 1},
                                    },
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
                                    "function": {
                                        "name": "calculate",
                                        "arguments": {"x": 1},
                                    },
                                }
                            ],
                        },
                    ]

        mock_lm = MixedMock()
        sc = SelfConsistency(lambda x: x, tool_vote="tool_name")
        result = sc.infer(mock_lm, "Test prompt", budget=3, return_response_only=False)

        # Should use tool voting since at least one response has tool calls
        # Responses without tool calls are filtered out entirely
        assert "calculate" in result.response_counts
        assert result.response_counts["calculate"] == 2
        assert len(result.response_counts) == 1  # Only tool responses are counted
