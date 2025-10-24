"""Clean tests for algorithms with improved organization and shared utilities."""

from collections import Counter
from copy import deepcopy

import pytest

from its_hub.algorithms.beam_search import BeamSearch, BeamSearchResult, Path
from its_hub.algorithms.bon import BestOfN, BestOfNResult
from its_hub.algorithms.particle_gibbs import (
    Particle,
    ParticleFiltering,
    ParticleFilteringResult,
    ParticleGibbs,
    ParticleGibbsResult,
    SelectionMethod,
)
from its_hub.algorithms.self_consistency import (
    SelfConsistency,
    SelfConsistencyResult,
    _select_hierarchical_most_common_or_random,
    _select_most_common_or_random,
    create_regex_projection_function,
)
from its_hub.lms import StepGeneration
from its_hub.types import ChatMessage, ChatMessages

# Import from our new shared utilities
from tests.mocks.language_models import StepMockLanguageModel
from tests.mocks.reward_models import MockOutcomeRewardModel, MockProcessRewardModel


class TestSelfConsistency:
    """Test the self-consistency algorithm utility functions."""

    @pytest.mark.parametrize("test_list,expected_counts,expected_element", [
        (['a', 'b', 'a', 'c', 'a'], Counter({'a': 3, 'b': 1, 'c': 1}), 'a'),
        (['a', 'b', 'a', 'b', 'c'], Counter({'a': 2, 'b': 2, 'c': 1}), ['a', 'b']),
        (['a', 'b', 'c', 'd'], Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1}), ['a', 'b', 'c', 'd']),
    ])
    def test_select_most_common_or_random(self, test_list, expected_counts, expected_element):
        """Test selection of most common element with various scenarios."""
        counts, selected_index = _select_most_common_or_random(test_list)

        assert counts == expected_counts

        if isinstance(expected_element, list):
            # Multiple possible winners
            assert test_list[selected_index] in expected_element
        else:
            # Single winner
            assert test_list[selected_index] == expected_element

    @pytest.mark.parametrize("test_tuples,expected_winner", [
        # Test case from issue: "a" is most common at level 0, "1" and "2" equally common at level 1
        ([("a", "1"), ("a", "2"), ("b", "1"), ("c", "1")], [("a", "1"), ("a", "2")]),
        # Clear hierarchy winner
        ([("a", "1"), ("a", "1"), ("b", "1")], [("a", "1")]),
        # All different at level 0
        ([("a", "1"), ("b", "2"), ("c", "3")], [("a", "1"), ("b", "2"), ("c", "3")]),
        # Same at level 0, different at level 1
        ([("a", "1"), ("a", "2"), ("a", "1")], [("a", "1")]),
        # Different depths
        ([("a", "1", "x"), ("a", "1", "y"), ("a", "2"), ("b", "1")], [("a", "1", "x"), ("a", "1", "y")]),
        # Single element tuples
        ([("a",), ("b",), ("a",)], [("a",)]),
    ])
    def test_select_hierarchical_most_common_or_random(self, test_tuples, expected_winner):
        """Test hierarchical selection of most common element with various scenarios."""
        counts, selected_index = _select_hierarchical_most_common_or_random(test_tuples)

        # Check that the selected tuple is one of the expected winners
        selected_tuple = test_tuples[selected_index]
        assert selected_tuple in expected_winner

        # Check that counts include all tuples
        assert sum(counts.values()) == len(test_tuples)
        for tuple_item in test_tuples:
            assert tuple_item in counts

    def test_select_hierarchical_empty_list(self):
        """Test hierarchical selection with empty list."""
        with pytest.raises(ValueError, match="Cannot select from empty list"):
            _select_hierarchical_most_common_or_random([])

    def test_self_consistency_flat_projection(self):
        """Test SelfConsistency with flat (string) projection function."""
        mock_lm = StepMockLanguageModel(["answer1", "answer2", "answer1", "answer3"])

        def flat_projection(response: str) -> str:
            return response[-1]  # Last character

        sc = SelfConsistency(flat_projection)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        assert isinstance(result.response_counts, Counter)
        # "1" appears twice (from "answer1"), should be selected
        assert result.the_one["content"] in ["answer1", "answer1"]
        assert result.the_one["role"] == "assistant"

    def test_self_consistency_hierarchical_projection(self):
        """Test SelfConsistency with hierarchical (tuple) projection function."""
        mock_lm = StepMockLanguageModel(["a1", "a2", "b1", "c1"])

        def hierarchical_projection(response: str) -> tuple:
            return (response[0], response[1])  # First char, second char as tuple

        sc = SelfConsistency(hierarchical_projection)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        assert isinstance(result.response_counts, Counter)
        # "a" is most common at level 0, so should select either "a1" or "a2"
        assert result.the_one["content"] in ["a1", "a2"]
        assert result.the_one["role"] == "assistant"

    def test_self_consistency_return_response_only(self):
        """Test SelfConsistency with return_response_only=True."""
        mock_lm = StepMockLanguageModel(["a1", "a2", "b1", "c1"])

        def hierarchical_projection(response: str) -> tuple:
            return (response[0], response[1])

        sc = SelfConsistency(hierarchical_projection)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=True)

        assert isinstance(result, dict)
        assert result["content"] in ["a1", "a2"]  # Should be one of the "a" responses
        assert result["role"] == "assistant"

    def test_self_consistency_with_chat_messages_class(self):
        """Test SelfConsistency with ChatMessages class input."""
        mock_lm = StepMockLanguageModel(["answer1", "answer2", "answer1", "answer3"])

        def flat_projection(response: str) -> str:
            return response[-1]  # Last character

        sc = SelfConsistency(flat_projection)

        # Test with ChatMessages wrapping a string
        chat_messages = ChatMessages("test prompt")
        result = sc.infer(mock_lm, chat_messages, budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        # "1" appears twice (from "answer1"), should be selected
        assert result.the_one["content"] in ["answer1", "answer1"]
        assert result.the_one["role"] == "assistant"

    def test_create_regex_projection_function_single_pattern(self):
        """Test creating projection function from single regex pattern."""
        # Test math answer extraction
        pattern = r'\\boxed\{([^}]+)\}'
        proj_func = create_regex_projection_function(pattern)

        # Test successful extraction
        response1 = "The solution is 42. Therefore, the answer is \\boxed{42}."
        result1 = proj_func(response1)
        assert result1 == ("42",)

        # Test no match
        response2 = "The answer is 42 but not in boxed format."
        result2 = proj_func(response2)
        assert result2 == (None,)

        # Test pattern without capturing groups
        pattern_no_groups = r'\\boxed\{[^}]+\}'
        proj_func_no_groups = create_regex_projection_function(pattern_no_groups)
        result3 = proj_func_no_groups(response1)
        assert result3 == ("\\boxed{42}",)

    def test_create_regex_projection_function_multiple_patterns(self):
        """Test creating projection function from multiple regex patterns."""
        # Test hierarchical extraction: method and answer
        patterns = [
            r'Method:\s*(\w+)',  # Extract method
            r'\\boxed\{([^}]+)\}'  # Extract final answer
        ]
        proj_func = create_regex_projection_function(patterns)

        # Test full match
        response1 = "Method: algebra\n\nSolving step by step:\nx = 5\n\nFinal answer: \\boxed{5}"
        result1 = proj_func(response1)
        assert result1 == ("algebra", "5")

        # Test partial match (only method)
        response2 = "Method: geometry\n\nThe answer is 10 but not boxed."
        result2 = proj_func(response2)
        assert result2 == ("geometry", None)

        # Test partial match (only answer)
        response3 = "Using an unspecified approach.\nAnswer: \\boxed{15}"
        result3 = proj_func(response3)
        assert result3 == (None, "15")

        # Test no matches
        response4 = "Some random text without patterns."
        result4 = proj_func(response4)
        assert result4 == (None, None)

    def test_create_regex_projection_function_case_insensitive(self):
        """Test case-insensitive matching in regex projection function."""
        pattern = r'ANSWER:\s*(\w+)'
        proj_func = create_regex_projection_function(pattern)

        # Test different cases
        responses = [
            "ANSWER: correct",
            "answer: correct",
            "Answer: correct",
            "AnSwEr: correct"
        ]

        for response in responses:
            result = proj_func(response)
            assert result == ("correct",), f"Failed for: {response}"

    def test_create_regex_projection_function_multiline(self):
        """Test regex projection function with multiline and DOTALL flags."""
        pattern = r'Step 1:.*?Result:\s*(\w+)'
        proj_func = create_regex_projection_function(pattern)

        response = """Step 1: Start here
        Do some calculations
        More work here
        Result: success"""

        result = proj_func(response)
        assert result == ("success",)

    def test_create_regex_projection_function_with_self_consistency(self):
        """Test integration of regex projection function with SelfConsistency."""
        # Create a pattern to extract answers from boxed format
        pattern = r'\\boxed\{([^}]+)\}'
        proj_func = create_regex_projection_function(pattern)

        # Mock responses with different answers
        responses = [
            "Solution: \\boxed{42}",
            "Answer: \\boxed{24}",
            "Result: \\boxed{42}",
            "Final: \\boxed{42}"
        ]
        mock_lm = StepMockLanguageModel(responses)

        sc = SelfConsistency(proj_func)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        # "42" appears 3 times, should be selected over "24" (1 time)
        extracted_answer = proj_func(result.the_one["content"])[0]
        assert extracted_answer == "42"

    def test_create_regex_projection_function_hierarchical_with_self_consistency(self):
        """Test hierarchical regex projection function with SelfConsistency."""
        # Create hierarchical patterns: approach and answer
        patterns = [
            r'Approach:\s*(\w+)',
            r'\\boxed\{([^}]+)\}'
        ]
        proj_func = create_regex_projection_function(patterns)

        # Mock responses - "algebra" approach should win, with "42" being most common answer
        responses = [
            "Approach: algebra\nSolution: \\boxed{42}",
            "Approach: algebra\nSolution: \\boxed{24}",
            "Approach: geometry\nSolution: \\boxed{42}",
            "Approach: calculus\nSolution: \\boxed{30}"
        ]
        mock_lm = StepMockLanguageModel(responses)

        sc = SelfConsistency(proj_func)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        # "algebra" appears twice (most common approach), so should select from those
        extracted = proj_func(result.the_one["content"])
        assert extracted[0] == "algebra"  # Should be algebra approach
        assert extracted[1] in ["42", "24"]  # Should be one of the algebra answers

    def test_default_projection_function(self):
        """Test the default projection function behavior."""
        from its_hub.algorithms.self_consistency import _default_projection_func

        # Test basic stripping
        assert _default_projection_func("  hello world  ") == "hello world"
        assert _default_projection_func("test") == "test"
        assert _default_projection_func("") == ""

        # Test with various whitespace
        assert _default_projection_func("\n\ttest\n\t") == "test"
        assert _default_projection_func("   ") == ""

        # Test that it preserves internal whitespace
        assert _default_projection_func("  hello world  ") == "hello world"
        assert _default_projection_func("\nhello\nworld\n") == "hello\nworld"

    def test_default_projection_function_with_self_consistency(self):
        """Test that default projection function works correctly with SelfConsistency."""
        responses = [
            "  answer: 42  ",  # Leading/trailing whitespace
            "answer: 42",     # No whitespace
            "  answer: 42  ", # Duplicate with whitespace
            "answer: 24"      # Different answer
        ]
        mock_lm = StepMockLanguageModel(responses)

        # Test with default projection function (None should use _default_projection_func)
        sc = SelfConsistency(consistency_space_projection_func=None)
        result = sc.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        # "answer: 42" should be most common after stripping whitespace
        assert result.the_one["content"].strip() == "answer: 42"

        # Verify the counts - "answer: 42" should appear 3 times after projection
        assert result.response_counts["answer: 42"] == 3
        assert result.response_counts["answer: 24"] == 1

    @pytest.mark.asyncio
    async def test_self_consistency_ainfer_flat_projection(self):
        """Test SelfConsistency async ainfer with flat (string) projection function."""
        mock_lm = StepMockLanguageModel(["answer1", "answer2", "answer1", "answer3"])

        def flat_projection(response: str) -> str:
            return response[-1]  # Last character

        sc = SelfConsistency(flat_projection)
        result = await sc.ainfer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        assert isinstance(result.response_counts, Counter)
        # "1" appears twice (from "answer1"), should be selected
        assert result.the_one["content"] in ["answer1", "answer1"]
        assert result.the_one["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_self_consistency_ainfer_hierarchical_projection(self):
        """Test SelfConsistency async ainfer with hierarchical (tuple) projection function."""
        mock_lm = StepMockLanguageModel(["a1", "a2", "b1", "c1"])

        def hierarchical_projection(response: str) -> tuple:
            return (response[0], response[1])  # First char, second char as tuple

        sc = SelfConsistency(hierarchical_projection)
        result = await sc.ainfer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        assert isinstance(result.response_counts, Counter)
        # "a" is most common at level 0, so should select either "a1" or "a2"
        assert result.the_one["content"] in ["a1", "a2"]
        assert result.the_one["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_self_consistency_ainfer_return_response_only(self):
        """Test SelfConsistency async ainfer with return_response_only=True."""
        mock_lm = StepMockLanguageModel(["a1", "a2", "b1", "c1"])

        def hierarchical_projection(response: str) -> tuple:
            return (response[0], response[1])

        sc = SelfConsistency(hierarchical_projection)
        result = await sc.ainfer(mock_lm, "test prompt", budget=4, return_response_only=True)

        assert isinstance(result, dict)
        assert result["content"] in ["a1", "a2"]  # Should be one of the "a" responses
        assert result["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_self_consistency_ainfer_with_chat_messages_class(self):
        """Test SelfConsistency async ainfer with ChatMessages class input."""
        mock_lm = StepMockLanguageModel(["answer1", "answer2", "answer1", "answer3"])

        def flat_projection(response: str) -> str:
            return response[-1]  # Last character

        sc = SelfConsistency(flat_projection)

        # Test with ChatMessages wrapping a string
        chat_messages = ChatMessages("test prompt")
        result = await sc.ainfer(mock_lm, chat_messages, budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        # "1" appears twice (from "answer1"), should be selected
        assert result.the_one["content"] in ["answer1", "answer1"]
        assert result.the_one["role"] == "assistant"

    def test_with_multimodal_content(self):
        """Test SelfConsistency with multi-modal list[dict] content."""
        # Mock LM returns responses with multimodal content
        mock_lm = StepMockLanguageModel([
            [{"type": "text", "text": "Answer: 42"}],
            [{"type": "text", "text": "Answer: 24"}],
            [{"type": "text", "text": "Answer: 42"}],
        ])

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is the answer?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            )
        ]

        sc = SelfConsistency(consistency_space_projection_func=None)
        result = sc.infer(mock_lm, messages, budget=3, return_response_only=True)

        # Should select most common answer (42 appears twice)
        assert isinstance(result, dict)
        # Content should be preserved as list
        assert result["content"] == [{"type": "text", "text": "Answer: 42"}]


class TestDataStructures:
    """Test core data structures used by algorithms."""

    def test_path_deepcopy(self):
        """Test Path deepcopy functionality."""
        steps = ['a', 'b', 'c']
        is_stopped = False
        score = 1.0
        path = Path(steps=deepcopy(steps), is_stopped=is_stopped, score=score)
        path_copy = path.deepcopy()
        path.steps.append('d')

        assert path_copy.steps == steps
        assert path_copy.is_stopped == is_stopped
        assert path_copy.score == score

    def test_particle_deepcopy(self):
        """Test Particle deepcopy functionality."""
        steps = ['a', 'b', 'c']
        is_stopped = False
        partial_log_weights = [0.3, 0.6, 1.0]
        particle = Particle(
            steps=deepcopy(steps),
            is_stopped=is_stopped,
            partial_log_weights=deepcopy(partial_log_weights)
        )
        particle_copy = particle.deepcopy()
        particle.steps.append('d')
        particle.partial_log_weights.append(1.2)

        assert particle_copy.steps == steps
        assert particle_copy.is_stopped == is_stopped
        assert particle_copy.log_weight == 1.0  # Should return last value of partial_log_weights
        assert particle_copy.partial_log_weights == partial_log_weights

    def test_with_multimodal_content(self):
        """Test BestOfN with multi-modal list[dict] content."""
        # Mock LM returns responses with multimodal content
        mock_lm = StepMockLanguageModel([
            [{"type": "text", "text": "Answer is 42"}],
            [{"type": "text", "text": "Answer is 24"}]
        ])
        mock_orm = MockOutcomeRewardModel([0.3, 0.7])

        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "What is the answer?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                ]
            )
        ]

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, messages, budget=2, return_response_only=True)

        # Should extract text from list content
        assert result["content"] == [{"type": "text", "text": "Answer is 24"}]


class TestBestOfN:
    """Test the Best-of-N algorithm."""

    def test_result_structure(self):
        """Test BestOfNResult data structure."""
        responses = [
            {"role": "assistant", "content": "response1"},
            {"role": "assistant", "content": "response2"},
            {"role": "assistant", "content": "response3"}
        ]
        scores = [0.5, 0.8, 0.3]
        selected_index = 1

        result = BestOfNResult(responses=responses, scores=scores, selected_index=selected_index)

        assert result.responses == responses
        assert result.scores == scores
        assert result.selected_index == selected_index
        assert result.the_one["content"] == "response2"

    @pytest.mark.parametrize("responses,scores,expected_index,expected_response", [
        (["response1", "response2", "response3"], [0.5, 0.8, 0.3], 1, "response2"),
        (["response1", "response2", "response3"], [0.8, 0.5, 0.8], 0, "response1"),  # Tie - first wins
        (["response1"], [0.7], 0, "response1"),  # Single response
    ])
    def test_selection_logic(self, responses, scores, expected_index, expected_response):
        """Test Best-of-N selection logic with various score scenarios."""
        mock_lm = StepMockLanguageModel(responses)
        mock_orm = MockOutcomeRewardModel(scores)

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=len(responses), return_response_only=False)

        assert result.selected_index == expected_index
        assert result.the_one["content"] == expected_response

    def test_return_response_only(self):
        """Test return_response_only parameter."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=True)

        assert result["content"] == "response2"

    def test_with_chat_messages_string(self):
        """Test BestOfN with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        bon = BestOfN(mock_orm)
        chat_messages = ChatMessages("test prompt")
        result = bon.infer(mock_lm, chat_messages, budget=3, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.the_one["content"] == "response2"
        assert len(result.responses) == 3

    def test_with_chat_messages_conversation(self):
        """Test BestOfN with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        # Create conversation history
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="2+2=4"),
            ChatMessage(role="user", content="What about 3+3?")
        ]
        chat_messages = ChatMessages(messages)

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, chat_messages, budget=3, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.the_one["content"] == "response2"
        # Verify the reward model received the ChatMessages object
        assert len(result.responses) == 3

    def test_with_list_chat_messages(self):
        """Test BestOfN with list[ChatMessage] input."""
        mock_lm = StepMockLanguageModel(["response1", "response2"])
        mock_orm = MockOutcomeRewardModel([0.3, 0.7])

        messages = [
            ChatMessage(role="user", content="Solve this problem")
        ]

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, messages, budget=2, return_response_only=True)

        assert result["content"] == "response2"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("responses,scores,expected_index,expected_response", [
        (["response1", "response2", "response3"], [0.5, 0.8, 0.3], 1, "response2"),
        (["response1", "response2", "response3"], [0.8, 0.5, 0.8], 0, "response1"),  # Tie - first wins
        (["response1"], [0.7], 0, "response1"),  # Single response
    ])
    async def test_ainfer_selection_logic(self, responses, scores, expected_index, expected_response):
        """Test Best-of-N async ainfer selection logic with various score scenarios."""
        mock_lm = StepMockLanguageModel(responses)
        mock_orm = MockOutcomeRewardModel(scores)

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, "test prompt", budget=len(responses), return_response_only=False)

        assert result.selected_index == expected_index
        assert result.the_one["content"] == expected_response

    @pytest.mark.asyncio
    async def test_ainfer_return_response_only(self):
        """Test async ainfer return_response_only parameter."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, "test prompt", budget=3, return_response_only=True)

        assert result["content"] == "response2"

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_string(self):
        """Test async ainfer BestOfN with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        bon = BestOfN(mock_orm)
        chat_messages = ChatMessages("test prompt")
        result = await bon.ainfer(mock_lm, chat_messages, budget=3, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.the_one["content"] == "response2"
        assert len(result.responses) == 3

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_conversation(self):
        """Test async ainfer BestOfN with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        # Create conversation history
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="2+2=4"),
            ChatMessage(role="user", content="What about 3+3?")
        ]
        chat_messages = ChatMessages(messages)

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, chat_messages, budget=3, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.the_one["content"] == "response2"
        # Verify the reward model received the ChatMessages object
        assert len(result.responses) == 3

    @pytest.mark.asyncio
    async def test_ainfer_with_list_chat_messages(self):
        """Test async ainfer BestOfN with list[ChatMessage] input."""
        mock_lm = StepMockLanguageModel(["response1", "response2"])
        mock_orm = MockOutcomeRewardModel([0.3, 0.7])

        messages = [
            ChatMessage(role="user", content="Solve this problem")
        ]

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, messages, budget=2, return_response_only=True)

        assert result["content"] == "response2"

    def test_deduplication_all_identical(self):
        """Test Best-of-N with all identical responses - should skip scoring."""
        # All responses are identical
        mock_lm = StepMockLanguageModel(["response1", "response1", "response1", "response1"])
        # Mock ORM should not be called since we skip scoring for identical responses
        mock_orm = MockOutcomeRewardModel([])

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        # Should select first response
        assert result.selected_index == 0
        assert result.the_one["content"] == "response1"
        # All identical responses should have score 1
        assert result.scores == [1, 1, 1, 1]
        # Verify ORM was never called
        assert mock_orm.score_call_count == 0

    def test_deduplication_some_duplicates(self):
        """Test Best-of-N with some duplicate responses - should deduplicate before scoring."""
        # 8 responses: 3x "response1", 2x "response2", 2x "response3", 1x "response4"
        mock_lm = StepMockLanguageModel([
            "response1", "response2", "response1", "response3",
            "response2", "response1", "response4", "response3"
        ])
        # Only 4 unique responses should be scored: response1, response2, response3, response4
        # Scores: response2 has highest score (0.9)
        mock_orm = MockOutcomeRewardModel([0.5, 0.9, 0.3, 0.7])

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=8, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        # response2 should be selected (appears at indices 1 and 4)
        assert result.selected_index in [1, 4]
        assert result.the_one["content"] == "response2"
        # Verify all 8 responses have scores
        assert len(result.scores) == 8
        # Verify score mapping: indices with same content have same score
        assert result.scores[0] == result.scores[2] == result.scores[5] == 0.5  # response1
        assert result.scores[1] == result.scores[4] == 0.9  # response2
        assert result.scores[3] == result.scores[7] == 0.3  # response3
        assert result.scores[6] == 0.7  # response4
        # Verify ORM was called only once (batched) with 4 unique responses
        assert mock_orm.score_call_count == 1
        # Verify 4 scores were returned in that single call
        assert mock_orm.call_count == 4

    def test_deduplication_all_unique(self):
        """Test Best-of-N with all unique responses - should score all responses."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])

        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.selected_index == 1
        assert result.the_one["content"] == "response2"
        # All 3 responses are unique, so all should be scored
        assert result.scores == [0.5, 0.8, 0.3]
        # ORM should be called once with all 3 unique responses
        assert mock_orm.score_call_count == 1
        assert mock_orm.call_count == 3

    @pytest.mark.asyncio
    async def test_ainfer_deduplication_all_identical(self):
        """Test async Best-of-N with all identical responses - should skip scoring."""
        mock_lm = StepMockLanguageModel(["response1", "response1", "response1", "response1"])
        mock_orm = MockOutcomeRewardModel([])

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, "test prompt", budget=4, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.selected_index == 0
        assert result.the_one["content"] == "response1"
        assert result.scores == [1, 1, 1, 1]
        assert mock_orm.score_call_count == 0

    @pytest.mark.asyncio
    async def test_ainfer_deduplication_some_duplicates(self):
        """Test async Best-of-N with some duplicate responses - should deduplicate before scoring."""
        mock_lm = StepMockLanguageModel([
            "response1", "response2", "response1", "response3",
            "response2", "response1", "response4", "response3"
        ])
        mock_orm = MockOutcomeRewardModel([0.5, 0.9, 0.3, 0.7])

        bon = BestOfN(mock_orm)
        result = await bon.ainfer(mock_lm, "test prompt", budget=8, return_response_only=False)

        assert isinstance(result, BestOfNResult)
        assert result.selected_index in [1, 4]
        assert result.the_one["content"] == "response2"
        assert len(result.scores) == 8
        assert result.scores[0] == result.scores[2] == result.scores[5] == 0.5
        assert result.scores[1] == result.scores[4] == 0.9
        assert result.scores[3] == result.scores[7] == 0.3
        assert result.scores[6] == 0.7
        assert mock_orm.score_call_count == 1
        assert mock_orm.call_count == 4


class TestBeamSearch:
    """Test the Beam Search algorithm."""

    def test_result_structure(self):
        """Test BeamSearchResult data structure."""
        responses = [
            {"role": "assistant", "content": "response1"},
            {"role": "assistant", "content": "response2"},
            {"role": "assistant", "content": "response3"}
        ]
        scores = [0.5, 0.8, 0.3]
        selected_index = 1
        steps_used = [2, 3, 1]

        result = BeamSearchResult(responses=responses, scores=scores, selected_index=selected_index, steps_used=steps_used)

        assert result.responses == responses
        assert result.scores == scores
        assert result.selected_index == selected_index
        assert result.steps_used == steps_used
        assert result.the_one["content"] == "response2"

    def test_basic_functionality(self):
        """Test basic beam search functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "stepA", "stepB"])
        mock_prm = MockProcessRewardModel([0.7, 0.9])

        sg = StepGeneration(step_token="\n", max_steps=2)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        result = beam_search.infer(mock_lm, "Solve this problem:", budget=2, return_response_only=True)

        assert isinstance(result, dict)

    def test_budget_validation(self):
        """Test budget validation constraints."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        # Test budget not divisible by beam_width
        with pytest.raises(AssertionError, match="budget must be divisible by beam_width"):
            beam_search.infer(mock_lm, "test prompt", budget=3)

    def test_path_selection(self):
        """Test that beam search selects the highest scoring path."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step", "good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1, 0.8, 0.2])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        result = beam_search.infer(mock_lm, "Solve this:", budget=4, return_response_only=False)

        assert isinstance(result, BeamSearchResult)
        assert result.selected_index == result.scores.index(max(result.scores))

    def test_with_chat_messages_string(self):
        """Test BeamSearch with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.9])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        chat_messages = ChatMessages("Solve this problem:")
        result = beam_search.infer(mock_lm, chat_messages, budget=2, return_response_only=False)

        assert isinstance(result, BeamSearchResult)
        assert isinstance(result.the_one, dict)

    def test_with_chat_messages_conversation(self):
        """Test BeamSearch with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.8, 0.6])

        messages = [
            ChatMessage(role="system", content="You are a problem solver"),
            ChatMessage(role="user", content="Solve step by step:")
        ]
        chat_messages = ChatMessages(messages)

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)
        result = beam_search.infer(mock_lm, chat_messages, budget=2, return_response_only=True)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ainfer_basic_functionality(self):
        """Test basic async beam search functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "stepA", "stepB"])
        mock_prm = MockProcessRewardModel([0.7, 0.9])

        sg = StepGeneration(step_token="\n", max_steps=2)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        result = await beam_search.ainfer(mock_lm, "Solve this problem:", budget=2, return_response_only=True)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ainfer_budget_validation(self):
        """Test async budget validation constraints."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        # Test budget not divisible by beam_width
        with pytest.raises(AssertionError, match="budget must be divisible by beam_width"):
            await beam_search.ainfer(mock_lm, "test prompt", budget=3)

    @pytest.mark.asyncio
    async def test_ainfer_path_selection(self):
        """Test that async beam search selects the highest scoring path."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step", "good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1, 0.8, 0.2])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        result = await beam_search.ainfer(mock_lm, "Solve this:", budget=4, return_response_only=False)

        assert isinstance(result, BeamSearchResult)
        assert result.selected_index == result.scores.index(max(result.scores))

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_string(self):
        """Test async BeamSearch with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.9])

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)

        chat_messages = ChatMessages("Solve this problem:")
        result = await beam_search.ainfer(mock_lm, chat_messages, budget=2, return_response_only=False)

        assert isinstance(result, BeamSearchResult)
        assert isinstance(result.the_one, dict)

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_conversation(self):
        """Test async BeamSearch with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.8, 0.6])

        messages = [
            ChatMessage(role="system", content="You are a problem solver"),
            ChatMessage(role="user", content="Solve step by step:")
        ]
        chat_messages = ChatMessages(messages)

        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)
        result = await beam_search.ainfer(mock_lm, chat_messages, budget=2, return_response_only=True)

        assert isinstance(result, dict)


class TestParticleGibbs:
    """Test the Particle Gibbs algorithm."""

    def test_result_structure(self):
        """Test ParticleGibbsResult data structure."""
        responses_lst = [
            [{"role": "assistant", "content": "response1"}, {"role": "assistant", "content": "response2"}],
            [{"role": "assistant", "content": "response3"}, {"role": "assistant", "content": "response4"}]
        ]
        log_weights_lst = [[0.1, 0.2], [0.3, 0.4]]
        ref_indices_lst = [[0], [1]]
        selected_index = 1
        steps_used_lst = [[2, 3], [1, 4]]

        result = ParticleGibbsResult(
            responses_lst=responses_lst,
            log_weights_lst=log_weights_lst,
            ref_indices_lst=ref_indices_lst,
            selected_index=selected_index,
            steps_used_lst=steps_used_lst
        )

        assert result.responses_lst == responses_lst
        assert result.log_weights_lst == log_weights_lst
        assert result.ref_indices_lst == ref_indices_lst
        assert result.selected_index == selected_index
        assert result.steps_used_lst == steps_used_lst
        assert result.the_one["content"] == "response4"

    def test_basic_functionality(self):
        """Test basic particle Gibbs functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)

        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)

        assert isinstance(result, dict)

    def test_budget_validation(self):
        """Test budget validation for particle Gibbs."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=3)

        with pytest.raises(AssertionError, match="budget must be divisible by num_iterations"):
            particle_gibbs.infer(mock_lm, "test prompt", budget=4)

    @pytest.mark.parametrize("selection_method,expected_type", [
        (SelectionMethod.ARGMAX, dict),
        (SelectionMethod.SAMPLE, dict),
        ("argmax", dict),  # Test string conversion
    ])
    def test_selection_methods(self, selection_method, expected_type):
        """Test different selection methods."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=selection_method)
        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)

        assert isinstance(result, expected_type)

    def test_multiple_iterations(self):
        """Test particle Gibbs with multiple iterations."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "step3", "step4"])
        mock_prm = MockProcessRewardModel([0.7, 0.6, 0.8, 0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(
            sg, mock_prm,
            num_iterations=2,
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )

        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=4, return_response_only=False)

        assert isinstance(result, ParticleGibbsResult)
        assert len(result.responses_lst) == 2  # num_iterations = 2
        assert len(result.log_weights_lst) == 2
        assert len(result.ref_indices_lst) == 2

    def test_ancestor_sampling_not_implemented(self):
        """Test that ancestor sampling raises NotImplementedError."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(
            sg, mock_prm,
            num_iterations=1,
            does_ancestor_sampling=True
        )

        with pytest.raises(NotImplementedError, match="Ancestor sampling is not implemented"):
            particle_gibbs.infer(mock_lm, "test prompt", budget=1)

    def test_with_chat_messages_string(self):
        """Test ParticleGibbs with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)

        chat_messages = ChatMessages("Solve this:")
        result = particle_gibbs.infer(mock_lm, chat_messages, budget=2, return_response_only=False)

        assert isinstance(result, ParticleGibbsResult)
        assert isinstance(result.the_one, dict)

    def test_with_chat_messages_conversation(self):
        """Test ParticleGibbs with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.8, 0.5])

        messages = [
            ChatMessage(role="system", content="Solve step by step"),
            ChatMessage(role="user", content="Problem:")
        ]
        chat_messages = ChatMessages(messages)

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
        result = particle_gibbs.infer(mock_lm, chat_messages, budget=2, return_response_only=True)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ainfer_basic_functionality(self):
        """Test basic async particle Gibbs functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)

        result = await particle_gibbs.ainfer(mock_lm, "Solve this:", budget=2, return_response_only=True)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ainfer_budget_validation(self):
        """Test async budget validation for particle Gibbs."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=3)

        with pytest.raises(AssertionError, match="budget must be divisible by num_iterations"):
            await particle_gibbs.ainfer(mock_lm, "test prompt", budget=4)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("selection_method,expected_type", [
        (SelectionMethod.ARGMAX, dict),
        (SelectionMethod.SAMPLE, dict),
        ("argmax", dict),  # Test string conversion
    ])
    async def test_ainfer_selection_methods(self, selection_method, expected_type):
        """Test async different selection methods."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=selection_method)
        result = await particle_gibbs.ainfer(mock_lm, "Solve this:", budget=2, return_response_only=True)

        assert isinstance(result, expected_type)

    @pytest.mark.asyncio
    async def test_ainfer_multiple_iterations(self):
        """Test async particle Gibbs with multiple iterations."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "step3", "step4"])
        mock_prm = MockProcessRewardModel([0.7, 0.6, 0.8, 0.5])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(
            sg, mock_prm,
            num_iterations=2,
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )

        result = await particle_gibbs.ainfer(mock_lm, "Solve this:", budget=4, return_response_only=False)

        assert isinstance(result, ParticleGibbsResult)
        assert len(result.responses_lst) == 2  # num_iterations = 2
        assert len(result.log_weights_lst) == 2
        assert len(result.ref_indices_lst) == 2

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_string(self):
        """Test async ParticleGibbs with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)

        chat_messages = ChatMessages("Solve this:")
        result = await particle_gibbs.ainfer(mock_lm, chat_messages, budget=2, return_response_only=False)

        assert isinstance(result, ParticleGibbsResult)
        assert isinstance(result.the_one, dict)

    @pytest.mark.asyncio
    async def test_ainfer_with_chat_messages_conversation(self):
        """Test async ParticleGibbs with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.8, 0.5])

        messages = [
            ChatMessage(role="system", content="Solve step by step"),
            ChatMessage(role="user", content="Problem:")
        ]
        chat_messages = ChatMessages(messages)

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
        result = await particle_gibbs.ainfer(mock_lm, chat_messages, budget=2, return_response_only=True)

        assert isinstance(result, dict)


class TestParticleFiltering:
    """Test the Particle Filtering algorithm (special case of Particle Gibbs)."""

    def test_is_single_iteration_particle_gibbs(self):
        """Test that ParticleFiltering is equivalent to ParticleGibbs with 1 iteration."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)

        particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)
        result = particle_filtering.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)

        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == 2  # budget = 2 (flattened from single iteration)

        # Test that .the_one property works correctly with flattened structure
        assert result.the_one == result.responses[result.selected_index]
        assert isinstance(result.the_one, dict)

    def test_particle_filtering_return_response_only(self):
        """Test ParticleFiltering with return_response_only=True."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)

        particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)
        result = particle_filtering.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)

        # Should return just the dict response
        assert isinstance(result, dict)
        assert result  # Should not be empty

    def test_with_chat_messages_string(self):
        """Test ParticleFiltering with ChatMessages wrapping a string."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)

        chat_messages = ChatMessages("Solve this:")
        result = particle_filtering.infer(mock_lm, chat_messages, budget=2, return_response_only=False)

        assert isinstance(result, ParticleFilteringResult)
        assert isinstance(result.the_one, dict)

    def test_with_chat_messages_conversation(self):
        """Test ParticleFiltering with ChatMessages containing conversation history."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.8, 0.5])

        messages = [
            ChatMessage(role="system", content="You are a step-by-step solver"),
            ChatMessage(role="user", content="Please solve:")
        ]
        chat_messages = ChatMessages(messages)

        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)
        result = particle_filtering.infer(mock_lm, chat_messages, budget=2, return_response_only=True)

        assert isinstance(result, dict)
