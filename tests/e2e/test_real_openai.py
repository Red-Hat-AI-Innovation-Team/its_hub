"""E2E tests with real OpenAI API calls.

These tests make actual API calls to OpenAI and are skipped if OPENAI_API_KEY is not set.
Use budget=2 to minimize API costs while still validating functionality.
"""

import pytest

from its_hub import BestOfN, SelfConsistency
from its_hub.reward_models import LLMJudge


class TestSelfConsistencyE2E:
    """E2E tests for Self-Consistency with real API."""

    def test_self_consistency_basic(self, openai_lm):
        """Test basic self-consistency with real API."""
        algorithm = SelfConsistency()
        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="What is 2+2? Answer with just the number.",
            budget=2,  # Minimal budget to reduce costs
            return_response_only=True,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "role" in result
        assert result["role"] == "assistant"
        assert "content" in result
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_self_consistency_with_full_result(self, openai_lm):
        """Test self-consistency returning full result object."""
        algorithm = SelfConsistency()
        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="What is the capital of France?",
            budget=2,
            return_response_only=False,
        )

        # Verify result structure
        assert hasattr(result, "the_one")
        assert hasattr(result, "responses")
        assert isinstance(result.the_one, dict)
        assert len(result.responses) == 2

    def test_self_consistency_with_tool_calls(self, openai_lm):
        """Test self-consistency with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        algorithm = SelfConsistency(tool_vote="tool_name")
        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="Add 2 and 3 using the add_numbers function.",
            budget=2,
            return_response_only=True,
            tools=tools,
            tool_choice="required",
        )

        # Verify tool calls present
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) > 0

    @pytest.mark.asyncio
    async def test_self_consistency_async(self, openai_lm):
        """Test async self-consistency."""
        algorithm = SelfConsistency()
        result = await algorithm.ainfer(
            lm=openai_lm,
            prompt_or_messages="What is the capital of France?",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)
        assert "content" in result


class TestBestOfNE2E:
    """E2E tests for Best-of-N with real API."""

    def test_best_of_n_with_llm_judge_sync(self, openai_lm):
        """Test Best-of-N with LLM judge (sync wrapper)."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)
        algorithm = BestOfN(reward_model=judge)

        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="Write a haiku about programming.",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)
        assert "content" in result

    def test_best_of_n_with_full_result(self, openai_lm):
        """Test Best-of-N returning full result object."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)
        algorithm = BestOfN(reward_model=judge)

        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="What is 5+3?",
            budget=2,
            return_response_only=False,
        )

        # Verify result structure
        assert hasattr(result, "the_one")
        assert hasattr(result, "candidates")
        assert hasattr(result, "scores")
        assert isinstance(result.the_one, dict)
        assert len(result.candidates) == 2
        assert len(result.scores) == 2

    @pytest.mark.asyncio
    async def test_best_of_n_async(self, openai_lm):
        """Test async Best-of-N."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)
        algorithm = BestOfN(reward_model=judge)

        result = await algorithm.ainfer(
            lm=openai_lm,
            prompt_or_messages="What is 5+3?",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)


class TestLLMJudgeE2E:
    """E2E tests for LLMJudge with real API."""

    @pytest.mark.asyncio
    async def test_llm_judge_single_conversation(self, openai_lm):
        """Test LLMJudge scoring a single conversation."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)

        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        score = await judge.ascore(conversation)

        assert isinstance(score, float)
        assert 0 <= score <= 10

    @pytest.mark.asyncio
    async def test_llm_judge_batch_conversations(self, openai_lm):
        """Test LLMJudge batch scoring."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)

        conversations = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "5"},  # Wrong answer
            ],
        ]

        scores = await judge.ascore(conversations)

        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    @pytest.mark.asyncio
    async def test_llm_judge_custom_prompt(self, openai_lm):
        """Test LLMJudge with custom prompt."""
        custom_prompt = """Rate the technical accuracy of this conversation on a scale of 0-10.

Conversation:
{conversation}

Return JSON: {{"score": <number>}}"""

        judge = LLMJudge(lm=openai_lm, judge_prompt=custom_prompt, fallback_score=5.0)

        conversation = [
            {"role": "user", "content": "Explain quicksort."},
            {
                "role": "assistant",
                "content": "Quicksort is a divide-and-conquer sorting algorithm.",
            },
        ]

        score = await judge.ascore(conversation)
        assert isinstance(score, float)


class TestCoreInterfaceE2E:
    """Test core interface patterns."""

    def test_minimal_imports(self):
        """Test that core imports work without [lm] extra."""
        from its_hub import (
            AbstractLanguageModel,
            AbstractScalingAlgorithm,
            BestOfN,
            SelfConsistency,
        )

        assert AbstractLanguageModel is not None
        assert AbstractScalingAlgorithm is not None
        assert SelfConsistency is not None
        assert BestOfN is not None

    def test_lm_extra_imports(self):
        """Test that [lm] extra imports work."""
        from its_hub import OpenAICompatibleLanguageModel, StepGeneration
        from its_hub.reward_models import LLMJudge

        assert OpenAICompatibleLanguageModel is not None
        assert StepGeneration is not None
        assert LLMJudge is not None
