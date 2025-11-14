"""
End-to-end tests with real OpenAI API.

These tests require:
1. A .env file with OPENAI_API_KEY
2. pip install its_hub[lm,dev]

To run: pytest tests/e2e/test_real_openai.py -v -s

WARNING: These tests make real API calls and will incur costs.
"""

import os
from pathlib import Path

import pytest

# Load .env from project root
try:
    from dotenv import load_dotenv
    # Project root is 3 levels up from tests/e2e/test_real_openai.py
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from its_hub import SelfConsistency, BestOfN, DummyRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel


# Note: .env is loaded via tests/e2e/conftest.py
# Tests will fail if OPENAI_API_KEY is not set


@pytest.fixture
def openai_lm():
    """Create a real OpenAI language model instance."""
    # Ensure .env is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")

    return OpenAICompatibleLanguageModel(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model,
        temperature=0.7,
    )


class TestSelfConsistencyE2E:
    """End-to-end tests for SelfConsistency with real OpenAI API."""

    def test_self_consistency_basic(self, openai_lm):
        """Test basic SelfConsistency with a simple question."""
        algorithm = SelfConsistency()

        prompt = "What is 2+2? Answer with just the number."
        result = algorithm.infer(openai_lm, prompt, budget=2, return_response_only=False)

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert len(result.responses) == 2

        # Verify the_one is a dict with content
        assert isinstance(result.the_one, dict)
        assert 'content' in result.the_one
        assert '4' in result.the_one['content']

        print(f"\n✓ Self-Consistency result: {result.the_one['content']}")
        print(f"✓ All responses: {[r['content'] for r in result.responses]}")

    def test_self_consistency_with_tool_calls(self, openai_lm):
        """Test SelfConsistency with real tool calls from OpenAI."""
        # Define a simple calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        # Test with tool_vote="tool_name" - vote on which tool is called
        algorithm = SelfConsistency(tool_vote="tool_name")

        prompt = "What is 2+2? Use the calculate tool."
        result = algorithm.infer(
            openai_lm,
            prompt,
            budget=2,
            return_response_only=False,
            tools=tools,
            tool_choice="auto"
        )

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert len(result.responses) == 2

        # At least one response should have tool_calls
        has_tool_calls = any('tool_calls' in r and r['tool_calls'] for r in result.responses)
        if has_tool_calls:
            print(f"\n✓ Tool calls present in responses")
            print(f"✓ Tool vote result: {result.the_one}")

    @pytest.mark.asyncio
    async def test_self_consistency_async(self, openai_lm):
        """Test async SelfConsistency."""
        algorithm = SelfConsistency()

        prompt = "What is 5+3? Answer with just the number."
        result = await algorithm.ainfer(openai_lm, prompt, budget=2)

        assert result is not None
        assert isinstance(result, dict)
        assert '8' in result['content']

        print(f"\n✓ Async Self-Consistency: {result['content']}")


class TestBestOfNE2E:
    """End-to-end tests for BestOfN with real OpenAI API."""

    def test_best_of_n_with_dummy_reward(self, openai_lm):
        """Test BestOfN with dummy reward model."""
        reward_model = DummyRewardModel(fixed_score=0.5)
        algorithm = BestOfN(reward_model)

        prompt = "Write a one-sentence story about a cat."
        result = algorithm.infer(openai_lm, prompt, budget=2, return_response_only=False)

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert hasattr(result, 'scores')
        assert len(result.responses) == 2
        assert len(result.scores) == 2

        # All scores should be 0.5 with dummy model
        assert all(score == 0.5 for score in result.scores)

        print(f"\n✓ Best-of-N result: {result.the_one['content'][:100]}...")
        print(f"✓ Scores: {result.scores}")


    @pytest.mark.asyncio
    async def test_best_of_n_async(self, openai_lm):
        """Test async BestOfN."""
        reward_model = DummyRewardModel(fixed_score=0.9)
        algorithm = BestOfN(reward_model)

        prompt = "What is 10-3? Answer with just the number."
        result = await algorithm.ainfer(openai_lm, prompt, budget=2)

        assert result is not None
        assert isinstance(result, dict)
        assert '7' in result['content']

        print(f"\n✓ Async Best-of-N: {result['content']}")


class TestCoreInterfaceE2E:
    """Test that core abstractions work with real LM."""

    def test_minimal_imports(self):
        """Verify minimal imports work without [lm] extra."""
        # This should work even without [lm] installed
        from its_hub import AbstractLanguageModel, SelfConsistency, BestOfN, DummyRewardModel
        from its_hub.base import AbstractOutcomeRewardModel

        assert AbstractLanguageModel is not None
        assert SelfConsistency is not None
        assert BestOfN is not None
        assert DummyRewardModel is not None
        assert AbstractOutcomeRewardModel is not None

        print("\n✓ All core abstractions importable")

    def test_openai_lm_direct_call(self, openai_lm):
        """Test direct OpenAI LM call."""
        from its_hub.types import ChatMessage

        messages = [ChatMessage(role="user", content="Say 'hello' in one word.")]
        response = openai_lm.generate(messages)

        assert response is not None
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'hello' in response['content'].lower()

        print(f"\n✓ Direct LM call: {response['content']}")

    @pytest.mark.asyncio
    async def test_openai_lm_async_call(self, openai_lm):
        """Test async OpenAI LM call."""
        from its_hub.types import ChatMessage

        messages = [ChatMessage(role="user", content="Say 'hi' in one word.")]
        response = await openai_lm.agenerate(messages)

        assert response is not None
        assert isinstance(response, dict)
        assert 'content' in response
        assert 'hi' in response['content'].lower()

        print(f"\n✓ Async LM call: {response['content']}")
