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

from its_hub import SelfConsistency, BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.reward_models import LLMJudge


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

    @pytest.fixture
    def judge_lm(self, openai_lm):
        """Create a judge LM instance."""
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAICompatibleLanguageModel(
            endpoint="https://api.openai.com/v1",
            api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.3,
        )

    def test_best_of_n_with_llm_judge_sync(self, openai_lm, judge_lm):
        """Test BestOfN with LLM judge (sync)."""
        judge = LLMJudge(lm=judge_lm, fallback_score=5.0)
        algorithm = BestOfN(judge)

        prompt = "Write a one-sentence story about a cat."
        result = algorithm.infer(openai_lm, prompt, budget=2, return_response_only=False)

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert hasattr(result, 'scores')
        assert len(result.responses) == 2
        assert len(result.scores) == 2

        # Scores should be floats from LLMJudge
        assert all(isinstance(s, float) for s in result.scores)
        assert all(0 <= s <= 10 for s in result.scores)

        print(f"\n✓ Best-of-N result: {result.the_one['content'][:100]}...")
        print(f"✓ Scores: {result.scores}")


    @pytest.mark.asyncio
    async def test_best_of_n_async(self, openai_lm, judge_lm):
        """Test async BestOfN with LLM judge."""
        judge = LLMJudge(lm=judge_lm, fallback_score=5.0)
        algorithm = BestOfN(judge)

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
        from its_hub import AbstractLanguageModel, SelfConsistency, BestOfN
        from its_hub.base import AbstractOutcomeRewardModel

        assert AbstractLanguageModel is not None
        assert SelfConsistency is not None
        assert BestOfN is not None
        assert AbstractOutcomeRewardModel is not None

        print("\n✓ All core abstractions importable")

    @pytest.mark.asyncio
    async def test_openai_lm_direct_call(self, openai_lm):
        """Test direct OpenAI LM call."""
        from its_hub.types import ChatMessage

        messages = [ChatMessage(role="user", content="Say 'hello' in one word.")]
        response = await openai_lm.agenerate(messages)

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


class TestLLMJudgeE2E:
    """End-to-end tests for LLMJudge with real OpenAI API."""

    @pytest.fixture
    def judge_lm(self, openai_lm):
        """Create a separate LM instance for the judge."""
        # Use cheaper/faster model for judging
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAICompatibleLanguageModel(
            endpoint="https://api.openai.com/v1",
            api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.3,
        )

    @pytest.mark.asyncio
    async def test_llm_judge_single_conversation_with_tool_calls(self, judge_lm):
        """Test LLMJudge scoring a single conversation with tool calls."""
        judge = LLMJudge(lm=judge_lm, fallback_score=5.0)

        # Single conversation with tool call
        conversation = [
            {"role": "user", "content": "What is 2+2? Use the calculator."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}'
                        }
                    }
                ]
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "4"},
            {"role": "assistant", "content": "The answer is 4."}
        ]

        score = await judge.ascore(conversation)

        # Verify score is returned
        assert isinstance(score, float)
        assert 0 <= score <= 10  # Default prompt uses 0-10 scale

        print(f"\n✓ LLMJudge single conversation with tool calls score: {score}")

    @pytest.mark.asyncio
    async def test_llm_judge_batch_conversations_with_tool_calls(self, judge_lm):
        """Test LLMJudge scoring multiple conversations with tool calls in batch."""
        judge = LLMJudge(lm=judge_lm, fallback_score=5.0)

        # Multiple conversations with tool calls
        conversations = [
            [
                {"role": "user", "content": "Calculate 5*3"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "5*3"}'
                            }
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_456", "content": "15"},
                {"role": "assistant", "content": "5 times 3 equals 15."}
            ],
            [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_789",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}'
                            }
                        }
                    ]
                },
                {"role": "tool", "tool_call_id": "call_789", "content": "Sunny, 22°C"},
                {"role": "assistant", "content": "The weather in Paris is sunny with a temperature of 22°C."}
            ],
        ]

        scores = await judge.ascore(conversations)

        # Verify batch scoring
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert all(0 <= s <= 10 for s in scores)

        print(f"\n✓ LLMJudge batch with tool calls scores: {scores}")

    @pytest.mark.asyncio
    async def test_best_of_n_with_llm_judge_and_tool_calls(self, openai_lm, judge_lm):
        """Test BestOfN with LLMJudge scoring tool call responses."""
        # Custom prompt to evaluate tool call quality
        custom_prompt = """Evaluate the quality of tool usage in this conversation.
Consider: appropriate tool selection, correct arguments, and helpful final response.

Conversation:
{conversation}

Return a JSON object with a score from 0-10.
Format: {{"score": <number>}}"""

        judge = LLMJudge(lm=judge_lm, judge_prompt=custom_prompt, fallback_score=5.0)
        algorithm = BestOfN(judge)

        # Define calculator tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

        prompt = "What is 15 * 8? Use the calculator tool."
        result = await algorithm.ainfer(
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
        assert hasattr(result, 'scores')
        assert len(result.responses) == 2
        assert len(result.scores) == 2

        # Scores should be floats from LLMJudge
        assert all(isinstance(s, float) for s in result.scores)
        assert all(0 <= s <= 10 for s in result.scores)

        # The selected response should have the highest score
        assert result.scores[result.selected_index] == max(result.scores)

        # At least one response should have tool calls
        has_tool_calls = any('tool_calls' in r and r['tool_calls'] for r in result.responses)
        if has_tool_calls:
            print(f"\n✓ BestOfN with LLMJudge (tool calls):")
            print(f"  - Responses contain tool calls: {has_tool_calls}")
            print(f"  - Scores: {result.scores}")
            print(f"  - Selected response score: {result.scores[result.selected_index]}")

    @pytest.mark.asyncio
    async def test_best_of_n_with_llm_judge_text_only(self, openai_lm, judge_lm):
        """Test BestOfN with LLMJudge on text-only conversations."""
        judge = LLMJudge(lm=judge_lm, fallback_score=5.0)
        algorithm = BestOfN(judge)

        prompt = "What is 10-3? Answer with just the number."
        result = await algorithm.ainfer(openai_lm, prompt, budget=2, return_response_only=False)

        # Verify result structure
        assert result is not None
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert hasattr(result, 'scores')
        assert len(result.responses) == 2
        assert len(result.scores) == 2

        # Scores should be floats from LLMJudge
        assert all(isinstance(s, float) for s in result.scores)
        assert all(0 <= s <= 10 for s in result.scores)

        # The selected response should have the highest score
        assert result.scores[result.selected_index] == max(result.scores)

        print(f"\n✓ BestOfN with LLMJudge (text only):")
        print(f"  - Responses: {[r['content'] for r in result.responses]}")
        print(f"  - Scores: {result.scores}")
        print(f"  - Selected: {result.the_one['content']} (score: {result.scores[result.selected_index]})")

    @pytest.mark.asyncio
    async def test_llm_judge_custom_prompt(self, judge_lm):
        """Test LLMJudge with custom judge prompt."""
        # Custom prompt that scores helpfulness
        custom_prompt = """Rate the helpfulness of this conversation on a scale of 0-10.
Return only a JSON object with your score.

Conversation:
{conversation}

Format: {{"score": <number>}}"""

        judge = LLMJudge(lm=judge_lm, judge_prompt=custom_prompt, fallback_score=5.0)

        conversation = [
            {"role": "user", "content": "How do I bake a cake?"},
            {"role": "assistant", "content": "Here's a simple recipe: Mix flour, eggs, sugar, and butter. Bake at 350°F for 30 minutes."},
        ]

        score = await judge.ascore(conversation)

        assert isinstance(score, float)
        assert 0 <= score <= 10

        print(f"\n✓ LLMJudge with custom prompt score: {score}")
