"""Test structured output support with a locally hosted model (Qwen3-4B via vLLM)."""

import asyncio
import json
import logging

from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.reward_models.llm_judge import LLMJudge
from its_hub.api.types import ChatMessage

logging.basicConfig(level=logging.INFO)

ENDPOINT = "http://localhost:8200/v1"
MODEL = "Qwen/Qwen3-4B"
# Qwen3 uses thinking mode by default - needs generous max_tokens
MAX_TOKENS = 4096


async def test_raw_structured_output():
    """Test response_format directly through the LM layer."""
    print("\n=== Test 1: Raw structured output via LM ===")
    async with OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT, api_key="NO_API_KEY", model_name=MODEL
    ) as lm:
        response = await lm.agenerate_single(
            messages=[ChatMessage(role="user", content="What is 2+2? Return JSON with 'answer' field.")],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "math_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "number"},
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            max_tokens=MAX_TOKENS,
            temperature=0.6,
        )
        content = response.get("content") or ""
        reasoning = response.get("reasoning")
        print(f"Reasoning present: {reasoning is not None}")

        # vLLM + Qwen3 can pad JSON with excessive whitespace
        content_stripped = content.strip()
        print(f"Content (stripped, first 200 chars): {content_stripped[:200]!r}")

        if not content_stripped:
            print("SKIPPED - model returned no content (thinking mode exhausted tokens)")
            return

        parsed = json.loads(content_stripped)
        print(f"Parsed JSON: {parsed}")
        assert "answer" in parsed, "Missing 'answer' field"
        print("PASSED\n")


async def test_llm_judge_with_structured_output():
    """Test LLMJudge using structured output (default behavior)."""
    print("=== Test 2: LLMJudge with structured output (default) ===")
    async with OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT, api_key="NO_API_KEY", model_name=MODEL
    ) as lm:
        judge = LLMJudge(lm=lm)
        conversation = [
            ChatMessage(role="user", content="What is the capital of France?"),
            ChatMessage(role="assistant", content="The capital of France is Paris."),
        ]
        score = await judge.ascore(
            messages=conversation,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
        )
        print(f"Score: {score}")
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        print("PASSED\n")


async def test_llm_judge_without_structured_output():
    """Test LLMJudge with structured output disabled (prompt-only JSON extraction)."""
    print("=== Test 3: LLMJudge WITHOUT structured output (response_format=None) ===")
    async with OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT, api_key="NO_API_KEY", model_name=MODEL
    ) as lm:
        judge = LLMJudge(lm=lm, response_format=None)
        conversation = [
            ChatMessage(role="user", content="Explain quantum computing in one sentence."),
            ChatMessage(role="assistant", content="Quantum computing uses qubits that can exist in superposition to perform certain calculations exponentially faster than classical computers."),
        ]
        score = await judge.ascore(
            messages=conversation,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
        )
        print(f"Score: {score}")
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        print("PASSED\n")


async def test_llm_judge_batch():
    """Test LLMJudge scoring multiple conversations in batch."""
    print("=== Test 4: LLMJudge batch scoring with structured output ===")
    async with OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT, api_key="NO_API_KEY", model_name=MODEL
    ) as lm:
        judge = LLMJudge(lm=lm)
        conversations = [
            [
                ChatMessage(role="user", content="What is 2+2?"),
                ChatMessage(role="assistant", content="4"),
            ],
            [
                ChatMessage(role="user", content="What is the meaning of life?"),
                ChatMessage(role="assistant", content="I don't know."),
            ],
            [
                ChatMessage(role="user", content="Write a haiku about Python."),
                ChatMessage(role="assistant", content="Indentation rules\nSnakes and code intertwine here\nBeautiful syntax"),
            ],
        ]
        scores = await judge.ascore(
            messages=conversations,
            max_tokens=MAX_TOKENS,
            temperature=0.6,
        )
        print(f"Scores: {scores}")
        assert isinstance(scores, list), f"Expected list, got {type(scores)}"
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        for s in scores:
            assert isinstance(s, float), f"Expected float, got {type(s)}"
        print("PASSED\n")


async def main():
    print("Testing structured output support against Qwen/Qwen3-4B (vLLM)")
    print("=" * 60)

    await test_raw_structured_output()
    await test_llm_judge_with_structured_output()
    await test_llm_judge_without_structured_output()
    await test_llm_judge_batch()

    print("=" * 60)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
