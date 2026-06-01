"""Tests for include_raw_choices: no circular references, JSON-serializable,
and correct behaviour when flowing through scaling algorithms."""

import json

import pytest

from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.algorithms.bon import BestOfN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_choice_message(content: str, index: int = 0) -> dict:
    """Build a message dict that looks like what the LM returns with
    include_raw_choices=True (after the fix)."""
    return {
        "role": "assistant",
        "content": content,
        "_raw_choice": {
            "index": index,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        },
    }


class RawChoiceMockLM:
    """Mock LM whose responses already carry _raw_choice metadata."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    async def agenerate_single(self, messages, **kwargs):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return _make_raw_choice_message(self.responses[idx])


class RawChoiceMockORM:
    """Mock outcome reward model that returns predetermined scores."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.call_count = 0

    def score(self, messages, **kwargs):
        if isinstance(messages[0], list):
            out = self.scores[self.call_count : self.call_count + len(messages)]
            self.call_count += len(messages)
            return out
        s = self.scores[self.call_count % len(self.scores)]
        self.call_count += 1
        return s

    async def ascore(self, messages, orchestrator=None, **kwargs):
        return self.score(messages)


# ---------------------------------------------------------------------------
# Unit tests: serialization & identity
# ---------------------------------------------------------------------------

@pytest.fixture
def lm_with_raw_choices(vllm_server):
    return OpenAICompatibleLanguageModel(
        endpoint=vllm_server + "/v1",
        api_key="test-key",
        model_name="test-model",
        include_raw_choices=True,
    )


@pytest.fixture
def lm_without_raw_choices(vllm_server):
    return OpenAICompatibleLanguageModel(
        endpoint=vllm_server + "/v1",
        api_key="test-key",
        model_name="test-model",
        include_raw_choices=False,
    )


@pytest.mark.asyncio
async def test_raw_choice_is_json_serializable(lm_with_raw_choices):
    """Regression: _raw_choice must not create a circular reference."""
    messages = [{"role": "user", "content": "hello"}]
    result = await lm_with_raw_choices.agenerate_single(messages)

    assert "_raw_choice" in result
    serialized = json.dumps(result)
    roundtripped = json.loads(serialized)
    assert roundtripped["_raw_choice"]["finish_reason"] == "stop"
    assert roundtripped["_raw_choice"]["message"]["role"] == "assistant"

    await lm_with_raw_choices.close()


@pytest.mark.asyncio
async def test_raw_choice_not_same_object_as_message(lm_with_raw_choices):
    """The message stored inside _raw_choice must be a separate dict."""
    messages = [{"role": "user", "content": "hello"}]
    result = await lm_with_raw_choices.agenerate_single(messages)

    assert result is not result["_raw_choice"]["message"]

    await lm_with_raw_choices.close()


@pytest.mark.asyncio
async def test_no_raw_choice_when_disabled(lm_without_raw_choices):
    """When include_raw_choices=False, no _raw_choice key should be present."""
    messages = [{"role": "user", "content": "hello"}]
    result = await lm_without_raw_choices.agenerate_single(messages)

    assert "_raw_choice" not in result

    await lm_without_raw_choices.close()


# ---------------------------------------------------------------------------
# Algorithm integration: SelfConsistency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_consistency_with_raw_choices():
    """SelfConsistency should work when responses contain _raw_choice."""
    lm = RawChoiceMockLM(["The answer is 42.", "The answer is 42.", "The answer is 7."])
    sc = SelfConsistency()

    result = await sc.ainfer(lm, "What is 6*7?", budget=3, return_response_only=False)

    assert result.the_one["content"] == "The answer is 42."
    assert "_raw_choice" in result.the_one
    # full result must be serializable
    json.dumps(result.the_one)


@pytest.mark.asyncio
async def test_self_consistency_raw_choices_return_response_only():
    """return_response_only=True should still work with _raw_choice present."""
    lm = RawChoiceMockLM(["answer A", "answer A", "answer B"])
    sc = SelfConsistency()

    result = await sc.ainfer(lm, "test", budget=3, return_response_only=True)

    assert isinstance(result, dict)
    assert result["content"] == "answer A"
    json.dumps(result)


# ---------------------------------------------------------------------------
# Algorithm integration: BestOfN
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_best_of_n_with_raw_choices():
    """BestOfN should work when responses contain _raw_choice."""
    lm = RawChoiceMockLM(["resp A", "resp B", "resp C"])
    orm = RawChoiceMockORM([0.3, 0.9, 0.1])
    bon = BestOfN(orm=orm)

    result = await bon.ainfer(lm, "test", budget=3, return_response_only=False)

    assert result.the_one["content"] == "resp B"
    assert "_raw_choice" in result.the_one
    json.dumps(result.the_one)


@pytest.mark.asyncio
async def test_best_of_n_dedup_with_raw_choices():
    """BestOfN deduplication must not break on _raw_choice metadata.

    Two "same" responses get deduped, so only 2 unique candidates are scored.
    The ORM returns [0.5, 0.8] — "different" wins with score 0.8."""
    lm = RawChoiceMockLM(["same", "same", "different"])
    orm = RawChoiceMockORM([0.5, 0.8])
    bon = BestOfN(orm=orm)

    result = await bon.ainfer(lm, "test", budget=3, return_response_only=False)

    assert result.the_one["content"] == "different"
    json.dumps(result.the_one)


# ---------------------------------------------------------------------------
# End-to-end with real mock HTTP server
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_consistency_e2e_with_raw_choices(lm_with_raw_choices):
    """Full stack: real HTTP mock -> LM with include_raw_choices -> algorithm."""
    sc = SelfConsistency()
    result = await sc.ainfer(
        lm_with_raw_choices,
        "What is 2+2?",
        budget=3,
        return_response_only=False,
    )

    assert "_raw_choice" in result.the_one
    serialized = json.dumps(result.the_one)
    roundtripped = json.loads(serialized)
    assert roundtripped["_raw_choice"]["finish_reason"] == "stop"

    await lm_with_raw_choices.close()
