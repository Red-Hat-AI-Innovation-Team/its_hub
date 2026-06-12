"""Tests for include_raw_choices: no circular references and JSON-serializable
responses from the language model."""

import json

import pytest

from its_hub import OpenAICompatibleLanguageModel

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
