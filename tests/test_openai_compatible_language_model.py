"""Tests for max_tokens → max_completion_tokens migration."""

import pytest

from its_hub.api.types import ChatMessage
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel


class TestConstructor:
    def test_new_name(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
            max_completion_tokens=500,
        )
        assert lm.max_completion_tokens == 500
        assert not hasattr(lm, "max_tokens")

    def test_neither(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
        )
        assert lm.max_completion_tokens is None
        assert not hasattr(lm, "max_tokens")

    def test_old_name_warns(self):
        with pytest.warns(DeprecationWarning, match="max_tokens.*deprecated"):
            lm = OpenAICompatibleLanguageModel(
                endpoint="http://localhost:8000/v1",
                api_key="test",
                model_name="test",
                max_tokens=500,
            )
        assert lm.max_completion_tokens == 500
        assert not hasattr(lm, "max_tokens")

    def test_both_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            OpenAICompatibleLanguageModel(
                endpoint="http://localhost:8000/v1",
                api_key="test",
                model_name="test",
                max_completion_tokens=500,
                max_tokens=500,
            )


class TestRequestBodyKey:
    def test_set_at_instance(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
            max_completion_tokens=300,
        )
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
        )
        assert data["max_completion_tokens"] == 300
        assert "max_tokens" not in data

    def test_per_call_override(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
        )
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
            max_completion_tokens=200,
        )
        assert data["max_completion_tokens"] == 200
        assert "max_tokens" not in data

    def test_logprobs_params(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
        )
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
            logprobs=True,
            top_logprobs=20,
        )
        assert data["logprobs"] is True
        assert data["top_logprobs"] == 20

    def test_logprobs_absent_by_default(self):
        lm = OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test",
        )
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
        )
        assert "logprobs" not in data
        assert "top_logprobs" not in data
