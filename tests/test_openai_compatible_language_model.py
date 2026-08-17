"""Tests for max_tokens → max_completion_tokens migration."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

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


@pytest.mark.asyncio
async def test_agenerate_single_honors_env_proxy(monkeypatch):
    """The LM routes through HTTP_PROXY (requires trust_env=True on the session).

    Pointing the LM at an unresolvable host means a direct connection can only
    fail; the request succeeds only because it is routed to our stand-in proxy.
    """

    # Stand-in proxy: answers every path with a valid chat-completion payload.
    async def stand_in_proxy_handler(request):
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    proxy_app = web.Application()
    proxy_app.router.add_route("*", "/{tail:.*}", stand_in_proxy_handler)
    stand_in_proxy = TestServer(proxy_app)
    await stand_in_proxy.start_server()
    try:
        # Clear inherited no-proxy config that could bypass our stand-in.
        for inherited_var in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(inherited_var, raising=False)
        proxy_url = f"http://127.0.0.1:{stand_in_proxy.port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)

        lm = OpenAICompatibleLanguageModel(
            endpoint="http://blackhole.invalid/v1",  # unresolvable without a proxy
            api_key="k",
            model_name="m",
            max_tries=1,
        )
        response = await lm.agenerate_single([ChatMessage(role="user", content="hi")])
        await lm.close()
    finally:
        await stand_in_proxy.close()

    assert response["content"] == "ok"
