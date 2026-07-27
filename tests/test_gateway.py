"""Tests for ITSGateway (core/gateway.py)."""

from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from its_hub.api.types import GenerationUsage, ITSRequestConfig
from its_hub.core.gateway import SUPPORTED_ALGORITHMS, ITSGateway


def _make_result(usage=None):
    """Build a mock SelfConsistencyResult-like object."""
    result = MagicMock()
    result.the_one = {"role": "assistant", "content": "answer"}
    result.responses = [
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": "other"},
    ]
    result.response_counts = Counter({"answer": 2, "other": 1})
    result.selected_index = 0
    result.usage = usage
    return result


def _make_config(**overrides):
    defaults = {"budget": 3, "api_endpoint": "http://llm/v1", "model": "gpt-4", "api_key": "sk-test"}
    defaults.update(overrides)
    return ITSRequestConfig(**defaults)


MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


class TestGatewayConstruction:
    def test_default_init(self):
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel"):
            gw = ITSGateway()
        assert gw._algorithm is not None
        assert gw._orchestrator is not None
        assert gw._algorithm_name == "SelfConsistency"

    def test_custom_algorithm(self):
        algo = MagicMock()
        type(algo).__name__ = "CustomAlgo"
        gw = ITSGateway(algorithm=algo)
        assert gw._algorithm is algo
        assert gw._algorithm_name == "CustomAlgo"

    def test_custom_orchestrator_passed_to_default_algorithm(self):
        orch = MagicMock()
        with patch("its_hub.core.gateway.SelfConsistency") as sc_cls:
            ITSGateway(orchestrator=orch)
            sc_cls.assert_called_once_with(orchestrator=orch)


class TestLMClientCaching:
    @pytest.mark.asyncio
    async def test_creates_new_client(self):
        gw = ITSGateway(algorithm=MagicMock())
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel") as lm_cls:
            lm_cls.return_value = MagicMock()
            lm = await gw._get_or_create_lm("http://a/v1", "m1", "key1")
            assert lm is lm_cls.return_value
            lm_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_reuses_cached_client(self):
        gw = ITSGateway(algorithm=MagicMock())
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel") as lm_cls:
            lm_cls.return_value = MagicMock()
            lm1 = await gw._get_or_create_lm("http://a/v1", "m1", "key1")
            lm2 = await gw._get_or_create_lm("http://a/v1", "m1", "key1")
            assert lm1 is lm2
            assert lm_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_different_endpoint_creates_new_client(self):
        gw = ITSGateway(algorithm=MagicMock())
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel") as lm_cls:
            lm_cls.return_value = MagicMock()
            await gw._get_or_create_lm("http://a/v1", "m1", "key1")
            lm_cls.return_value = MagicMock()
            await gw._get_or_create_lm("http://b/v1", "m1", "key1")
            assert lm_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_different_api_key_creates_new_client(self):
        gw = ITSGateway(algorithm=MagicMock())
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel") as lm_cls:
            lm_cls.return_value = MagicMock()
            await gw._get_or_create_lm("http://a/v1", "m1", "key-A")
            lm_cls.return_value = MagicMock()
            await gw._get_or_create_lm("http://a/v1", "m1", "key-B")
            assert lm_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_ashutdown_closes_clients(self):
        gw = ITSGateway(algorithm=MagicMock())
        mock_lm = MagicMock()
        mock_lm.close = AsyncMock()
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel", return_value=mock_lm):
            await gw._get_or_create_lm("http://a/v1", "m1", "key1")
        assert len(gw._lm_cache) == 1
        await gw.ashutdown()
        mock_lm.close.assert_awaited_once()
        assert len(gw._lm_cache) == 0

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_cache_full(self):
        gw = ITSGateway(algorithm=MagicMock(), max_lm_cache_size=2)
        mocks = []
        with patch("its_hub.core.gateway.OpenAICompatibleLanguageModel") as lm_cls:
            for i in range(3):
                mock_lm = MagicMock()
                mock_lm.close = AsyncMock()
                lm_cls.return_value = mock_lm
                mocks.append(mock_lm)
                await gw._get_or_create_lm(f"http://host{i}/v1", "m1", "key1")

        assert len(gw._lm_cache) == 2
        mocks[0].close.assert_awaited_once()


class TestRunChatCompletion:
    @pytest.mark.asyncio
    async def test_response_only(self):
        usage = GenerationUsage(prompt_tokens=10, completion_tokens=20, num_calls=3)
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result(usage))
        gw = ITSGateway(algorithm=algo)
        with patch.object(gw, "_get_or_create_lm", return_value=MagicMock()):
            result = await gw.arun_chat_completion(_make_config(), MESSAGES)
        assert result["message"] == {"role": "assistant", "content": "answer"}
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30
        assert result["usage"]["num_calls"] == 3

    @pytest.mark.asyncio
    async def test_full_result(self):
        usage = GenerationUsage(prompt_tokens=5, completion_tokens=10, num_calls=2)
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result(usage))
        gw = ITSGateway(algorithm=algo)
        with patch.object(gw, "_get_or_create_lm", return_value=MagicMock()):
            result = await gw.arun_chat_completion(
                _make_config(), MESSAGES, return_response_only=False
            )
        assert "responses" in result
        assert "response_counts" in result
        assert "selected_index" in result
        assert "the_one" in result
        assert result["selected_index"] == 0

    @pytest.mark.asyncio
    async def test_tools_forwarded(self):
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result())
        gw = ITSGateway(algorithm=algo)
        tools = [{"type": "function", "function": {"name": "f"}}]
        with patch.object(gw, "_get_or_create_lm", return_value=MagicMock()):
            await gw.arun_chat_completion(_make_config(), MESSAGES, tools=tools, tool_choice="auto")
        call_kwargs = algo.ainfer.call_args.kwargs
        assert call_kwargs["tools"] is tools
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_missing_model_raises(self):
        gw = ITSGateway(algorithm=MagicMock())
        config = _make_config(model=None)
        with pytest.raises(ValueError, match="Model must be specified"):
            await gw.arun_chat_completion(config, MESSAGES)

    @pytest.mark.asyncio
    async def test_algorithm_exception_propagates(self):
        algo = MagicMock()
        algo.ainfer = AsyncMock(side_effect=RuntimeError("boom"))
        gw = ITSGateway(algorithm=algo)
        with patch.object(gw, "_get_or_create_lm", return_value=MagicMock()), \
             pytest.raises(RuntimeError, match="boom"):
            await gw.arun_chat_completion(_make_config(), MESSAGES)

    @pytest.mark.asyncio
    async def test_usage_empty_when_none(self):
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result(usage=None))
        gw = ITSGateway(algorithm=algo)
        with patch.object(gw, "_get_or_create_lm", return_value=MagicMock()):
            result = await gw.arun_chat_completion(_make_config(), MESSAGES)
        assert result["usage"] == {}


class TestHashApiKey:
    def test_same_key_same_hash(self):
        assert ITSGateway._hash_api_key("sk-abc") == ITSGateway._hash_api_key("sk-abc")

    def test_different_keys_different_hash(self):
        assert ITSGateway._hash_api_key("sk-abc") != ITSGateway._hash_api_key("sk-xyz")

    def test_none_key(self):
        h = ITSGateway._hash_api_key(None)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_hash_length(self):
        assert len(ITSGateway._hash_api_key("sk-test")) == 16


class TestConfigure:
    def test_configure_self_consistency(self):
        gw = ITSGateway(algorithm=MagicMock())
        gw.configure(
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
        )
        assert gw._algorithm_name == "SelfConsistency"

    def test_configure_with_tool_vote(self):
        gw = ITSGateway(algorithm=MagicMock())
        gw.configure(
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
            tool_vote="tool_name",
            exclude_tool_args=["timestamp"],
        )
        assert gw._algorithm_name == "SelfConsistency"

    def test_configure_unsupported_algorithm(self):
        gw = ITSGateway(algorithm=MagicMock())
        with pytest.raises(ValueError, match="not supported"):
            gw.configure(alg="beam-search")

    def test_configure_invalid_regex(self):
        gw = ITSGateway(algorithm=MagicMock())
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            gw.configure(alg="self-consistency", regex_patterns=["[invalid("])

    def test_configure_invalid_tool_vote(self):
        gw = ITSGateway(algorithm=MagicMock())
        with pytest.raises(ValueError, match="tool_vote must be one of"):
            gw.configure(
                alg="self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
                tool_vote="invalid",
            )


class TestSupportedAlgorithms:
    def test_self_consistency_supported(self):
        assert "self-consistency" in SUPPORTED_ALGORITHMS

    def test_is_frozenset(self):
        assert isinstance(SUPPORTED_ALGORITHMS, frozenset)


class TestEndpointValidation:
    @pytest.mark.asyncio
    async def test_missing_endpoint_raises(self):
        gw = ITSGateway(algorithm=MagicMock())
        config = _make_config(api_endpoint="")
        with pytest.raises(ValueError, match="api_endpoint"):
            await gw.arun_chat_completion(config, MESSAGES)
