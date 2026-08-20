"""Tests for ITSGateway (core/gateway.py)."""

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from its_hub.api.types import (
    SUPPORTED_ALGORITHMS,
    GenerationUsage,
    ITSRequestConfigUpdate,
)
from its_hub.core.gateway import ITSGateway
from tests.mocks.recording_llm import RecordingLLMHandler


def _make_result(usage=None):
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
    defaults = {
        "budget": 3,
        "alg": "self-consistency",
        "api_endpoint": "http://llm/v1",
        "model": "gpt-4",
        "api_key": "sk-test",
    }
    defaults.update(overrides)
    return ITSRequestConfigUpdate(**defaults)


MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


def _mock_lm():
    lm = MagicMock()
    lm.close = AsyncMock()
    return lm


def _patch_build_lm(gw):
    return patch.object(gw, "_build_lm", return_value=_mock_lm())


def _algo(gw):
    return gw._build_algorithm(gw.default_config.resolve())


class TestGatewayConstruction:
    def test_default_init(self):
        gw = ITSGateway()
        assert gw._orchestrator is not None
        # Stored default is partial; system defaults (alg, budget) materialize
        # at resolve() time on ITSRequestConfig.
        resolved = gw.default_config.merge(
            ITSRequestConfigUpdate(api_endpoint="http://x/v1", model="m")
        ).resolve()
        assert resolved.alg == "self-consistency"
        assert resolved.budget == 4

    def test_custom_default_config(self):
        config = ITSRequestConfigUpdate(
            alg="beta-self-consistency",
            api_endpoint="http://x/v1",
            model="m",
            api_key="k",
        )
        gw = ITSGateway(default_config=config)
        assert gw.default_config is config
        assert gw.default_config.alg == "beta-self-consistency"

    def test_custom_orchestrator_passed_to_algorithm(self):
        orch = MagicMock()
        gw = ITSGateway(orchestrator=orch)
        assert gw._orchestrator is orch
        algo = gw._build_algorithm(_make_config().resolve())
        assert algo.orchestrator is orch


class TestLMClientLifecycle:
    @pytest.mark.asyncio
    async def test_creates_new_lm_per_request(self):
        gw = ITSGateway()
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result())
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            patch.object(gw, "_build_lm", return_value=_mock_lm()) as build_lm,
        ):
            await gw.arun_chat_completion(_make_config(), MESSAGES)
            await gw.arun_chat_completion(_make_config(), MESSAGES)
            assert build_lm.call_count == 2

    @pytest.mark.asyncio
    async def test_closes_lm_after_request(self):
        gw = ITSGateway()
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result())
        lm = _mock_lm()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            patch.object(gw, "_build_lm", return_value=lm),
        ):
            await gw.arun_chat_completion(_make_config(), MESSAGES)
        lm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_lm_even_when_algorithm_raises(self):
        gw = ITSGateway()
        algo = MagicMock()
        algo.ainfer = AsyncMock(side_effect=RuntimeError("boom"))
        lm = _mock_lm()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            patch.object(gw, "_build_lm", return_value=lm),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await gw.arun_chat_completion(_make_config(), MESSAGES)
        lm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_different_models_do_not_interfere(self, llm_server):
        gw = ITSGateway()

        RecordingLLMHandler.hold()

        config_a = _make_config(api_endpoint=f"{llm_server}/v1", model="model-A")
        config_b = _make_config(api_endpoint=f"{llm_server}/v1", model="model-B")

        task_a = asyncio.create_task(gw.arun_chat_completion(config_a, MESSAGES))
        await RecordingLLMHandler.wait_for_bodies(3)

        task_b = asyncio.create_task(gw.arun_chat_completion(config_b, MESSAGES))
        await RecordingLLMHandler.wait_for_bodies(6)

        RecordingLLMHandler.release()

        result_a = await task_a
        result_b = await task_b

        assert result_a["message"]["content"] == "answer from model-A"
        assert result_b["message"]["content"] == "answer from model-B"


class TestRunChatCompletion:
    @pytest.mark.asyncio
    async def test_response_only(self):
        usage = GenerationUsage(prompt_tokens=10, completion_tokens=20, num_calls=3)
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result(usage))
        gw = ITSGateway()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            _patch_build_lm(gw),
        ):
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
        gw = ITSGateway()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            _patch_build_lm(gw),
        ):
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
        gw = ITSGateway()
        tools = [{"type": "function", "function": {"name": "f"}}]
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            _patch_build_lm(gw),
        ):
            await gw.arun_chat_completion(
                _make_config(), MESSAGES, tools=tools, tool_choice="auto"
            )
        call_kwargs = algo.ainfer.call_args.kwargs
        assert call_kwargs["tools"] is tools
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_missing_model_raises(self):
        gw = ITSGateway()
        config = _make_config(model=None)
        with pytest.raises(ValueError, match="model must be specified"):
            await gw.arun_chat_completion(config, MESSAGES)

    @pytest.mark.asyncio
    async def test_algorithm_exception_propagates(self):
        algo = MagicMock()
        algo.ainfer = AsyncMock(side_effect=RuntimeError("boom"))
        gw = ITSGateway()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            _patch_build_lm(gw),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await gw.arun_chat_completion(_make_config(), MESSAGES)

    @pytest.mark.asyncio
    async def test_usage_empty_when_none(self):
        algo = MagicMock()
        algo.ainfer = AsyncMock(return_value=_make_result(usage=None))
        gw = ITSGateway()
        with (
            patch.object(gw, "_build_algorithm", return_value=algo),
            _patch_build_lm(gw),
        ):
            result = await gw.arun_chat_completion(_make_config(), MESSAGES)
        assert result["usage"] == {}

    @pytest.mark.asyncio
    async def test_temperature_forwarded_to_lm(self, llm_server):
        gw = ITSGateway()
        config = _make_config(api_endpoint=f"{llm_server}/v1", temperature=0.7)
        await gw.arun_chat_completion(config, MESSAGES)
        assert RecordingLLMHandler.received_bodies[-1].get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_reconfigure_does_not_swap_in_flight_algorithm(self, llm_server):
        """Reconfigure mid-request must not swap the algorithm used by the
        in-flight request."""
        gw = ITSGateway()
        # `alg` is deliberately omitted from the per-request overlay so the
        # algorithm is sourced from the (reconfigurable) service default —
        # only then does a mid-flight /configure actually exercise the
        # snapshot. With `alg` in the overlay, `result["alg"]` would be
        # "self-consistency" regardless of any reconfigure.
        base = ITSRequestConfigUpdate(
            api_endpoint=f"{llm_server}/v1",
            model="m",
        )
        gw.configure(base)

        RecordingLLMHandler.hold()

        task = asyncio.create_task(gw.arun_chat_completion(base, MESSAGES))
        await RecordingLLMHandler.wait_for_bodies(1)

        gw.configure(
            ITSRequestConfigUpdate(
                alg="beta-self-consistency",
            )
        )

        RecordingLLMHandler.release()
        result = await task

        assert result["alg"] == "self-consistency"
        assert gw.default_config.alg == "beta-self-consistency"


class TestConfigure:
    def test_configure_self_consistency(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(alg="self-consistency", regex_patterns=[r"\\boxed{([^}]+)}"]),
        )
        assert gw.default_config.alg == "self-consistency"
        assert type(_algo(gw)).__name__ == "SelfConsistency"

    def test_configure_defaults_tool_vote_when_omitted(self):
        gw = ITSGateway()
        gw.configure(_make_config(alg="self-consistency"))
        assert _algo(gw).tool_vote == "tool_hierarchical"

    def test_configure_with_tool_vote(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(
                alg="self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
                tool_vote="tool_name",
                exclude_tool_args=["timestamp"],
            ),
        )
        algo = _algo(gw)
        assert algo.tool_vote == "tool_name"
        assert algo.exclude_args == ["timestamp"]

    def test_configure_tool_vote_persists_when_omitted(self):
        """`None` means trickle down, so a /configure omitting tool_vote
        cannot clear a previously-set value."""
        gw = ITSGateway()
        gw.configure(_make_config(alg="self-consistency", tool_vote="tool_name"))
        gw.configure(_make_config(alg="self-consistency"))  # no tool_vote
        assert _algo(gw).tool_vote == "tool_name"

    def test_overlay_unsupported_algorithm(self):
        gw = ITSGateway()
        with pytest.raises(ValueError, match="not supported"):
            gw.configure(_make_config(alg="beam-search"))

    def test_overlay_invalid_regex(self):
        gw = ITSGateway()
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            gw.configure(
                _make_config(alg="self-consistency", regex_patterns=["[invalid("]),
            )

    def test_configure_preserves_orchestrator(self):
        orch = MagicMock()
        gw = ITSGateway(orchestrator=orch)
        gw.configure(
            _make_config(alg="self-consistency", regex_patterns=[r"\\boxed{([^}]+)}"]),
        )
        assert _algo(gw).orchestrator is orch

    def test_overlay_invalid_tool_vote(self):
        gw = ITSGateway()
        with pytest.raises(ValueError, match="tool_vote must be one of"):
            gw.configure(
                _make_config(
                    alg="self-consistency",
                    regex_patterns=[r"\\boxed{([^}]+)}"],
                    tool_vote="invalid",
                ),
            )

    def test_overlay_invalid_temperature(self):
        gw = ITSGateway()
        with pytest.raises(ValueError, match="temperature must be in"):
            gw.configure(_make_config(temperature=3.0))

    def test_configure_adaptive_self_consistency(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(
                alg="adaptive-self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
            ),
        )
        algo = _algo(gw)
        assert type(algo).__name__ == "AdaptiveSelfConsistency"
        assert algo.threshold == pytest.approx(0.75)

    def test_configure_adaptive_threshold_plumbed(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(
                alg="adaptive-self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
                threshold=0.9,
            ),
        )
        assert _algo(gw).threshold == pytest.approx(0.9)

    def test_configure_beta_self_consistency(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(
                alg="beta-self-consistency", regex_patterns=[r"\\boxed{([^}]+)}"]
            ),
        )
        algo = _algo(gw)
        assert type(algo).__name__ == "BetaSelfConsistency"
        assert algo.confidence_threshold == pytest.approx(0.95)

    def test_configure_beta_confidence_threshold_plumbed(self):
        gw = ITSGateway()
        gw.configure(
            _make_config(
                alg="beta-self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
                confidence_threshold=0.8,
            ),
        )
        assert _algo(gw).confidence_threshold == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "alg",
        ["adaptive-self-consistency", "beta-self-consistency"],
    )
    def test_configure_family_shares_tool_vote(self, alg):
        gw = ITSGateway()
        gw.configure(_make_config(alg=alg, tool_vote="tool_name"))
        assert _algo(gw).tool_vote == "tool_name"

    def test_configure_preserves_orchestrator_for_family(self):
        orch = MagicMock()
        gw = ITSGateway(orchestrator=orch)
        gw.configure(
            _make_config(
                alg="beta-self-consistency", regex_patterns=[r"\\boxed{([^}]+)}"]
            ),
        )
        assert _algo(gw).orchestrator is orch


class TestSupportedAlgorithms:
    def test_self_consistency_supported(self):
        assert "self-consistency" in SUPPORTED_ALGORITHMS

    def test_adaptive_self_consistency_supported(self):
        assert "adaptive-self-consistency" in SUPPORTED_ALGORITHMS

    def test_beta_self_consistency_supported(self):
        assert "beta-self-consistency" in SUPPORTED_ALGORITHMS

    def test_is_frozenset(self):
        assert isinstance(SUPPORTED_ALGORITHMS, frozenset)


class TestEndpointValidation:
    @pytest.mark.asyncio
    async def test_missing_endpoint_raises(self):
        gw = ITSGateway()
        config = _make_config(api_endpoint="")
        with pytest.raises(ValueError, match="api_endpoint"):
            await gw.arun_chat_completion(config, MESSAGES)
