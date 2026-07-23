"""Tests for BetaSelfConsistency algorithm."""

import asyncio
import threading

import pytest

from its_hub.api import ChatMessages
from its_hub.core.algorithms.beta_self_consistency import BetaSelfConsistency
from its_hub.core.algorithms.self_consistency import (
    SelfConsistencyResult,
    create_regex_projection_function,
)
from its_hub.core.orchestrator import LMOrchestrator


class SequentialMockLM:
    """Mock LM that returns responses in sequential order, thread-safe."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0
        self._lock = threading.Lock()

    async def agenerate_single(self, messages, **kwargs):
        with self._lock:
            idx = self.call_count % len(self.responses)
            self.call_count += 1
        resp = self.responses[idx]
        if isinstance(resp, dict):
            return resp
        return {"role": "assistant", "content": resp}


class DelayedMockLM:
    """Mock LM where each response has a controlled delay to test ordering.
    Responses arrive in delay order, so we can predict which arrive first.
    """

    def __init__(self, responses_with_delays: list[tuple]):
        """Args: list of (response_content, delay_seconds)"""
        self.responses_with_delays = responses_with_delays
        self.call_count = 0
        self._lock = threading.Lock()

    async def agenerate_single(self, messages, **kwargs):
        with self._lock:
            idx = self.call_count % len(self.responses_with_delays)
            self.call_count += 1
        content, delay = self.responses_with_delays[idx]
        await asyncio.sleep(delay)
        if isinstance(content, dict):
            return content
        return {"role": "assistant", "content": content}


class TestBetaSelfConsistencyInit:
    """Test constructor validation."""

    def test_default_threshold(self):
        bsc = BetaSelfConsistency()
        assert bsc.confidence_threshold == 0.95

    def test_custom_threshold(self):
        bsc = BetaSelfConsistency(confidence_threshold=0.8)
        assert bsc.confidence_threshold == 0.8

    def test_threshold_at_boundary(self):
        bsc = BetaSelfConsistency(confidence_threshold=1.0)
        assert bsc.confidence_threshold == 1.0

    @pytest.mark.parametrize("bad_threshold", [0.0, 0.5, -0.1, 1.1])
    def test_invalid_threshold_raises(self, bad_threshold):
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            BetaSelfConsistency(confidence_threshold=bad_threshold)

    def test_default_orchestrator_created(self):
        bsc = BetaSelfConsistency()
        assert isinstance(bsc.orchestrator, LMOrchestrator)

    def test_custom_orchestrator_used(self):
        orch = LMOrchestrator(max_concurrency=4)
        bsc = BetaSelfConsistency(orchestrator=orch)
        assert bsc.orchestrator is orch

    def test_inherits_from_self_consistency(self):
        from its_hub.core.algorithms.self_consistency import SelfConsistency

        bsc = BetaSelfConsistency()
        assert isinstance(bsc, SelfConsistency)


class TestBetaStoppingProbability:
    """Test the static beta_stopping_probability method."""

    def test_unanimous_1(self):
        prob = BetaSelfConsistency.beta_stopping_probability(1, 0)
        assert abs(prob - 0.75) < 1e-10

    def test_unanimous_2(self):
        prob = BetaSelfConsistency.beta_stopping_probability(2, 0)
        assert abs(prob - 0.875) < 1e-10

    def test_unanimous_4(self):
        prob = BetaSelfConsistency.beta_stopping_probability(4, 0)
        assert abs(prob - 0.96875) < 1e-10

    def test_split_50_50(self):
        prob = BetaSelfConsistency.beta_stopping_probability(5, 5)
        assert abs(prob - 0.5) < 1e-10

    def test_strong_majority(self):
        prob = BetaSelfConsistency.beta_stopping_probability(10, 1)
        assert prob > 0.99

    def test_monotonic_with_v1(self):
        probs = [
            BetaSelfConsistency.beta_stopping_probability(v1, 2) for v1 in range(3, 10)
        ]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_monotonic_with_v2(self):
        probs = [
            BetaSelfConsistency.beta_stopping_probability(5, v2) for v2 in range(0, 5)
        ]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1]


class TestBetaSelfConsistencyEarlyStopping:
    """Test the fire-all-cancel-early behavior."""

    @pytest.mark.asyncio
    async def test_stops_early_with_agreement(self):
        """With all identical answers, should use fewer than budget samples."""
        lm = SequentialMockLM(["42"] * 64)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=64, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) < 64
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_uses_full_budget_when_split(self):
        """Alternating answers never reach 0.95 confidence."""
        lm = SequentialMockLM(["a", "b"] * 32)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert len(result.responses) == 8

    @pytest.mark.asyncio
    async def test_budget_1(self):
        """Budget=1: single sample, no stopping check."""
        lm = SequentialMockLM(["only_answer"])
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=1, return_response_only=True)

        assert result["content"] == "only_answer"
        assert lm.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_2_same_answers(self):
        """Budget=2: 2 identical. P=0.875 < 0.95 so no early stop."""
        lm = SequentialMockLM(["yes", "yes"])
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=2, return_response_only=False)

        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_threshold_1_0_uses_full_budget(self):
        """threshold=1.0 is impossible to reach, should exhaust budget."""
        lm = SequentialMockLM(["42"] * 8)
        bsc = BetaSelfConsistency(confidence_threshold=1.0)

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert len(result.responses) == 8

    @pytest.mark.asyncio
    async def test_lower_threshold_stops_sooner(self):
        """Lower threshold requires fewer samples for the same agreement."""
        lm_strict = SequentialMockLM(["42"] * 64)
        lm_relaxed = SequentialMockLM(["42"] * 64)

        bsc_strict = BetaSelfConsistency(confidence_threshold=0.95)
        bsc_relaxed = BetaSelfConsistency(confidence_threshold=0.8)

        result_strict = await bsc_strict.ainfer(
            lm_strict, "test", budget=64, return_response_only=False
        )
        result_relaxed = await bsc_relaxed.ainfer(
            lm_relaxed, "test", budget=64, return_response_only=False
        )

        assert len(result_relaxed.responses) <= len(result_strict.responses)

    @pytest.mark.asyncio
    async def test_delayed_ordering_early_stop(self):
        """Fast-arriving unanimous responses trigger early stop while slow ones are cancelled."""
        # 4 fast "42" responses (arrive first), 4 slow responses (should be cancelled)
        responses = [("42", 0.0)] * 4 + [("99", 0.5)] * 4
        lm = DelayedMockLM(responses)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        # v1=4, v2=0 → P=0.96875 ≥ 0.95 → stop after 4 fast responses
        assert len(result.responses) == 4
        assert all(r["content"] == "42" for r in result.responses)


class TestBetaSelfConsistencyProjection:
    """Test with custom projection functions."""

    @pytest.mark.asyncio
    async def test_regex_projection(self):
        pattern = r"\\boxed\{([^}]+)\}"
        proj_func = create_regex_projection_function(pattern)

        lm = SequentialMockLM(
            ["The answer is \\boxed{42}."] * 16 + ["\\boxed{99}"] * 48
        )
        bsc = BetaSelfConsistency(
            confidence_threshold=0.95,
            consistency_space_projection_func=proj_func,
        )

        result = await bsc.ainfer(lm, "test", budget=64, return_response_only=False)

        assert len(result.responses) < 64

    @pytest.mark.asyncio
    async def test_default_projection_strips_whitespace(self):
        lm = SequentialMockLM(["  answer  ", "answer", "  answer  ", "answer"] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=64, return_response_only=False)

        assert len(result.responses) < 64


class TestBetaSelfConsistencyInterface:
    """Test the algorithm interface and result types."""

    @pytest.mark.asyncio
    async def test_return_response_only_true(self):
        lm = SequentialMockLM(["42"] * 8)
        bsc = BetaSelfConsistency()

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=True)

        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert result["content"] == "42"

    @pytest.mark.asyncio
    async def test_return_response_only_false(self):
        lm = SequentialMockLM(["42"] * 8)
        bsc = BetaSelfConsistency()

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert result.the_one["content"] == "42"
        assert result.usage is not None

    def test_sync_infer(self):
        lm = SequentialMockLM(["42"] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = bsc.infer(lm, "test", budget=16, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_with_chat_messages(self):
        lm = SequentialMockLM(["42"] * 8)
        bsc = BetaSelfConsistency()

        chat_messages = ChatMessages("Solve this problem")
        result = await bsc.ainfer(
            lm, chat_messages, budget=8, return_response_only=True
        )

        assert result["content"] == "42"


class TestBetaSelfConsistencyToolCalls:
    """Test tool-call voting with beta stopping."""

    def _make_tool_response(self, name, args):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        }

    @pytest.mark.asyncio
    async def test_tool_vote_name_early_stop(self):
        resp = self._make_tool_response("get_weather", '{"city": "NYC"}')
        lm = SequentialMockLM([resp] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.95, tool_vote="tool_name")

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        assert len(result.responses) < 16
        assert result.the_one["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_exclude_args_in_tool_voting(self):
        r1 = self._make_tool_response("search", '{"q": "cats", "request_id": "abc"}')
        r2 = self._make_tool_response("search", '{"q": "cats", "request_id": "xyz"}')
        lm = SequentialMockLM([r1, r2] * 8)
        bsc = BetaSelfConsistency(
            confidence_threshold=0.95,
            tool_vote="tool_args",
            exclude_args=["request_id"],
        )

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        # After excluding request_id, all have identical args → stops early
        assert len(result.responses) < 16
