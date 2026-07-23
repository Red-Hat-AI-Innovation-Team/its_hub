"""Tests for BetaSelfConsistency algorithm."""

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
        # v1=1, v2=0: P = 1 - I_0.5(2, 1) = 1 - 0.25 = 0.75
        prob = BetaSelfConsistency.beta_stopping_probability(1, 0)
        assert abs(prob - 0.75) < 1e-10

    def test_unanimous_2(self):
        # v1=2, v2=0: P = 1 - I_0.5(3, 1) = 1 - 0.125 = 0.875
        prob = BetaSelfConsistency.beta_stopping_probability(2, 0)
        assert abs(prob - 0.875) < 1e-10

    def test_unanimous_4(self):
        # v1=4, v2=0: P = 1 - I_0.5(5, 1) = 1 - 0.03125 = 0.96875
        prob = BetaSelfConsistency.beta_stopping_probability(4, 0)
        assert abs(prob - 0.96875) < 1e-10

    def test_split_50_50(self):
        # v1=v2: should be close to 0.5 by symmetry
        prob = BetaSelfConsistency.beta_stopping_probability(5, 5)
        assert abs(prob - 0.5) < 1e-10

    def test_strong_majority(self):
        # v1=10, v2=1: should be very high
        prob = BetaSelfConsistency.beta_stopping_probability(10, 1)
        assert prob > 0.99

    def test_monotonic_with_v1(self):
        # More v1 counts → higher probability
        probs = [
            BetaSelfConsistency.beta_stopping_probability(v1, 2) for v1 in range(3, 10)
        ]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_monotonic_with_v2(self):
        # More v2 counts → lower probability
        probs = [
            BetaSelfConsistency.beta_stopping_probability(5, v2) for v2 in range(0, 5)
        ]
        for i in range(len(probs) - 1):
            assert probs[i] > probs[i + 1]


class TestBetaSelfConsistencyEarlyStopping:
    """Test the sample-one-at-a-time + beta stopping behavior."""

    @pytest.mark.asyncio
    async def test_stops_with_unanimous_agreement(self):
        """With all identical answers at threshold=0.95, needs 4 samples to stop.
        v1=4, v2=0 → P=0.96875 ≥ 0.95."""
        lm = SequentialMockLM(["42"] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 4
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_unanimous_with_lower_threshold(self):
        """With threshold=0.8, unanimous answers need 2 samples.
        v1=2, v2=0 → P=0.875 ≥ 0.8."""
        lm = SequentialMockLM(["42"] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.8)

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_disagreement_delays_stopping(self):
        """One disagreement requires more samples to reach confidence."""
        # Responses: 42, 24, 42, 42, 42, 42, 42, ...
        lm = SequentialMockLM(["42", "24", "42", "42", "42", "42", "42", "42"])
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        # After 2: v1=1,v2=1 → P=0.5
        # After 3: v1=2,v2=1 → P=0.6875
        # After 4: v1=3,v2=1 → P=0.8125
        # After 5: v1=4,v2=1 → P=0.890625
        # After 6: v1=5,v2=1 → P=0.9375
        # After 7: v1=6,v2=1 → P=0.964844 → stop
        assert len(result.responses) == 7
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_uses_full_budget_when_split(self):
        """Alternating answers never reach 0.95 confidence."""
        lm = SequentialMockLM(["a", "b"] * 8)
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
        """Budget=2: 2 identical answers. v1=2,v2=0 → P=0.875 < 0.95.
        Can't stop early because budget is exhausted."""
        lm = SequentialMockLM(["yes", "yes"])
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=2, return_response_only=False)

        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_samples_one_at_a_time(self):
        """Verify that exactly one sample is generated per iteration."""
        lm = SequentialMockLM(["42"] * 16)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        # Should stop at 4 samples, each generated individually
        assert lm.call_count == 4

    @pytest.mark.asyncio
    async def test_threshold_1_0_requires_very_high_confidence(self):
        """threshold=1.0 is impossible to reach, should exhaust budget."""
        lm = SequentialMockLM(["42"] * 8)
        bsc = BetaSelfConsistency(confidence_threshold=1.0)

        result = await bsc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert len(result.responses) == 8


class TestBetaSelfConsistencyProjection:
    """Test with custom projection functions."""

    @pytest.mark.asyncio
    async def test_regex_projection(self):
        pattern = r"\\boxed\{([^}]+)\}"
        proj_func = create_regex_projection_function(pattern)

        lm = SequentialMockLM(
            [
                "Let me solve this. The answer is \\boxed{42}.",
                "Using algebra, we get \\boxed{42}.",
                "By computation \\boxed{42}.",
                "Therefore \\boxed{42}.",
            ]
            + ["\\boxed{99}"] * 12
        )
        bsc = BetaSelfConsistency(
            confidence_threshold=0.95,
            consistency_space_projection_func=proj_func,
        )

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        # All 4 project to ("42",) → stops at 4 (unanimous)
        assert len(result.responses) == 4

    @pytest.mark.asyncio
    async def test_default_projection_strips_whitespace(self):
        lm = SequentialMockLM(
            ["  answer  ", "answer", "  answer  ", "answer"] + ["other"] * 12
        )
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        # All project to "answer" → unanimous → stops at 4
        assert len(result.responses) == 4


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
        lm = SequentialMockLM(["42", "42", "42", "42", "24"] + ["42"] * 3)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = bsc.infer(lm, "test", budget=8, return_response_only=False)

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

    @pytest.mark.asyncio
    async def test_response_counts_populated(self):
        lm = SequentialMockLM(["42", "24", "42", "42", "42"] + ["42"] * 11)
        bsc = BetaSelfConsistency(confidence_threshold=0.95)

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        assert result.response_counts["42"] >= 3
        assert "24" in result.response_counts or len(result.responses) <= 4


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

        # Unanimous tool name → stops at 4
        assert len(result.responses) == 4
        assert result.the_one["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_vote_args_with_disagreement(self):
        r1 = self._make_tool_response("search", '{"q": "cats"}')
        r2 = self._make_tool_response("search", '{"q": "dogs"}')
        # cats, dogs, cats, cats, cats, cats, cats → v1=6,v2=1 at sample 7 → P=0.9648 → stop
        lm = SequentialMockLM([r1, r2, r1, r1, r1, r1, r1] + [r1] * 9)
        bsc = BetaSelfConsistency(confidence_threshold=0.95, tool_vote="tool_args")

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        assert len(result.responses) == 7

    @pytest.mark.asyncio
    async def test_exclude_args_in_tool_voting(self):
        r1 = self._make_tool_response("search", '{"q": "cats", "request_id": "abc"}')
        r2 = self._make_tool_response("search", '{"q": "cats", "request_id": "xyz"}')
        lm = SequentialMockLM([r1, r2, r1, r2] + [r1] * 12)
        bsc = BetaSelfConsistency(
            confidence_threshold=0.95,
            tool_vote="tool_args",
            exclude_args=["request_id"],
        )

        result = await bsc.ainfer(lm, "test", budget=16, return_response_only=False)

        # After excluding request_id, all have args {"q": "cats"} → unanimous
        assert len(result.responses) == 4
