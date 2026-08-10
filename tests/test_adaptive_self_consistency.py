"""Tests for AdaptiveSelfConsistency algorithm."""

import threading

import pytest

from its_hub.api import ChatMessages
from its_hub.core.algorithms.adaptive_self_consistency import AdaptiveSelfConsistency
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


class TestAdaptiveSelfConsistencyInit:
    """Test constructor validation."""

    def test_default_threshold(self):
        asc = AdaptiveSelfConsistency()
        assert asc.threshold == 0.75

    def test_custom_threshold(self):
        asc = AdaptiveSelfConsistency(threshold=0.9)
        assert asc.threshold == 0.9

    def test_threshold_at_boundary(self):
        asc = AdaptiveSelfConsistency(threshold=1.0)
        assert asc.threshold == 1.0

    @pytest.mark.parametrize("bad_threshold", [0.0, 0.5, -0.1, 1.1])
    def test_invalid_threshold_raises(self, bad_threshold):
        with pytest.raises(ValueError, match="threshold must be in"):
            AdaptiveSelfConsistency(threshold=bad_threshold)

    def test_default_orchestrator_created(self):
        asc = AdaptiveSelfConsistency()
        assert isinstance(asc.orchestrator, LMOrchestrator)

    def test_custom_orchestrator_used(self):
        orch = LMOrchestrator(max_concurrency=4)
        asc = AdaptiveSelfConsistency(orchestrator=orch)
        assert asc.orchestrator is orch


class TestAdaptiveSelfConsistencyEarlyStopping:
    """Test the doubling + early stopping behavior."""

    @pytest.mark.asyncio
    async def test_stops_after_round_1_when_all_agree(self):
        """2/2 = 100% >= 75% → stop after first round."""
        lm = SequentialMockLM(["42"] * 8)
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert len(result.responses) == 2
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_stops_after_round_2(self):
        """Round 1: 1/2=50%. Round 2: 3/4=75% → stop."""
        lm = SequentialMockLM(["42", "24", "42", "42", "x", "x", "x", "x"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert len(result.responses) == 4
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_uses_full_budget_when_no_agreement(self):
        """Never reaches 75% → exhausts budget."""
        lm = SequentialMockLM(["a", "b", "c", "d", "a", "b", "c", "d"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        assert len(result.responses) == 8

    @pytest.mark.asyncio
    async def test_doubling_sequence(self):
        """Verify the 2→4→8 doubling pattern when threshold never met."""
        lm = SequentialMockLM(["a", "b", "c", "d", "e", "f", "g", "h"])
        asc = AdaptiveSelfConsistency(threshold=1.0)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # With threshold=1.0 and all different answers, should use full budget
        assert len(result.responses) == 8
        # Verify call count matches budget
        assert lm.call_count == 8

    @pytest.mark.asyncio
    async def test_partial_last_batch(self):
        """Budget=5: rounds are 2→2→1 (last batch is partial)."""
        lm = SequentialMockLM(["a", "b", "c", "d", "e"])
        asc = AdaptiveSelfConsistency(threshold=1.0)

        result = await asc.ainfer(lm, "test", budget=5, return_response_only=False)

        assert len(result.responses) == 5
        assert lm.call_count == 5

    @pytest.mark.asyncio
    async def test_budget_1(self):
        """Budget=1: single sample, no voting needed."""
        lm = SequentialMockLM(["only_answer"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=1, return_response_only=True)

        assert result["content"] == "only_answer"
        assert lm.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_2_agreement(self):
        """Budget=2: 2/2=100% ≥ 75% → stop (both agree)."""
        lm = SequentialMockLM(["yes", "yes"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=2, return_response_only=False)

        # With budget=2, first batch fills budget → no early stopping check needed
        assert len(result.responses) == 2
        assert result.the_one["content"] == "yes"

    @pytest.mark.asyncio
    async def test_strict_threshold_needs_unanimity(self):
        """threshold=1.0 requires all answers identical to stop early."""
        lm = SequentialMockLM(["42", "42", "42", "24"] + ["42"] * 4)
        asc = AdaptiveSelfConsistency(threshold=1.0)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # Round 1: ["42","42"] → 100% → STOP (both agree)
        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_relaxed_threshold(self):
        """threshold=0.6: easier to trigger early stop."""
        lm = SequentialMockLM(["42", "24", "42", "42", "42", "42", "42", "42"])
        asc = AdaptiveSelfConsistency(threshold=0.6)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # Round 1: ["42","24"] → 50% < 60% → continue
        # Round 2: ["42","24","42","42"] → 75% ≥ 60% → stop
        assert len(result.responses) == 4


class TestAdaptiveSelfConsistencyProjection:
    """Test with custom projection functions."""

    @pytest.mark.asyncio
    async def test_regex_projection(self):
        """Early stopping works with regex-based answer extraction."""
        pattern = r"\\boxed\{([^}]+)\}"
        proj_func = create_regex_projection_function(pattern)

        lm = SequentialMockLM(
            [
                "Let me solve this. The answer is \\boxed{42}.",
                "Using algebra, we get \\boxed{42}.",
                "By computation \\boxed{24}.",
                "Therefore \\boxed{42}.",
            ]
            + ["\\boxed{99}"] * 12
        )
        asc = AdaptiveSelfConsistency(
            threshold=0.75,
            consistency_space_projection_func=proj_func,
        )

        result = await asc.ainfer(lm, "test", budget=16, return_response_only=False)

        # Round 1: 2 samples → projections ("42",), ("42",) → 100% → STOP
        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_default_projection_strips_whitespace(self):
        lm = SequentialMockLM(["  answer  ", "answer", "  answer  "] + ["other"] * 5)
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # Round 1: projections are "answer", "answer" → 100% → STOP
        assert len(result.responses) == 2


class TestAdaptiveSelfConsistencyInterface:
    """Test the algorithm interface and result types."""

    @pytest.mark.asyncio
    async def test_return_response_only_true(self):
        lm = SequentialMockLM(["42", "42"])
        asc = AdaptiveSelfConsistency()

        result = await asc.ainfer(lm, "test", budget=2, return_response_only=True)

        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert result["content"] == "42"

    @pytest.mark.asyncio
    async def test_return_response_only_false(self):
        lm = SequentialMockLM(["42", "42"])
        asc = AdaptiveSelfConsistency()

        result = await asc.ainfer(lm, "test", budget=2, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert result.the_one["content"] == "42"
        assert result.usage is not None

    def test_sync_infer(self):
        """Test the sync wrapper works."""
        lm = SequentialMockLM(["42", "42", "42", "24"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = asc.infer(lm, "test", budget=4, return_response_only=False)

        assert isinstance(result, SelfConsistencyResult)
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_with_chat_messages(self):
        lm = SequentialMockLM(["42", "42"])
        asc = AdaptiveSelfConsistency()

        chat_messages = ChatMessages("Solve this problem")
        result = await asc.ainfer(
            lm, chat_messages, budget=2, return_response_only=True
        )

        assert result["content"] == "42"

    @pytest.mark.asyncio
    async def test_response_counts_populated(self):
        lm = SequentialMockLM(["42", "24", "42", "42"])
        asc = AdaptiveSelfConsistency(threshold=0.75)

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # Should have stopped at 4 samples with 3x "42" and 1x "24"
        assert result.response_counts["42"] == 3
        assert result.response_counts["24"] == 1

    def test_inherits_from_self_consistency(self):
        from its_hub.core.algorithms.self_consistency import SelfConsistency

        asc = AdaptiveSelfConsistency()
        assert isinstance(asc, SelfConsistency)


class TestAdaptiveSelfConsistencyToolCalls:
    """Test tool-call voting with early stopping."""

    def _make_tool_response(self, name, args):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        }

    @pytest.mark.asyncio
    async def test_tool_vote_name_early_stop(self):
        """Early stop when tool name agrees across samples."""
        resp = self._make_tool_response("get_weather", '{"city": "NYC"}')
        resp_diff = self._make_tool_response("get_time", '{"tz": "UTC"}')
        lm = SequentialMockLM([resp, resp, resp_diff] + [resp] * 5)
        asc = AdaptiveSelfConsistency(threshold=0.75, tool_vote="tool_name")

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # Round 1: 2 samples, both "get_weather" → 100% → STOP
        assert len(result.responses) == 2
        assert result.the_one["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_vote_args_no_early_stop(self):
        """Different args prevent early stopping."""
        r1 = self._make_tool_response("search", '{"q": "cats"}')
        r2 = self._make_tool_response("search", '{"q": "dogs"}')
        r3 = self._make_tool_response("search", '{"q": "cats"}')
        r4 = self._make_tool_response("search", '{"q": "cats"}')
        lm = SequentialMockLM([r1, r2, r3, r4])
        asc = AdaptiveSelfConsistency(threshold=0.75, tool_vote="tool_args")

        result = await asc.ainfer(lm, "test", budget=4, return_response_only=False)

        # Round 1: ["cats", "dogs"] → 50% < 75% → continue
        # Round 2: 3x "cats", 1x "dogs" → 75% → STOP
        assert len(result.responses) == 4

    @pytest.mark.asyncio
    async def test_tool_vote_hierarchical_early_stop(self):
        """Hierarchical tool vote: name + args must agree."""
        resp = self._make_tool_response("calc", '{"expr": "2+2"}')
        lm = SequentialMockLM([resp] * 8)
        asc = AdaptiveSelfConsistency(threshold=0.75, tool_vote="tool_hierarchical")

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # All identical → stops at round 1
        assert len(result.responses) == 2

    @pytest.mark.asyncio
    async def test_exclude_args_in_tool_voting(self):
        """exclude_args filters non-semantic args before comparison."""
        r1 = self._make_tool_response("search", '{"q": "cats", "request_id": "abc"}')
        r2 = self._make_tool_response("search", '{"q": "cats", "request_id": "xyz"}')
        lm = SequentialMockLM([r1, r2] + [r1] * 6)
        asc = AdaptiveSelfConsistency(
            threshold=0.75,
            tool_vote="tool_args",
            exclude_args=["request_id"],
        )

        result = await asc.ainfer(lm, "test", budget=8, return_response_only=False)

        # After excluding request_id, both have args {"q": "cats"} → 100% → STOP
        assert len(result.responses) == 2
