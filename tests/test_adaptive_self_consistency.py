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

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self._lock = threading.Lock()

    async def agenerate_single(self, messages, **kwargs):
        with self._lock:
            idx = self.call_count % len(self.responses)
            self.call_count += 1
        return {"role": "assistant", "content": self.responses[idx]}


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
