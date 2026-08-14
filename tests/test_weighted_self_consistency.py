"""Tests for WeightedSelfConsistency algorithm."""

import numpy as np
import pytest

from its_hub.core.algorithms.weighted_self_consistency import (
    WeightedSelfConsistency,
    WeightedSelfConsistencyResult,
    _scores_to_weights,
    _select_weighted_majority,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logprobs_content(entropies: list[float]) -> list[dict]:
    """Build OpenAI-format logprobs content from target entropy values."""
    content = []
    for h in entropies:
        if h <= 0:
            content.append({"top_logprobs": [{"token": "a", "logprob": 0.0}]})
        else:
            p = min(max(np.exp(-h), 0.01), 0.99)
            content.append(
                {
                    "top_logprobs": [
                        {"token": "a", "logprob": float(np.log(p))},
                        {"token": "b", "logprob": float(np.log(1 - p))},
                    ]
                }
            )
    return content


def _make_uniform_logprobs_content(n_tokens: int, k: int = 20) -> list[dict]:
    """Build logprobs content where each position has k near-uniform tokens.

    Produces high Shannon entropy ≈ log(k).
    """
    lp = float(np.log(1.0 / k))
    entry = {"top_logprobs": [{"token": f"t{j}", "logprob": lp} for j in range(k)]}
    return [entry] * n_tokens


def _make_response(content: str, entropies: list[float] | None = None) -> dict:
    """Build a mock response dict, optionally with logprobs."""
    resp: dict = {"role": "assistant", "content": content}
    if entropies is not None:
        resp["_logprobs"] = {"content": _make_logprobs_content(entropies)}
    return resp


def _make_response_raw(content: str, logprobs_content: list[dict]) -> dict:
    """Build a mock response with explicit logprobs content."""
    return {
        "role": "assistant",
        "content": content,
        "_logprobs": {"content": logprobs_content},
    }


class LogprobsMockLM:
    """Mock LM that returns pre-built responses in order."""

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0

    async def agenerate_single(self, messages, **kwargs):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


# ---------------------------------------------------------------------------
# Unit tests: _scores_to_weights
# ---------------------------------------------------------------------------


class TestScoresToWeights:
    def test_entropy_lower_is_heavier(self):
        scores = [0.1, 0.5, 1.0]
        weights = _scores_to_weights(scores, "entropy")
        assert weights[0] > weights[1] > weights[2]

    def test_certainty_higher_is_heavier(self):
        scores = [0.1, 0.5, 1.0]
        weights = _scores_to_weights(scores, "certainty")
        assert weights[0] < weights[1] < weights[2]

    def test_inf_entropy_gives_zero_weight(self):
        scores = [0.5, float("inf")]
        weights = _scores_to_weights(scores, "entropy")
        assert weights[1] == 0.0
        assert weights[0] > 0.0

    def test_neg_inf_certainty_gives_zero_weight(self):
        scores = [0.5, float("-inf")]
        weights = _scores_to_weights(scores, "certainty")
        assert weights[1] == 0.0
        assert weights[0] > 0.0

    def test_equal_scores_give_equal_weights(self):
        scores = [0.3, 0.3, 0.3]
        weights = _scores_to_weights(scores, "entropy")
        assert weights[0] == pytest.approx(weights[1])
        assert weights[1] == pytest.approx(weights[2])


# ---------------------------------------------------------------------------
# Unit tests: _select_weighted_majority
# ---------------------------------------------------------------------------


class TestSelectWeightedMajority:
    def test_confident_minority_wins(self):
        responses = [
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "99"},
            {"role": "assistant", "content": "99"},
            {"role": "assistant", "content": "99"},
        ]
        # "42" candidates are very confident, "99" candidates are not
        weights = [0.9, 0.9, 0.1, 0.1, 0.1]
        scores = [0.1, 0.15, 2.0, 2.5, 3.0]

        selected, group_weights = _select_weighted_majority(
            responses, weights, scores, "entropy", str.strip
        )
        assert responses[selected]["content"] == "42"
        assert group_weights["42"] > group_weights["99"]

    def test_consistent_majority_wins_when_weights_close(self):
        responses = [
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "99"},
            {"role": "assistant", "content": "99"},
            {"role": "assistant", "content": "99"},
        ]
        weights = [0.5, 0.4, 0.4, 0.4]
        scores = [0.5, 0.6, 0.6, 0.6]

        selected, group_weights = _select_weighted_majority(
            responses, weights, scores, "entropy", str.strip
        )
        assert responses[selected]["content"] == "99"
        assert group_weights["99"] > group_weights["42"]

    def test_best_confidence_within_winning_group(self):
        responses = [
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "42"},
        ]
        weights = [0.5, 0.8, 0.6]
        scores = [0.5, 0.1, 0.3]  # index 1 has lowest entropy

        selected, _ = _select_weighted_majority(
            responses, weights, scores, "entropy", str.strip
        )
        assert selected == 1

    def test_certainty_metric_picks_argmax(self):
        responses = [
            {"role": "assistant", "content": "A"},
            {"role": "assistant", "content": "A"},
            {"role": "assistant", "content": "A"},
        ]
        weights = [0.5, 0.8, 0.6]
        scores = [0.5, 2.0, 1.0]  # index 1 has highest certainty

        selected, _ = _select_weighted_majority(
            responses, weights, scores, "certainty", str.strip
        )
        assert selected == 1


# ---------------------------------------------------------------------------
# Integration tests: full ainfer() pipeline
# ---------------------------------------------------------------------------


class TestWeightedSelfConsistencyAinfer:
    @pytest.mark.asyncio
    async def test_confident_majority_wins(self):
        n_tokens = 100
        responses = [
            _make_response("42", [0.01] * n_tokens),
            _make_response("42", [0.02] * n_tokens),
            _make_response("42", [0.01] * n_tokens),
            _make_response("99", [1.0] * n_tokens),
            _make_response("99", [1.2] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(metric="entropy")

        result = await algo.ainfer(lm, "test", budget=5, return_response_only=False)

        assert isinstance(result, WeightedSelfConsistencyResult)
        assert result.the_one["content"] == "42"

    @pytest.mark.asyncio
    async def test_confident_minority_beats_uncertain_majority(self):
        n_tokens = 100
        confident_lp = _make_logprobs_content([0.01] * n_tokens)
        uncertain_lp = _make_uniform_logprobs_content(n_tokens, k=20)
        responses = [
            # 2 very confident "42" candidates (entropy ≈ 0.06)
            _make_response_raw("42", confident_lp),
            _make_response_raw("42", confident_lp),
            # 3 very uncertain "99" candidates (entropy ≈ log(20) ≈ 3.0)
            _make_response_raw("99", uncertain_lp),
            _make_response_raw("99", uncertain_lp),
            _make_response_raw("99", uncertain_lp),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(metric="entropy")

        result = await algo.ainfer(lm, "test", budget=5, return_response_only=False)

        assert result.the_one["content"] == "42"
        assert result.group_weights["42"] > result.group_weights["99"]

    @pytest.mark.asyncio
    async def test_uniform_confidence_degenerates_to_majority(self):
        n_tokens = 100
        entropy_val = 0.5
        responses = [
            _make_response("42", [entropy_val] * n_tokens),
            _make_response("99", [entropy_val] * n_tokens),
            _make_response("99", [entropy_val] * n_tokens),
            _make_response("99", [entropy_val] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(metric="entropy")

        result = await algo.ainfer(lm, "test", budget=4, return_response_only=False)

        assert result.the_one["content"] == "99"

    @pytest.mark.asyncio
    async def test_certainty_metric(self):
        n_tokens = 100
        responses = [
            _make_response("A", [0.01] * n_tokens),
            _make_response("A", [0.02] * n_tokens),
            _make_response("B", [0.5] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(metric="certainty")

        result = await algo.ainfer(lm, "test", budget=3, return_response_only=False)

        assert isinstance(result, WeightedSelfConsistencyResult)
        assert result.the_one["content"] in ("A", "B")

    @pytest.mark.asyncio
    async def test_return_response_only(self):
        n_tokens = 100
        responses = [
            _make_response("answer", [0.1] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency()

        result = await algo.ainfer(lm, "test", budget=1, return_response_only=True)

        assert isinstance(result, dict)
        assert result["content"] == "answer"

    @pytest.mark.asyncio
    async def test_custom_projection_func(self):
        n_tokens = 100

        def extract_last_word(s: str) -> str:
            return s.strip().split()[-1]

        responses = [
            _make_response("the answer is 42", [0.01] * n_tokens),
            _make_response("result: 42", [0.02] * n_tokens),
            _make_response("I think 99", [1.0] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(
            consistency_space_projection_func=extract_last_word,
            metric="entropy",
        )

        result = await algo.ainfer(lm, "test", budget=3, return_response_only=False)

        assert result.the_one["content"] in ("the answer is 42", "result: 42")

    @pytest.mark.asyncio
    async def test_missing_logprobs_excluded(self):
        n_tokens = 100
        responses = [
            _make_response("42", [0.01] * n_tokens),
            _make_response("42", None),  # no logprobs
            _make_response("99", [0.5] * n_tokens),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency(metric="entropy")

        result = await algo.ainfer(lm, "test", budget=3, return_response_only=False)

        assert isinstance(result, WeightedSelfConsistencyResult)
        no_lp_idx = next(
            i for i, r in enumerate(result.responses) if "_logprobs" not in r
        )
        assert result.weights[no_lp_idx] == 0.0

    @pytest.mark.asyncio
    async def test_no_logprobs_at_all_raises(self):
        responses = [
            _make_response("42", None),
            _make_response("99", None),
        ]
        lm = LogprobsMockLM(responses)
        algo = WeightedSelfConsistency()

        with pytest.raises(ValueError, match="No candidates have logprobs"):
            await algo.ainfer(lm, "test", budget=2)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestWeightedSelfConsistencyValidation:
    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            WeightedSelfConsistency(metric="invalid")

    def test_invalid_agg(self):
        with pytest.raises(ValueError, match="agg"):
            WeightedSelfConsistency(agg="invalid")

    def test_invalid_top_logprobs(self):
        with pytest.raises(ValueError, match="top_logprobs"):
            WeightedSelfConsistency(top_logprobs=0)
        with pytest.raises(ValueError, match="top_logprobs"):
            WeightedSelfConsistency(top_logprobs=21)
