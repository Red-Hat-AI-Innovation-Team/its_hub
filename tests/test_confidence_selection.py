"""Tests for ConfidenceSelection algorithm."""


import numpy as np
import pytest

from its_hub.core.algorithms.confidence_selection import (
    ConfidenceSelection,
    ConfidenceSelectionResult,
    adaptive_tail_window,
    compute_token_certainties,
    compute_token_entropies,
    select_by_tail_certainty,
    select_by_tail_entropy,
    tail_scores,
    trim_length_outliers,
)
from its_hub.core.orchestrator import LMOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logprobs_content(entropies: list[float]) -> list[dict]:
    """Build OpenAI-format logprobs content from target entropy values.

    Creates top_logprobs with two tokens whose probabilities yield
    approximately the desired per-token entropy.
    """
    content = []
    for h in entropies:
        if h <= 0:
            content.append(
                {"top_logprobs": [{"token": "a", "logprob": 0.0}]}
            )
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


class LogprobsMockLM:
    """Mock LM that returns responses with logprobs data."""

    def __init__(self, responses_with_logprobs: list[dict]):
        self.responses = responses_with_logprobs
        self.call_count = 0

    async def agenerate_single(self, messages, **kwargs):
        idx = self.call_count % len(self.responses)
        self.call_count += 1
        return self.responses[idx]


# ---------------------------------------------------------------------------
# Unit tests: compute_token_entropies
# ---------------------------------------------------------------------------


class TestComputeTokenEntropies:
    def test_single_token_zero_entropy(self):
        content = [{"top_logprobs": [{"token": "x", "logprob": 0.0}]}]
        result = compute_token_entropies(content)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_two_tokens_uniform(self):
        p = 0.5
        content = [
            {
                "top_logprobs": [
                    {"token": "a", "logprob": float(np.log(p))},
                    {"token": "b", "logprob": float(np.log(p))},
                ]
            }
        ]
        result = compute_token_entropies(content)
        expected = -2 * p * np.log(p)
        assert result[0] == pytest.approx(expected, rel=1e-4)

    def test_multiple_tokens(self):
        content = [
            {"top_logprobs": [{"token": "a", "logprob": -0.1}]},
            {"top_logprobs": [{"token": "b", "logprob": -0.5}]},
            {"top_logprobs": [{"token": "c", "logprob": -1.0}]},
        ]
        result = compute_token_entropies(content)
        assert len(result) == 3

    def test_empty_top_logprobs(self):
        content = [{"top_logprobs": []}]
        result = compute_token_entropies(content)
        assert result[0] == 0.0

    def test_empty_input(self):
        assert compute_token_entropies([]) == []


# ---------------------------------------------------------------------------
# Unit tests: trim_length_outliers
# ---------------------------------------------------------------------------


class TestTrimLengthOutliers:
    def test_small_sample_returns_all(self):
        lengths = [10, 20, 30, 40, 50]
        assert trim_length_outliers(lengths) == list(range(5))

    def test_exactly_16_applies_trimming(self):
        lengths = [1] + [100] * 14 + [10000]
        result = trim_length_outliers(lengths)
        assert 0 not in result
        assert 15 not in result

    def test_uniform_lengths(self):
        lengths = [100] * 20
        result = trim_length_outliers(lengths)
        assert result == list(range(20))

    def test_aggressive_trim(self):
        lengths = [1] * 8 + [1000] * 8
        result = trim_length_outliers(lengths, trim_pct=0.45)
        assert len(result) >= 1
        assert all(0 <= i < 16 for i in result)


# ---------------------------------------------------------------------------
# Unit tests: adaptive_tail_window
# ---------------------------------------------------------------------------


class TestAdaptiveTailWindow:
    def test_basic(self):
        ept = [list(np.random.rand(200)) for _ in range(4)]
        included = list(range(4))
        tail = adaptive_tail_window(ept, included, tail_min=10, tail_max=500)
        assert 10 <= tail <= 500

    def test_clamps_to_min(self):
        ept = [[10.0] * 10 for _ in range(4)]
        included = list(range(4))
        tail = adaptive_tail_window(ept, included, tail_min=64, tail_max=2048)
        assert tail == 64

    def test_clamps_to_max(self):
        ept = [[0.001] * 1_000_000 for _ in range(4)]
        included = list(range(4))
        tail = adaptive_tail_window(ept, included, tail_min=64, tail_max=2048)
        assert tail == 2048


# ---------------------------------------------------------------------------
# Unit tests: tail_scores
# ---------------------------------------------------------------------------


class TestTailScores:
    def test_excluded_gets_inf(self):
        ept = [[0.1] * 100, [0.2] * 100, [0.3] * 100]
        scores = tail_scores(ept, included=[0, 2], tail=50)
        assert scores[1] == float("inf")
        assert scores[0] < float("inf")
        assert scores[2] < float("inf")

    def test_empty_gets_inf(self):
        ept = [[], [0.1] * 100]
        scores = tail_scores(ept, included=[0, 1], tail=50)
        assert scores[0] == float("inf")

    def test_median_vs_mean(self):
        ept = [[0.1] * 50 + [10.0] + [0.1] * 49]
        scores_median = tail_scores(ept, included=[0], tail=100, agg="median")
        scores_mean = tail_scores(ept, included=[0], tail=100, agg="mean")
        assert scores_median[0] < scores_mean[0]


# ---------------------------------------------------------------------------
# Unit tests: select_by_tail_entropy
# ---------------------------------------------------------------------------


class TestSelectByTailEntropy:
    def test_selects_lowest_entropy(self):
        low = [0.01] * 200
        mid = [0.5] * 200
        high = [1.0] * 200
        selected, scores, _tail = select_by_tail_entropy(
            [high, low, mid], tail_min=10, tail_max=500
        )
        assert selected == 1
        assert scores[1] < scores[0]
        assert scores[1] < scores[2]

    def test_single_response(self):
        selected, scores, _tail = select_by_tail_entropy(
            [[0.5] * 100], tail_min=10, tail_max=500
        )
        assert selected == 0
        assert len(scores) == 1

    def test_returns_tail_window(self):
        ept = [[0.5] * 200 for _ in range(4)]
        _, _, tail = select_by_tail_entropy(ept, tail_min=10, tail_max=500)
        assert 10 <= tail <= 500

    def test_skips_empty_before_trimming(self):
        ept = [[]] * 15 + [[0.1] * 100] + [[]] * 4
        selected, scores, _tail = select_by_tail_entropy(ept, tail_min=10, tail_max=500)
        assert selected == 15
        assert scores[15] < float("inf")
        assert all(s == float("inf") for i, s in enumerate(scores) if i != 15)

    def test_all_empty_raises(self):
        with pytest.raises(ValueError, match="No candidates have entropy data"):
            select_by_tail_entropy([[], [], []])


# ---------------------------------------------------------------------------
# Integration tests: ConfidenceSelection algorithm
# ---------------------------------------------------------------------------


class TestConfidenceSelection:
    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="agg"):
            ConfidenceSelection(agg="invalid")

        with pytest.raises(ValueError, match="top_logprobs"):
            ConfidenceSelection(top_logprobs=0)

        with pytest.raises(ValueError, match="top_logprobs"):
            ConfidenceSelection(top_logprobs=21)

    def test_infer_selects_most_confident(self):
        low_entropy_logprobs = _make_logprobs_content([0.01] * 200)
        high_entropy_logprobs = _make_logprobs_content([1.0] * 200)

        responses = [
            {
                "role": "assistant",
                "content": "wrong answer",
                "_logprobs": {"content": high_entropy_logprobs},
            },
            {
                "role": "assistant",
                "content": "correct answer",
                "_logprobs": {"content": low_entropy_logprobs},
            },
            {
                "role": "assistant",
                "content": "another wrong",
                "_logprobs": {"content": high_entropy_logprobs},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test prompt", budget=3, return_response_only=False)

        assert isinstance(result, ConfidenceSelectionResult)
        assert result.the_one["content"] == "correct answer"
        assert result.scores[result.selected_index] == min(result.scores)
        assert result.tail_window > 0

    def test_infer_response_only(self):
        logprobs = _make_logprobs_content([0.1] * 100)
        responses = [
            {
                "role": "assistant",
                "content": f"response {i}",
                "_logprobs": {"content": logprobs},
            }
            for i in range(3)
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test prompt", budget=3, return_response_only=True)

        assert isinstance(result, dict)
        assert "content" in result

    def test_infer_handles_missing_logprobs(self):
        good = _make_logprobs_content([0.1] * 100)
        responses = [
            {"role": "assistant", "content": "no logprobs"},
            {
                "role": "assistant",
                "content": "has logprobs",
                "_logprobs": {"content": good},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test prompt", budget=2, return_response_only=False)

        assert result.the_one["content"] == "has logprobs"
        assert float("inf") in result.scores

    def test_infer_all_missing_logprobs_raises(self):
        responses = [
            {"role": "assistant", "content": "no logprobs 1"},
            {"role": "assistant", "content": "no logprobs 2"},
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        with pytest.raises(ValueError, match="No candidates have entropy data"):
            algo.infer(lm, "test prompt", budget=2)

    def test_infer_16_responses_with_missing_logprobs(self):
        good = _make_logprobs_content([0.1] * 100)
        responses = [{"role": "assistant", "content": "no logprobs"}] * 15 + [
            {
                "role": "assistant",
                "content": "has logprobs",
                "_logprobs": {"content": good},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test prompt", budget=16, return_response_only=False)

        assert result.the_one["content"] == "has logprobs"
        non_inf = [s for s in result.scores if s != float("inf")]
        assert len(non_inf) >= 1

    def test_infer_with_messages_input(self):
        logprobs = _make_logprobs_content([0.1] * 100)
        responses = [
            {
                "role": "assistant",
                "content": "answer",
                "_logprobs": {"content": logprobs},
            }
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        messages = [{"role": "user", "content": "test"}]
        result = algo.infer(lm, messages, budget=1, return_response_only=True)
        assert result["content"] == "answer"

    def test_default_orchestrator(self):
        logprobs = _make_logprobs_content([0.1] * 100)
        responses = [
            {
                "role": "assistant",
                "content": "answer",
                "_logprobs": {"content": logprobs},
            }
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(tail_min=10, tail_max=500)
        result = algo.infer(lm, "test", budget=1, return_response_only=False)
        assert result.the_one["content"] == "answer"

    def test_usage_tracking(self):
        logprobs = _make_logprobs_content([0.1] * 100)
        responses = [
            {
                "role": "assistant",
                "content": f"r{i}",
                "_logprobs": {"content": logprobs},
            }
            for i in range(3)
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test", budget=3, return_response_only=False)
        assert result.usage is not None


# ---------------------------------------------------------------------------
# Unit tests: compute_token_certainties
# ---------------------------------------------------------------------------


def _make_certainty_logprobs(logprobs_per_token: list[list[float]]) -> list[dict]:
    """Build OpenAI-format logprobs content from raw logprob lists."""
    content = []
    for lps in logprobs_per_token:
        content.append(
            {
                "top_logprobs": [
                    {"token": f"t{j}", "logprob": lp} for j, lp in enumerate(lps)
                ]
            }
        )
    return content


class TestComputeTokenCertainties:
    def test_uniform_gives_zero(self):
        lp = float(np.log(0.5))
        content = _make_certainty_logprobs([[lp, lp]])
        result = compute_token_certainties(content)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_peaked_gives_positive(self):
        content = _make_certainty_logprobs([[np.log(0.99), np.log(0.01)]])
        result = compute_token_certainties(content)
        assert result[0] > 0

    def test_more_peaked_gives_higher_certainty(self):
        mild = _make_certainty_logprobs([[np.log(0.7), np.log(0.3)]])
        strong = _make_certainty_logprobs([[np.log(0.99), np.log(0.01)]])
        assert compute_token_certainties(strong)[0] > compute_token_certainties(mild)[0]

    def test_single_token(self):
        content = _make_certainty_logprobs([[0.0]])
        result = compute_token_certainties(content)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_empty_top_logprobs(self):
        content = [{"top_logprobs": []}]
        result = compute_token_certainties(content)
        assert result[0] == 0.0

    def test_empty_input(self):
        assert compute_token_certainties([]) == []

    def test_multiple_tokens(self):
        content = _make_certainty_logprobs(
            [[np.log(0.9), np.log(0.1)]] * 5
        )
        result = compute_token_certainties(content)
        assert len(result) == 5
        assert all(c > 0 for c in result)

    def test_explicit_vocab_size(self):
        content = _make_certainty_logprobs([[np.log(0.9), np.log(0.1)]])
        default_v = compute_token_certainties(content)
        large_v = compute_token_certainties(content, vocab_size=32000)
        assert default_v[0] != pytest.approx(large_v[0], abs=1e-3)


# ---------------------------------------------------------------------------
# Unit tests: select_by_tail_certainty
# ---------------------------------------------------------------------------


class TestSelectByTailCertainty:
    def test_selects_highest_certainty(self):
        high = [2.0] * 200
        mid = [0.5] * 200
        low = [0.01] * 200
        selected, scores, _tail = select_by_tail_certainty(
            [low, high, mid], tail_min=10, tail_max=500
        )
        assert selected == 1
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_single_response(self):
        selected, _scores, _tail = select_by_tail_certainty(
            [[1.0] * 100], tail_min=10, tail_max=500
        )
        assert selected == 0

    def test_returns_tail_window(self):
        cpt = [[0.5] * 200 for _ in range(4)]
        _, _, tail = select_by_tail_certainty(cpt, tail_min=10, tail_max=500)
        assert 10 <= tail <= 500

    def test_excluded_gets_neg_inf(self):
        cpt = [[1.0] * 200, [], [0.5] * 200]
        _, scores, _ = select_by_tail_certainty(cpt, tail_min=10, tail_max=500)
        assert scores[1] == float("-inf")

    def test_all_empty_raises(self):
        with pytest.raises(ValueError, match="No candidates have certainty data"):
            select_by_tail_certainty([[], [], []])


# ---------------------------------------------------------------------------
# Integration tests: ConfidenceSelection with metric="certainty"
# ---------------------------------------------------------------------------


class TestConfidenceSelectionCertainty:
    def test_constructor_validates_metric(self):
        with pytest.raises(ValueError, match="metric"):
            ConfidenceSelection(metric="invalid")

    def test_certainty_selects_most_peaked(self):
        peaked_logprobs = _make_certainty_logprobs(
            [[np.log(0.99), np.log(0.01)]] * 200
        )
        flat_logprobs = _make_certainty_logprobs(
            [[np.log(0.55), np.log(0.45)]] * 200
        )

        responses = [
            {
                "role": "assistant",
                "content": "flat answer",
                "_logprobs": {"content": flat_logprobs},
            },
            {
                "role": "assistant",
                "content": "peaked answer",
                "_logprobs": {"content": peaked_logprobs},
            },
            {
                "role": "assistant",
                "content": "another flat",
                "_logprobs": {"content": flat_logprobs},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            metric="certainty",
            tail_min=10,
            tail_max=500,
            orchestrator=LMOrchestrator(),
        )
        result = algo.infer(lm, "test prompt", budget=3, return_response_only=False)

        assert isinstance(result, ConfidenceSelectionResult)
        assert result.the_one["content"] == "peaked answer"
        assert result.scores[result.selected_index] == max(result.scores)

    def test_certainty_response_only(self):
        logprobs = _make_certainty_logprobs([[np.log(0.9), np.log(0.1)]] * 100)
        responses = [
            {
                "role": "assistant",
                "content": f"r{i}",
                "_logprobs": {"content": logprobs},
            }
            for i in range(3)
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            metric="certainty",
            tail_min=10,
            tail_max=500,
            orchestrator=LMOrchestrator(),
        )
        result = algo.infer(lm, "test prompt", budget=3, return_response_only=True)
        assert isinstance(result, dict)
        assert "content" in result

    def test_certainty_handles_missing_logprobs(self):
        good = _make_certainty_logprobs([[np.log(0.9), np.log(0.1)]] * 100)
        responses = [
            {"role": "assistant", "content": "no logprobs"},
            {
                "role": "assistant",
                "content": "has logprobs",
                "_logprobs": {"content": good},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            metric="certainty",
            tail_min=10,
            tail_max=500,
            orchestrator=LMOrchestrator(),
        )
        result = algo.infer(lm, "test prompt", budget=2, return_response_only=False)
        assert result.the_one["content"] == "has logprobs"
        assert float("-inf") in result.scores

    def test_certainty_with_vocab_size(self):
        logprobs = _make_certainty_logprobs([[np.log(0.9), np.log(0.1)]] * 100)
        responses = [
            {
                "role": "assistant",
                "content": "answer",
                "_logprobs": {"content": logprobs},
            }
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            metric="certainty",
            vocab_size=32000,
            tail_min=10,
            tail_max=500,
            orchestrator=LMOrchestrator(),
        )
        result = algo.infer(lm, "test", budget=1, return_response_only=False)
        assert result.the_one["content"] == "answer"

    def test_entropy_default_unchanged(self):
        low_entropy_logprobs = _make_logprobs_content([0.01] * 200)
        high_entropy_logprobs = _make_logprobs_content([1.0] * 200)
        responses = [
            {
                "role": "assistant",
                "content": "high entropy",
                "_logprobs": {"content": high_entropy_logprobs},
            },
            {
                "role": "assistant",
                "content": "low entropy",
                "_logprobs": {"content": low_entropy_logprobs},
            },
        ]

        lm = LogprobsMockLM(responses)
        algo = ConfidenceSelection(
            tail_min=10, tail_max=500, orchestrator=LMOrchestrator()
        )
        result = algo.infer(lm, "test", budget=2, return_response_only=False)
        assert result.the_one["content"] == "low entropy"
        assert algo.metric == "entropy"
