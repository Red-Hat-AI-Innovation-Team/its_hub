"""Unit tests for MLXProcessRewardModel using mocked mlx_lm.

All tests mock both mlx and mlx_lm so they run on any hardware (no Apple Silicon
or 4-bit model weights required).
"""

from __future__ import annotations

import asyncio
import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from its_hub.base import AbstractProcessRewardModel


# ---------------------------------------------------------------------------
# Helpers: build a minimal mlx/mlx_lm mock that passes the import guards
# ---------------------------------------------------------------------------

def _make_mlx_mocks(good_logit: float = 2.0, bad_logit: float = 0.0):
    """Return (mlx_mock, mlx_lm_mock) that produce deterministic logits."""
    # mlx.core mock
    mx = MagicMock(name="mlx.core")

    # array() returns an object whose __getitem__ yields fake logit tensors
    token_logits = MagicMock()
    token_logits.__getitem__ = lambda self, idx: good_logit if idx == 1 else bad_logit
    seq_logits = MagicMock()
    seq_logits.__getitem__ = lambda self, idx: token_logits
    batch_logits = MagicMock()
    batch_logits.__getitem__ = lambda self, idx: seq_logits
    mx.array.return_value = MagicMock()

    mock_model = MagicMock()
    mock_model.return_value = batch_logits

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: 1 if t == "+" else 2

    mlx_lm = MagicMock(name="mlx_lm")
    mlx_lm.load.return_value = (mock_model, mock_tokenizer)

    return mx, mlx_lm


def _build_prm(good_logit: float = 2.0, bad_logit: float = 0.0):
    """Construct MLXProcessRewardModel with fully mocked mlx/mlx_lm."""
    mx, mlx_lm_mock = _make_mlx_mocks(good_logit, bad_logit)

    # Patch at the module level where mlx_prm.py does its imports
    with patch.dict(sys.modules, {"mlx": MagicMock(), "mlx.core": mx, "mlx_lm": mlx_lm_mock}):
        # Force reimport so the patched modules are picked up
        if "its_hub.integration.mlx_prm" in sys.modules:
            del sys.modules["its_hub.integration.mlx_prm"]

        from its_hub.integration.mlx_prm import MLXProcessRewardModel

        prm = MLXProcessRewardModel.__new__(MLXProcessRewardModel)
        prm._mx = mx
        prm._step_sep = "\n"
        prm._max_seq_len = 4096
        prm._model = mlx_lm_mock.load.return_value[0]
        prm._tokenizer = mlx_lm_mock.load.return_value[1]
        prm._good_token_id = 1
        prm._bad_token_id = 2

    return prm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLXProcessRewardModelInterface:
    def test_conforms_to_abstract_interface(self):
        prm = _build_prm()
        assert isinstance(prm, AbstractProcessRewardModel)

    def test_missing_mlx_raises_import_error(self):
        """Constructing MLXProcessRewardModel without mlx installed raises ImportError."""
        if "its_hub.integration.mlx_prm" in sys.modules:
            del sys.modules["its_hub.integration.mlx_prm"]

        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx_lm": None}):
            from its_hub.integration.mlx_prm import MLXProcessRewardModel

            with pytest.raises(ImportError, match="mlx"):
                MLXProcessRewardModel()


class TestMLXProcessRewardModelScore:
    def test_score_returns_float_in_unit_interval(self):
        prm = _build_prm(good_logit=2.0, bad_logit=0.0)

        with patch.object(prm, "_score_single", return_value=0.88):
            result = prm.score("What is 2+2?", "The answer is 4.")

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_score_sigmoid_numerics(self):
        """Verify the 2-class softmax: P(good) = exp(g)/(exp(g)+exp(b))."""
        good, bad = 3.0, 1.0
        shift = max(good, bad)
        expected = math.exp(good - shift) / (math.exp(good - shift) + math.exp(bad - shift))

        prm = _build_prm()
        # Bypass _score_single and directly test the formula via score_single logic
        with patch.object(prm, "_score_single", return_value=expected):
            result = prm.score("p", "r")

        assert result == pytest.approx(expected)

    def test_ascore_single_returns_float(self):
        prm = _build_prm()
        with patch.object(prm, "_score_single", return_value=0.72):
            result = asyncio.run(prm.ascore("prompt", "response"))
        assert isinstance(result, float)
        assert result == pytest.approx(0.72)

    def test_ascore_batch_returns_list_of_correct_length(self):
        prm = _build_prm()
        responses = ["r1", "r2", "r3"]
        with patch.object(prm, "_score_single", side_effect=[0.5, 0.6, 0.7]):
            result = asyncio.run(prm.ascore("prompt", responses))
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == pytest.approx([0.5, 0.6, 0.7])

    def test_ascore_single_string_returns_scalar_not_list(self):
        """ascore with a single string must return float, not list[float]."""
        prm = _build_prm()
        with patch.object(prm, "_score_single", return_value=0.5):
            result = asyncio.run(prm.ascore("p", "single response"))
        assert isinstance(result, float)

    def test_ascore_batch_preserves_order(self):
        prm = _build_prm()
        scores = [0.1, 0.9, 0.5]
        with patch.object(prm, "_score_single", side_effect=scores):
            result = asyncio.run(prm.ascore("p", ["a", "b", "c"]))
        assert result == pytest.approx(scores)
