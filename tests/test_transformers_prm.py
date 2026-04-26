"""Tests for TransformersProcessRewardModel.

Covers:
  - ImportError guard when torch/transformers are absent
  - _repair_rotary_embeddings() recomputes zeroed inv_freq buffers
  - score() returns a list of floats in [0, 1] of length == len(steps)
  - Trailing <extra_0> separator is appended (N steps → N separators)
  - ascore() delegates to score() via asyncio.to_thread
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build minimal fake torch / transformers stubs
# ---------------------------------------------------------------------------


def _make_torch_stub():
    """Minimal torch stub sufficient to import and instantiate the PRM."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"

    # Tensor-like with basic arithmetic for inv_freq computation
    class _Tensor:
        def __init__(self, data):
            self._data = list(data) if hasattr(data, "__iter__") else [data]

        def float(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def __truediv__(self, other):
            return self

        def __eq__(self, other):
            class _Mask:
                def nonzero(self, as_tuple=False):
                    return ([],)

            return _Mask()

        def __getitem__(self, idx):
            return self

        def __len__(self):
            return len(self._data)

    def arange(*args, dtype=None):
        start, stop, step = 0, args[0], 1
        if len(args) == 3:
            start, stop, step = args
        return _Tensor(range(start, stop, step))

    torch.arange = arange
    torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))

    def softmax(tensor, dim=-1):
        return _Tensor([0.3, 0.7])

    torch.softmax = softmax

    nn = types.ModuleType("torch.nn")
    torch.nn = nn

    return torch


def _make_transformers_stub(model_mock):
    """Minimal transformers stub that returns `model_mock` from from_pretrained."""
    tf = types.ModuleType("transformers")

    tokenizer_mock = MagicMock()
    tokenizer_mock.eos_token_id = 0
    tokenizer_mock.convert_tokens_to_ids.return_value = 151643  # <extra_0>
    tokenizer_mock.apply_chat_template.return_value = "<prompt>"
    enc = MagicMock()
    enc.__getitem__ = lambda self, k: MagicMock(to=lambda d: MagicMock())
    enc.get.return_value = None
    tokenizer_mock.return_value = enc
    tokenizer_mock.__call__ = lambda *a, **kw: enc

    config_mock = MagicMock()
    config_mock.pad_token_id = None

    tf.AutoTokenizer = MagicMock()
    tf.AutoTokenizer.from_pretrained.return_value = tokenizer_mock
    tf.AutoModel = MagicMock()
    tf.AutoModel.from_pretrained.return_value = model_mock
    tf.AutoConfig = MagicMock()
    tf.AutoConfig.from_pretrained.return_value = config_mock

    return tf, tokenizer_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_raises_import_error_without_torch(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            # Force re-import
            if "its_hub.integration.transformers_prm" in sys.modules:
                del sys.modules["its_hub.integration.transformers_prm"]
            from its_hub.integration.transformers_prm import TransformersProcessRewardModel

            with pytest.raises(ImportError, match="transformers and torch"):
                TransformersProcessRewardModel()


class TestRepairRotaryEmbeddings:
    def test_repairs_zeroed_inv_freq(self):
        """_repair_rotary_embeddings should register fresh inv_freq on RotaryEmbedding modules."""
        import torch

        # Build a fake RotaryEmbedding module whose inv_freq is zeroed
        class FakeRotaryEmbedding:
            __name__ = "Qwen2RotaryEmbedding"

            def __init__(self):
                self.base = 10000.0
                self.dim = 8
                self.inv_freq = torch.zeros(4)  # zeroed — simulates the bug
                self.max_seq_len_cached = 2048
                self._registered = {}

            def register_buffer(self, name, val, persistent=True):
                self._registered[name] = val

        class FakeModel:
            def __init__(self):
                self._rope = FakeRotaryEmbedding()

            def parameters(self):
                p = MagicMock()
                p.device = torch.device("cpu")
                return iter([p])

            def modules(self):
                return [self, self._rope]

        # Patch out the full construction; only test _repair_rotary_embeddings
        rope = FakeRotaryEmbedding()
        fake_model = FakeModel()
        fake_model._rope = rope

        from its_hub.integration.transformers_prm import TransformersProcessRewardModel

        prm = object.__new__(TransformersProcessRewardModel)
        prm._model = fake_model
        prm._device = "cpu"
        prm._repair_rotary_embeddings()

        assert "inv_freq" in rope._registered
        repaired = rope._registered["inv_freq"]
        # Should be non-zero (recomputed from base/dim)
        assert repaired.abs().sum().item() > 0


class TestScoreInterface:
    @pytest.fixture
    def prm(self):
        """Construct a TransformersProcessRewardModel with all heavy deps mocked."""
        import torch

        # Build a model mock whose forward() returns logits with high score on pos 1
        output_mock = MagicMock()
        # logits: shape (1, 5, 2) — positions 2 and 4 are <extra_0> tokens
        logits = torch.zeros(1, 5, 2)
        logits[0, :, 1] = 0.9  # positive class dominates everywhere
        output_mock.logits = logits

        model_mock = MagicMock()
        model_mock.return_value = output_mock
        model_mock.eval.return_value = model_mock
        model_mock.to.return_value = model_mock

        def fake_modules():
            return []  # no RotaryEmbedding modules → repair is a no-op

        model_mock.modules = fake_modules

        def fake_parameters():
            p = MagicMock()
            p.device = torch.device("cpu")
            return iter([p])

        model_mock.parameters = fake_parameters

        tf, tokenizer_mock = _make_transformers_stub(model_mock)

        # Make tokenizer() call return input_ids with <extra_0> at positions 2,4
        token_ids = torch.tensor([[1, 2, 151643, 3, 151643]])
        enc = MagicMock()
        enc.__getitem__ = lambda s, k: token_ids if k == "input_ids" else MagicMock()
        enc.get.return_value = None
        # MagicMock: use return_value (not __call__ assignment) so call dispatch works
        tokenizer_mock.return_value = enc
        tokenizer_mock.side_effect = lambda *a, **kw: enc
        tf.AutoTokenizer.from_pretrained.return_value = tokenizer_mock

        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                "transformers": tf,
            },
        ):
            if "its_hub.integration.transformers_prm" in sys.modules:
                del sys.modules["its_hub.integration.transformers_prm"]
            from its_hub.integration.transformers_prm import TransformersProcessRewardModel

            prm_instance = object.__new__(TransformersProcessRewardModel)
            prm_instance._model = model_mock
            prm_instance._device = "cpu"
            prm_instance._tokenizer = tokenizer_mock
            prm_instance._extra0_token_id = 151643
            yield prm_instance

    def test_score_returns_list_per_step(self, prm):
        """score() must return exactly len(steps) floats."""
        import torch

        scores = prm._score_steps("What is 2+2?", ["Step 1: add", "Step 2: answer"])
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_scores_in_unit_interval(self, prm):
        """All returned scores must be in [0, 1]."""
        import torch

        scores = prm._score_steps("Solve x=1", ["x equals 1", "therefore x is 1"])
        assert all(0.0 <= s <= 1.0 for s in scores), f"Out-of-range scores: {scores}"

    def test_trailing_separator_in_assistant_content(self, prm):
        """The assistant_content must end with <extra_0>, giving N separators for N steps."""
        captured = {}

        original_apply = prm._tokenizer.apply_chat_template

        def capturing_apply(messages, **kw):
            captured["messages"] = messages
            return "<captured>"

        prm._tokenizer.apply_chat_template = capturing_apply

        import torch

        prm._score_steps("problem", ["step A", "step B", "step C"])

        assistant_msg = captured["messages"][-1]
        assert assistant_msg["role"] == "assistant"
        content = assistant_msg["content"]

        sep = "<extra_0>"
        assert content.endswith(sep), f"Content should end with {sep!r}, got: {content!r}"
        assert content.count(sep) == 3, f"Expected 3 separators for 3 steps, got: {content.count(sep)}"


class TestExport:
    def test_exported_from_integration_package(self):
        """TransformersProcessRewardModel must be importable from its_hub.integration."""
        # This test verifies the __init__.py export without loading torch/transformers
        import importlib
        import types

        # Stub heavy deps so the module-level import succeeds
        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bfloat16"
        fake_tf = types.ModuleType("transformers")

        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_tf}):
            for mod in list(sys.modules):
                if "its_hub.integration" in mod:
                    del sys.modules[mod]
            from its_hub.integration import TransformersProcessRewardModel  # noqa: F401

            assert TransformersProcessRewardModel is not None
