"""Transformers-based Process Reward Model for classifier-head PRMs.

Implements the correct scoring algorithm for Qwen2.5-Math-PRM-7B:
  1. Steps are joined with <extra_0> (the model's canonical step separator).
  2. One forward pass through the base transformer yields hidden states.
  3. model.score(hidden_states) applies the 2-class head → [batch, seq, 2].
  4. Softmax(positive class) at each <extra_0> position = per-step score.

Unlike MLXProcessRewardModel (which targets Math-Shepherd-style "+/-" token
scoring), this implementation handles classifier-head PRMs that cannot be
loaded by mlx_lm.
"""

from __future__ import annotations

import asyncio

from its_hub.base import AbstractProcessRewardModel
from its_hub.types import ChatMessage, ChatMessages
from its_hub.utils import QWEN_SYSTEM_PROMPT

_STEP_SEP_TOKEN = "<extra_0>"  # Qwen2.5-Math-PRM canonical step separator


class TransformersProcessRewardModel(AbstractProcessRewardModel):
    """Process Reward Model using transformers + MPS/CUDA for Qwen2.5-Math-PRM-7B.

    Input columns consumed by the companion sdg_hub block: ``problem`` (str)
    and ``steps`` (``list[str]``).  One forward pass returns one score per step.

    Args:
        model_name: HuggingFace model path or local directory.
        device: PyTorch device string.  ``"mps"`` for Apple Silicon,
            ``"cuda"`` for NVIDIA, ``"cpu"`` for CPU-only.
        dtype: ``torch.dtype`` for model weights.  Defaults to
            ``torch.bfloat16``; bfloat16 shares float32's exponent range so
            it avoids the NaN hidden states that fp16 produces on MPS for
            Qwen2ForProcessRewardModel's deep backbone.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: str = "mps",
        dtype=None,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TransformersProcessRewardModel requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from exc

        _dtype = dtype if dtype is not None else torch.bfloat16
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Qwen2RMConfig does not always carry pad_token_id from the JSON;
        # load the config first and backfill it before constructing the model.
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=_dtype,
        ).eval().to(device)

        # Transformers 5.x uses meta-tensor initialisation: non-persistent buffers
        # (inv_freq, cos_cached, sin_cached) on Qwen2RotaryEmbedding are materialised
        # as zeros rather than being recomputed.  Force-recompute them now.
        self._repair_rotary_embeddings()

        self._extra0_token_id: int = self._tokenizer.convert_tokens_to_ids(_STEP_SEP_TOKEN)

    # ------------------------------------------------------------------
    # Rotary embedding repair
    # ------------------------------------------------------------------

    def _repair_rotary_embeddings(self) -> None:
        """Recompute inv_freq / cos_cached / sin_cached for every rotary layer.

        Transformers ≥5 uses meta-tensor initialisation during from_pretrained.
        Non-persistent buffers (inv_freq, cos_cached, sin_cached) are skipped by
        load_state_dict and end up as zeros.  We recompute them from the module's
        stored base / dim / max_seq_len_cached.
        """
        import torch

        device = next(self._model.parameters()).device
        repaired = 0
        for module in self._model.modules():
            if "RotaryEmbedding" not in type(module).__name__:
                continue
            if not (hasattr(module, "base") and hasattr(module, "dim") and hasattr(module, "inv_freq")):
                continue
            inv_freq = 1.0 / (
                module.base
                ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float().to(device) / module.dim)
            )
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            if hasattr(module, "_set_cos_sin_cache"):
                module._set_cos_sin_cache(
                    seq_len=module.max_seq_len_cached,
                    device=device,
                    dtype=torch.float32,
                )
            repaired += 1

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        """Single forward pass; returns one score per step."""
        import torch

        # Append a trailing separator so each step has exactly one <extra_0> token.
        # "<extra_0>".join(steps) + "<extra_0>" produces N separators for N steps.
        assistant_content = _STEP_SEP_TOKEN.join(steps) + _STEP_SEP_TOKEN
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self._tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        # Locate <extra_0> positions in the token sequence
        extra0_positions = (
            (input_ids[0] == self._extra0_token_id).nonzero(as_tuple=True)[0]
        )

        with torch.no_grad():
            # Qwen2ForProcessRewardModel.forward() returns TokenClassifierOutput
            # with logits of shape (1, seq, 2) already computed via the score head.
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,  # avoids DynamicCache.from_legacy_cache in transformers 5.x
            )
            logits = output.logits  # (1, seq, 2)

        if len(extra0_positions) == 0:
            # No step boundaries found — use last-token score as fallback
            last_logits = logits[0, -1, :].float()
            score = float(torch.softmax(last_logits, dim=-1)[1])
            return [score] * len(steps)

        scores: list[float] = []
        for pos in extra0_positions:
            step_logits = logits[0, int(pos), :].float()
            scores.append(float(torch.softmax(step_logits, dim=-1)[1]))

        # Align to step count in case of truncation or tokenisation quirks
        while len(scores) < len(steps):
            scores.append(scores[-1])
        return scores[: len(steps)]

    # ------------------------------------------------------------------
    # AbstractProcessRewardModel interface
    # ------------------------------------------------------------------

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        return self._score_steps(chat_messages.to_prompt(), steps)

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        prompt = chat_messages.to_prompt()
        return await asyncio.to_thread(self._score_steps, prompt, steps)
