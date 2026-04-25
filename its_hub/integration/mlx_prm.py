"""MLX-based Process Reward Model for Apple Silicon.

Removes the CUDA dependency required by LocalVllmProcessRewardModel, allowing
local PRM scoring on macOS with Apple Silicon hardware.
"""

from __future__ import annotations

import asyncio
import math

from its_hub.base import AbstractProcessRewardModel
from its_hub.types import ChatMessage, ChatMessages


class MLXProcessRewardModel(AbstractProcessRewardModel):
    """Process Reward Model backed by MLX with 4-bit quantized weights.

    Scores step-by-step reasoning trajectories using Qwen2.5-Math-PRM-7B
    (or any compatible model) loaded via mlx-lm on Apple Silicon.

    For each trajectory, the model assigns a per-step quality score by
    computing P(correct) = softmax([logit(good_token), logit(bad_token)])
    at each step-boundary position.  The score returned for a given
    (prompt, response) pair is the quality of the *last* step in the
    response — matching how ParticleFiltering calls the PRM incrementally.

    Args:
        model_name: HuggingFace model path or local directory.  Defaults to
            the 4-bit quantised community upload of Qwen2.5-Math-PRM-7B.
        step_sep: Token string that separates reasoning steps.
        good_token: Vocabulary token representing a correct step.
        bad_token: Vocabulary token representing an incorrect step.
        max_seq_len: Maximum token length fed to the model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        step_sep: str = "\n",
        good_token: str = "+",
        bad_token: str = "-",
        max_seq_len: int = 4096,
    ):
        try:
            import mlx.core as mx
            import mlx_lm
        except ImportError as exc:
            raise ImportError(
                "MLXProcessRewardModel requires the MLX framework for Apple Silicon. "
                "Install with: pip install mlx mlx-lm"
            ) from exc

        self._mx = mx
        self._step_sep = step_sep
        self._max_seq_len = max_seq_len

        self._model, self._tokenizer = mlx_lm.load(model_name)

        self._good_token_id = self._tokenizer.convert_tokens_to_ids(good_token)
        self._bad_token_id = self._tokenizer.convert_tokens_to_ids(bad_token)

    # ------------------------------------------------------------------
    # Internal scoring helpers
    # ------------------------------------------------------------------

    def _get_step_boundary_positions(self, token_ids: list[int]) -> list[int]:
        """Return indices immediately *after* each step-separator occurrence."""
        sep_ids = self._tokenizer.encode(self._step_sep, add_special_tokens=False)
        sep_len = len(sep_ids)
        positions: list[int] = []
        for i in range(len(token_ids) - sep_len + 1):
            if token_ids[i : i + sep_len] == sep_ids:
                positions.append(i + sep_len - 1)  # position of last sep token
        return positions

    def _score_single(self, prompt: str, response: str) -> float:
        """Compute a scalar score for one (prompt, response) pair."""
        import mlx.core as mx

        full_text = prompt + response
        token_ids: list[int] = self._tokenizer.encode(
            full_text, add_special_tokens=True
        )
        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]

        input_ids = mx.array(token_ids)[None]  # (1, seq_len)
        logits = self._model(input_ids)  # (1, seq_len, vocab_size)
        logits = logits[0]  # (seq_len, vocab_size)
        mx.eval(logits)

        # Find step boundaries; fall back to last token if none found
        boundary_positions = self._get_step_boundary_positions(token_ids)
        score_position = boundary_positions[-1] if boundary_positions else len(token_ids) - 1
        # Clamp to sequence length (may have been truncated)
        score_position = min(score_position, len(token_ids) - 1)

        pos_logits = logits[score_position]  # (vocab_size,)
        good_logit = float(pos_logits[self._good_token_id])
        bad_logit = float(pos_logits[self._bad_token_id])

        # Numerical-stable 2-class softmax
        shift = max(good_logit, bad_logit)
        good_exp = math.exp(good_logit - shift)
        bad_exp = math.exp(bad_logit - shift)
        return good_exp / (good_exp + bad_exp)

    # ------------------------------------------------------------------
    # AbstractProcessRewardModel interface
    # ------------------------------------------------------------------

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        prompt = chat_messages.to_prompt()

        is_single = isinstance(response_or_responses, str)
        responses = [response_or_responses] if is_single else response_or_responses

        scores = await asyncio.to_thread(self._score_batch, prompt, responses)
        return scores[0] if is_single else scores

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        return asyncio.run(self.ascore(prompt_or_messages, response_or_responses))

    def _score_batch(self, prompt: str, responses: list[str]) -> list[float]:
        return [self._score_single(prompt, r) for r in responses]
