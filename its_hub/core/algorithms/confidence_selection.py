"""ConfidenceSelection: select the best response by tail confidence metrics.

Generates N candidate responses with logprobs enabled, computes per-token
confidence scores from the top-k log probabilities, then selects the candidate
whose tail (final portion) shows the highest model confidence.

Supports two metrics:
- **entropy** (default): Shannon entropy; selects the *lowest* tail entropy.
- **certainty**: KL(uniform || p_model); selects the *highest* tail certainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
    ChatMessage,
    ChatMessages,
    GenerationUsage,
)
from its_hub.core.orchestrator import LMOrchestrator

# ---------------------------------------------------------------------------
# Pure functions — ported from research reference implementation
# ---------------------------------------------------------------------------


def compute_token_entropies(logprobs_content: list[dict]) -> list[float]:
    """Compute per-token Shannon entropy from OpenAI-format logprobs.

    Args:
        logprobs_content: The ``choice["logprobs"]["content"]`` list. Each
            element is a dict with a ``top_logprobs`` key containing a list
            of ``{"token": str, "logprob": float}`` dicts.

    Returns:
        Per-token entropy values (nats).
    """
    entropies: list[float] = []
    for token_info in logprobs_content:
        top_lps = token_info.get("top_logprobs", [])
        if not top_lps:
            entropies.append(0.0)
            continue
        log_ps = np.array([t["logprob"] for t in top_lps])
        ps = np.exp(log_ps)
        entropy = float(-np.sum(ps * np.log(ps + 1e-10)))
        entropies.append(entropy)
    return entropies


def compute_token_certainties(
    logprobs_content: list[dict],
    vocab_size: int | None = None,
) -> list[float]:
    """Compute per-token self-certainty from OpenAI-format logprobs.

    Self-certainty is defined as ``KL(uniform || p_model)``:

        certainty(t) = -log(V) - (1/V) * sum_v log p(v)

    Higher values indicate a more peaked (decisive) distribution.

    Args:
        logprobs_content: The ``choice["logprobs"]["content"]`` list.
        vocab_size: Vocabulary size *V* in the formula.  When *None*
            (default), uses the number of top-k logprobs at each position.

    Returns:
        Per-token certainty values (nats).
    """
    certainties: list[float] = []
    for token_info in logprobs_content:
        top_lps = token_info.get("top_logprobs", [])
        if not top_lps:
            certainties.append(0.0)
            continue
        log_ps = np.array([t["logprob"] for t in top_lps])
        v = vocab_size if vocab_size is not None else len(log_ps)
        v = max(v, 1)
        certainty = float(-np.log(v) - np.sum(log_ps) / v)
        certainties.append(certainty)
    return certainties


def trim_length_outliers(
    lengths: list[int],
    trim_pct: float = 0.05,
) -> list[int]:
    """Return indices of candidates whose length is within the percentile range defined by *trim_pct*."""
    n = len(lengths)
    if n < 16:
        return list(range(n))
    lo = float(np.percentile(lengths, trim_pct * 100))
    hi = float(np.percentile(lengths, (1 - trim_pct) * 100))
    included = [i for i in range(n) if lo <= lengths[i] <= hi]
    return included


def adaptive_tail_window(
    entropies_per_token: list[list[float]],
    included: list[int],
    tail_min: int = 64,
    tail_max: int = 2048,
) -> int:
    """Compute tail window size adaptive to length and entropy.

    Formula: ``clamp(sqrt(mean_len) / mean_entropy, tail_min, tail_max)``

    sqrt(mean_len) scales sub-linearly with response length.  Dividing by
    mean entropy makes the window entropy-adaptive: confident (low-entropy)
    sequences get a wider tail so more signal is captured, while uncertain
    sequences get a shorter tail to avoid dilution.
    """
    lengths = [len(entropies_per_token[i]) for i in included]
    mean_len = float(np.mean(lengths))
    mean_entropy = float(
        np.mean(
            [
                np.mean(entropies_per_token[i][-2048:])
                for i in included
                if entropies_per_token[i]
            ]
        )
    )
    mean_entropy = max(mean_entropy, 1e-6)
    tail = int(np.clip(np.sqrt(mean_len) / mean_entropy, tail_min, tail_max))
    return tail


def tail_scores(
    scores_per_token: list[list[float]],
    included: list[int],
    tail: int,
    agg: str = "median",
    default_score: float = float("inf"),
) -> list[float]:
    """Compute aggregated score over the tail window for each candidate."""
    n = len(scores_per_token)
    included_set = set(included)
    fn = np.median if agg == "median" else np.mean
    scores: list[float] = []
    for i in range(n):
        if i not in included_set or not scores_per_token[i]:
            scores.append(default_score)
        else:
            window = scores_per_token[i][-tail:]
            scores.append(float(fn(window)))
    return scores


def select_by_tail_entropy(
    entropies_per_token: list[list[float]],
    tail_min: int = 64,
    tail_max: int = 2048,
    agg: str = "median",
    trim_pct: float = 0.05,
) -> tuple[int, list[float], int]:
    """Select the response with lowest tail entropy.

    Args:
        entropies_per_token: N lists of per-token Shannon entropy (nats).
        tail_min: Minimum tail window size in tokens.
        tail_max: Maximum tail window size in tokens.
        agg: Aggregation over the tail window (``"median"`` or ``"mean"``).
        trim_pct: Fraction of length outliers to trim from each end.

    Returns:
        ``(selected_index, scores, tail_window)`` where *scores[i]* is the
        aggregated entropy over the tail window for candidate *i*, and
        *tail_window* is the adaptive window size used.
    """
    usable = [i for i, e in enumerate(entropies_per_token) if e]
    if not usable:
        raise ValueError("No candidates have entropy data; cannot select.")

    usable_lengths = [len(entropies_per_token[i]) for i in usable]
    trimmed_usable = trim_length_outliers(usable_lengths, trim_pct)
    included = [usable[j] for j in trimmed_usable]

    tail = adaptive_tail_window(entropies_per_token, included, tail_min, tail_max)
    scores = tail_scores(entropies_per_token, included, tail, agg)
    selected = int(np.argmin(scores))
    return selected, scores, tail


def select_by_tail_certainty(
    certainties_per_token: list[list[float]],
    tail_min: int = 64,
    tail_max: int = 2048,
    agg: str = "median",
    trim_pct: float = 0.05,
) -> tuple[int, list[float], int]:
    """Select the response with highest tail certainty.

    Same adaptive pipeline as :func:`select_by_tail_entropy` but picks by
    **argmax** (higher certainty = more peaked distribution = more confident).

    Returns:
        ``(selected_index, scores, tail_window)``
    """
    usable = [i for i, c in enumerate(certainties_per_token) if c]
    if not usable:
        raise ValueError("No candidates have certainty data; cannot select.")

    usable_lengths = [len(certainties_per_token[i]) for i in usable]
    trimmed_usable = trim_length_outliers(usable_lengths, trim_pct)
    included = [usable[j] for j in trimmed_usable]

    tail = adaptive_tail_window(certainties_per_token, included, tail_min, tail_max)
    scores = tail_scores(
        certainties_per_token, included, tail, agg, default_score=float("-inf")
    )
    selected = int(np.argmax(scores))
    return selected, scores, tail


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceSelectionResult(AbstractScalingResult):
    responses: list[dict]
    scores: list[float]
    selected_index: int
    tail_window: int
    usage: GenerationUsage | None = None

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class ConfidenceSelection(AbstractScalingAlgorithm):
    """Select the best-of-N response by tail confidence metrics.

    Generates *budget* candidate responses with token-level logprobs,
    computes per-token confidence scores, then picks the most confident
    candidate based on the chosen metric:

    - ``"entropy"`` (default): selects **lowest** tail entropy.
    - ``"certainty"``: selects **highest** tail certainty
      (KL divergence from uniform).

    No external reward model is required.
    """

    def __init__(
        self,
        top_logprobs: int = 20,
        tail_min: int = 64,
        tail_max: int = 2048,
        agg: str = "median",
        trim_pct: float = 0.05,
        metric: str = "entropy",
        vocab_size: int | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if agg not in ("median", "mean"):
            raise ValueError(f"agg must be 'median' or 'mean', got: {agg!r}")
        if not 1 <= top_logprobs <= 20:
            raise ValueError(f"top_logprobs must be 1-20, got: {top_logprobs}")
        if metric not in ("entropy", "certainty"):
            raise ValueError(
                f"metric must be 'entropy' or 'certainty', got: {metric!r}"
            )

        self.top_logprobs = top_logprobs
        self.tail_min = tail_min
        self.tail_max = tail_max
        self.agg = agg
        self.trim_pct = trim_pct
        self.metric = metric
        self.vocab_size = vocab_size

        if orchestrator is None:
            orchestrator = LMOrchestrator()
        self.orchestrator = orchestrator

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | ConfidenceSelectionResult:
        """Run inference with confidence-based selection."""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        usage = GenerationUsage()

        responses = await self.orchestrator.agenerate(
            lm,
            chat_messages.to_batch(budget),
            tools=tools,
            tool_choice=tool_choice,
            usage_accumulator=usage,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        use_certainty = self.metric == "certainty"
        if use_certainty:

            def score_fn(content: list[dict]) -> list[float]:
                return compute_token_certainties(content, self.vocab_size)

            select_fn = select_by_tail_certainty
            label = "certainty"
        else:
            score_fn = compute_token_entropies
            select_fn = select_by_tail_entropy
            label = "entropy"

        scores_per_token: list[list[float]] = []
        for i, resp in enumerate(responses):
            lp = resp.get("_logprobs")
            if lp is not None and lp.get("content"):
                scores_per_token.append(score_fn(lp["content"]))
            else:
                logging.warning(
                    "Response %d has no logprobs data; excluded from %s selection",
                    i,
                    label,
                )
                scores_per_token.append([])

        selected_index, scores, tail_window = select_fn(
            scores_per_token,
            tail_min=self.tail_min,
            tail_max=self.tail_max,
            agg=self.agg,
            trim_pct=self.trim_pct,
        )

        result = ConfidenceSelectionResult(
            responses=responses,
            scores=scores,
            selected_index=selected_index,
            tail_window=tail_window,
            usage=usage,
        )
        return result.the_one if return_response_only else result
