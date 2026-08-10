"""ConfidenceSelection: select the best response by lowest tail entropy.

Generates N candidate responses with logprobs enabled, computes per-token
Shannon entropy from the top-k log probabilities, then selects the candidate
whose tail (final portion) has the lowest aggregated entropy — indicating the
model was most confident about its conclusion.
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


def trim_length_outliers(
    lengths: list[int],
    trim_pct: float = 0.05,
) -> list[int]:
    """Return indices of candidates within the [p5, p95] length range."""
    n = len(lengths)
    if n < 16:
        return list(range(n))
    lo = float(np.percentile(lengths, trim_pct * 100))
    hi = float(np.percentile(lengths, (1 - trim_pct) * 100))
    included = [i for i in range(n) if lo <= lengths[i] <= hi]
    return included if len(included) >= 2 else list(range(n))


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
    entropies_per_token: list[list[float]],
    included: list[int],
    tail: int,
    agg: str = "median",
) -> list[float]:
    """Compute aggregated entropy over the tail window for each candidate."""
    n = len(entropies_per_token)
    included_set = set(included)
    fn = np.median if agg == "median" else np.mean
    scores: list[float] = []
    for i in range(n):
        if i not in included_set or not entropies_per_token[i]:
            scores.append(float("inf"))
        else:
            window = entropies_per_token[i][-tail:]
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
    """Select the best-of-N response by lowest tail entropy.

    Generates *budget* candidate responses with token-level logprobs,
    computes per-token Shannon entropy from the top-k log probabilities,
    then picks the candidate whose tail (final tokens) has the lowest
    aggregated entropy — the response where the model was most confident.

    No external reward model is required.
    """

    def __init__(
        self,
        top_logprobs: int = 20,
        tail_min: int = 64,
        tail_max: int = 2048,
        agg: str = "median",
        trim_pct: float = 0.05,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if agg not in ("median", "mean"):
            raise ValueError(f"agg must be 'median' or 'mean', got: {agg!r}")
        if not 1 <= top_logprobs <= 20:
            raise ValueError(f"top_logprobs must be 1-20, got: {top_logprobs}")

        self.top_logprobs = top_logprobs
        self.tail_min = tail_min
        self.tail_max = tail_max
        self.agg = agg
        self.trim_pct = trim_pct

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

        entropies_per_token: list[list[float]] = []
        for i, resp in enumerate(responses):
            lp = resp.get("_logprobs")
            if lp is not None and lp.get("content"):
                entropies_per_token.append(compute_token_entropies(lp["content"]))
            else:
                logging.warning(
                    "Response %d has no logprobs data; assigning infinite entropy", i
                )
                entropies_per_token.append([])

        selected_index, scores, tail_window = select_by_tail_entropy(
            entropies_per_token,
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
