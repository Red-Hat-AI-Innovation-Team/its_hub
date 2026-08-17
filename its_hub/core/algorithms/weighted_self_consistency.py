"""WeightedSelfConsistency: confidence-weighted majority voting.

Generates N candidate responses with logprobs, computes per-candidate tail
confidence scores, then performs majority voting where each vote is weighted
by the candidate's confidence.  Selects the answer group with the highest
total weight, and within that group picks the most confident candidate.

Combines the signals of SelfConsistency (agreement across responses) and
ConfidenceSelection (model-internal certainty) into a single algorithm.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
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
from its_hub.core.algorithms.confidence_selection import (
    adaptive_tail_window,
    compute_token_certainties,
    compute_token_entropies,
    tail_scores,
    trim_length_outliers,
)
from its_hub.core.algorithms.self_consistency import _default_projection_func
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.utils import extract_content_from_lm_response


@dataclass
class WeightedSelfConsistencyResult(AbstractScalingResult):
    responses: list[dict]
    scores: list[float]
    weights: list[float]
    group_weights: dict[str, float]
    selected_index: int
    tail_window: int
    usage: GenerationUsage | None = None

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


def _scores_to_weights(scores: list[float], metric: str) -> list[float]:
    """Convert tail confidence scores to positive vote weights.

    For entropy (lower is better):  weight = exp(-score)
    For certainty (higher is better): weight = score  (raw, no transform)

    Excluded candidates (score=inf/-inf) get weight 0.
    """
    arr = np.array(scores, dtype=np.float64)
    raw = np.exp(-arr) if metric == "entropy" else np.maximum(arr, 0.0)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    return raw.tolist()


def _select_weighted_majority(
    responses: list[dict],
    weights: list[float],
    scores: list[float],
    metric: str,
    projection_func: Callable,
) -> tuple[int, dict[str, float]]:
    """Group responses by projected answer, sum weights, pick the winner.

    Returns (selected_index, group_weights) where selected_index is the
    index of the best-confidence candidate within the winning group.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, resp in enumerate(responses):
        content = extract_content_from_lm_response(resp)
        key = projection_func(content)
        groups[key].append(i)

    group_weights: dict[str, float] = {}
    for key, indices in groups.items():
        group_weights[key] = sum(weights[i] for i in indices)

    winning_key = max(group_weights, key=group_weights.get)
    winning_indices = groups[winning_key]

    if metric == "entropy":
        best_in_group = min(winning_indices, key=lambda i: scores[i])
    else:
        best_in_group = max(winning_indices, key=lambda i: scores[i])

    return best_in_group, group_weights


class WeightedSelfConsistency(AbstractScalingAlgorithm):
    """Confidence-weighted majority voting.

    Generates *budget* candidate responses with token-level logprobs,
    computes per-candidate tail confidence, then performs majority voting
    where each candidate's vote is weighted by its confidence score.

    - When all candidates have similar confidence, this degenerates to
      standard majority voting (SelfConsistency behavior).
    - When candidates all disagree, this degenerates to picking the most
      confident response (ConfidenceSelection behavior).
    """

    def __init__(
        self,
        consistency_space_projection_func: Callable | None = None,
        metric: str = "entropy",
        top_logprobs: int = 20,
        tail_min: int = 64,
        tail_max: int = 2048,
        agg: str = "median",
        trim_pct: float = 0.05,
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

        self.consistency_space_projection_func = (
            consistency_space_projection_func or _default_projection_func
        )
        self.metric = metric
        self.top_logprobs = top_logprobs
        self.tail_min = tail_min
        self.tail_max = tail_max
        self.agg = agg
        self.trim_pct = trim_pct
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
    ) -> dict | WeightedSelfConsistencyResult:
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

        # --- Compute per-candidate tail confidence scores ---

        use_certainty = self.metric == "certainty"
        if use_certainty:

            def score_fn(content: list[dict]) -> list[float]:
                return compute_token_certainties(content, self.vocab_size)

            label = "certainty"
        else:
            score_fn = compute_token_entropies
            label = "entropy"

        scores_per_token: list[list[float]] = []
        for i, resp in enumerate(responses):
            lp = resp.get("_logprobs")
            if lp is not None and lp.get("content"):
                scores_per_token.append(score_fn(lp["content"]))
            else:
                logging.warning(
                    "Response %d has no logprobs data; excluded from %s weighting",
                    i,
                    label,
                )
                scores_per_token.append([])

        usable = [i for i, s in enumerate(scores_per_token) if s]
        if not usable:
            raise ValueError(
                "No candidates have logprobs data; cannot compute weights."
            )

        usable_lengths = [len(scores_per_token[i]) for i in usable]
        trimmed_usable = trim_length_outliers(usable_lengths, self.trim_pct)
        included = [usable[j] for j in trimmed_usable]

        tail_window = adaptive_tail_window(
            scores_per_token, included, self.tail_min, self.tail_max
        )

        default_score = float("-inf") if use_certainty else float("inf")
        scores = tail_scores(
            scores_per_token, included, tail_window, self.agg, default_score
        )

        # --- Convert to weights and select via weighted majority ---

        weights = _scores_to_weights(scores, self.metric)

        selected_index, group_weights = _select_weighted_majority(
            responses,
            weights,
            scores,
            self.metric,
            self.consistency_space_projection_func,
        )

        result = WeightedSelfConsistencyResult(
            responses=responses,
            scores=scores,
            weights=weights,
            group_weights=group_weights,
            selected_index=selected_index,
            tail_window=tail_window,
            usage=usage,
        )
        return result.the_one if return_response_only else result
