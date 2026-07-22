import logging
from collections import Counter
from collections.abc import Callable

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractScalingAlgorithm,
    ChatMessage,
    ChatMessages,
    GenerationUsage,
)
from its_hub.core.algorithms.self_consistency import (
    SelfConsistencyResult,
    _default_projection_func,
    _select_hierarchical_most_common_or_random,
    _select_most_common_or_random,
)
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.utils import extract_content_from_lm_response

logger = logging.getLogger(__name__)


class AdaptiveSelfConsistency(AbstractScalingAlgorithm):
    """Self-consistency with exponential doubling and early stopping.

    Starts with 2 samples, checks if a supermajority (default 75%) agree,
    and doubles the sample count each round until the budget is reached.
    Previous samples are kept across rounds — no resampling.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        consistency_space_projection_func: Callable | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if not 0.5 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.5, 1.0], got: {threshold}")

        self.threshold = threshold
        self.consistency_space_projection_func = (
            consistency_space_projection_func or _default_projection_func
        )

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
    ) -> dict | SelfConsistencyResult:
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        usage = GenerationUsage()

        all_responses: list[dict] = []
        next_batch_size = min(2, budget)
        round_num = 0

        while len(all_responses) < budget:
            batch_size = min(next_batch_size, budget - len(all_responses))
            round_num += 1

            new_responses = await self.orchestrator.agenerate(
                lm,
                chat_messages.to_batch(batch_size),
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
            )
            all_responses.extend(new_responses)

            # Check stopping criterion (skip if we've exhausted the budget)
            if len(all_responses) >= budget:
                break

            projected = [
                self.consistency_space_projection_func(
                    extract_content_from_lm_response(r)
                )
                for r in all_responses
            ]
            counts = Counter(projected)
            top_count = counts.most_common(1)[0][1]
            agreement = top_count / len(all_responses)

            if agreement >= self.threshold:
                logger.info(
                    "Early stop at round %d: %d/%d samples agree (%.0f%% >= %.0f%%)",
                    round_num,
                    top_count,
                    len(all_responses),
                    agreement * 100,
                    self.threshold * 100,
                )
                break

            # Double: next batch size equals current total
            next_batch_size = len(all_responses)

        logger.info(
            "AdaptiveSelfConsistency used %d/%d samples in %d round(s)",
            len(all_responses),
            budget,
            round_num,
        )

        return self._process_responses(all_responses, return_response_only, usage)

    def _process_responses(
        self,
        responses: list[dict],
        return_response_only: bool = True,
        usage: GenerationUsage | None = None,
    ) -> dict | SelfConsistencyResult:
        projected = [
            self.consistency_space_projection_func(extract_content_from_lm_response(r))
            for r in responses
        ]

        if projected and isinstance(projected[0], tuple):
            counts, selected_index = _select_hierarchical_most_common_or_random(
                projected
            )
        else:
            counts, selected_index = _select_most_common_or_random(projected)

        result = SelfConsistencyResult(
            responses=responses,
            response_counts=counts,
            selected_index=selected_index,
            usage=usage,
        )
        return result.the_one if return_response_only else result
