import logging
from collections import Counter
from collections.abc import Callable

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    ChatMessage,
    ChatMessages,
    GenerationUsage,
)
from its_hub.core.algorithms.self_consistency import (
    SelfConsistency,
    SelfConsistencyResult,
)

logger = logging.getLogger(__name__)


class AdaptiveSelfConsistency(SelfConsistency):
    """Self-consistency with exponential doubling and early stopping.

    Starts with 2 samples, checks if a supermajority (default 75%) agree,
    and doubles the sample count each round until the budget is reached.
    Previous samples are kept across rounds — no resampling.

    Inherits tool_vote, exclude_args, and projection support from
    SelfConsistency.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        consistency_space_projection_func: Callable | None = None,
        tool_vote: str | None = None,
        exclude_args: list[str] | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if not 0.5 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0.5, 1.0], got: {threshold}")

        super().__init__(
            consistency_space_projection_func=consistency_space_projection_func,
            tool_vote=tool_vote,
            exclude_args=exclude_args,
            orchestrator=orchestrator,
        )
        self.threshold = threshold

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

            eligible_indices, projected = self._project_responses(all_responses)

            if eligible_indices:
                counts = Counter(projected)
                top_count = counts.most_common(1)[0][1]
                agreement = top_count / len(eligible_indices)

                if agreement >= self.threshold:
                    logger.info(
                        "Early stop at round %d: %d/%d samples agree (%.0f%% >= %.0f%%)",
                        round_num,
                        top_count,
                        len(eligible_indices),
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
