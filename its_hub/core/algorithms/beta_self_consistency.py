import logging
from collections import Counter
from collections.abc import Callable

from scipy.special import betainc

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


class BetaSelfConsistency(SelfConsistency):
    """Self-consistency with Beta-distribution adaptive stopping (Aggarwal et al., 2023).

    Samples one response at a time and computes a Beta posterior probability
    that the current majority answer will remain the majority. Stops when this
    probability exceeds confidence_threshold (default 0.95).

    The stopping criterion uses the regularized incomplete beta function:
        P(majority stays) = 1 - I_{0.5}(v1 + 1, v2 + 1)
    where v1 is the count of the most frequent answer and v2 is the count of
    the second most frequent answer.

    Reference: "Let's Sample Step by Step: Adaptive-Consistency for Efficient
    Reasoning and Coding with LLMs" (EMNLP 2023)

    Inherits tool_vote, exclude_args, and projection support from
    SelfConsistency.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        consistency_space_projection_func: Callable | None = None,
        tool_vote: str | None = None,
        exclude_args: list[str] | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if not 0.5 < confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0.5, 1.0], got: {confidence_threshold}"
            )

        super().__init__(
            consistency_space_projection_func=consistency_space_projection_func,
            tool_vote=tool_vote,
            exclude_args=exclude_args,
            orchestrator=orchestrator,
        )
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def beta_stopping_probability(v1: int, v2: int) -> float:
        """Compute P(majority answer remains majority) using Beta posterior.

        Args:
            v1: Count of the most frequent answer.
            v2: Count of the second most frequent answer.

        Returns:
            Probability in [0, 1] that the majority answer stays dominant.
        """
        return 1.0 - float(betainc(v1 + 1, v2 + 1, 0.5))

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

        for sample_num in range(1, budget + 1):
            new_responses = await self.orchestrator.agenerate(
                lm,
                chat_messages.to_batch(1),
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
            )
            all_responses.extend(new_responses)

            if sample_num >= budget:
                break

            eligible_indices, projected = self._project_responses(all_responses)

            if len(eligible_indices) >= 2:
                counts = Counter(projected)
                most_common = counts.most_common(2)
                v1 = most_common[0][1]
                v2 = most_common[1][1] if len(most_common) > 1 else 0

                prob = self.beta_stopping_probability(v1, v2)

                if prob >= self.confidence_threshold:
                    logger.info(
                        "Early stop at sample %d: P(majority stays)=%.4f >= %.4f",
                        sample_num,
                        prob,
                        self.confidence_threshold,
                    )
                    break

        logger.info(
            "BetaSelfConsistency used %d/%d samples",
            len(all_responses),
            budget,
        )

        return self._process_responses(all_responses, return_response_only, usage)
