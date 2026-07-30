import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from math import comb

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

    Fires all budget requests concurrently and checks the Beta posterior
    probability as each response arrives. Stops and cancels remaining
    requests once the probability that the majority answer will remain
    dominant exceeds confidence_threshold (default 0.95).

    This fully utilizes vLLM's continuous batching while still getting
    the sample-efficiency benefits of adaptive stopping.

    Let v1 and v2 be the vote counts for the two most common answers. Starting
    from a uniform Beta(1, 1) prior, their posterior is Beta(v1 + 1, v2 + 1).
    Generation stops when the posterior probability that the leading answer's
    vote share exceeds 0.5 reaches the configured threshold:

        P(leader's vote share > 0.5) = 1 - I_0.5(v1 + 1, v2 + 1)

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
        # At x=0.5 and with integer parameters, the beta CDF is exactly the
        # lower tail of a fair binomial distribution. This avoids requiring
        # scipy.special.betainc.
        num_trials = v1 + v2 + 1
        beta_cdf = sum(comb(num_trials, k) for k in range(v2 + 1)) / (1 << num_trials)
        return 1.0 - beta_cdf

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

        messages_list = chat_messages.to_chat_messages()

        async def _generate_one() -> dict:
            # A one-item batch lets each response complete independently while
            # the shared orchestrator still enforces its concurrency limit.
            responses = await self.orchestrator.agenerate(
                lm,
                [messages_list],
                tools=tools,
                tool_choice=tool_choice,
                usage_accumulator=usage,
            )
            return responses[0]

        tasks = [asyncio.create_task(_generate_one()) for _ in range(budget)]

        all_responses: list[dict] = []
        pending = set(tasks)
        accounted_tasks: set[asyncio.Task] = set()
        stopped_early = False

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                # asyncio.wait may return several tasks at once. Account for every
                # completed LM call before evaluating whether to stop.
                accounted_tasks.update(done)
                all_responses.extend(task.result() for task in done)

                if len(all_responses) >= 2 and len(all_responses) < budget:
                    eligible_indices, projected = self._project_responses(all_responses)

                    if len(eligible_indices) >= 2:
                        counts = Counter(projected)
                        most_common = counts.most_common(2)
                        v1 = most_common[0][1]
                        v2 = most_common[1][1] if len(most_common) > 1 else 0

                        prob = self.beta_stopping_probability(v1, v2)

                        if prob >= self.confidence_threshold:
                            logger.info(
                                "Early stop: %d/%d samples, "
                                "P(majority stays)=%.4f >= %.4f",
                                len(all_responses),
                                budget,
                                prob,
                                self.confidence_threshold,
                            )
                            stopped_early = True
                            break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        if stopped_early:
            all_responses.extend(
                outcome
                for task, outcome in zip(tasks, outcomes)
                if task not in accounted_tasks
                and not isinstance(outcome, BaseException)
            )
        else:
            logger.info(
                "BetaSelfConsistency used all %d/%d samples",
                len(all_responses),
                budget,
            )

        return self._process_responses(all_responses, return_response_only, usage)
