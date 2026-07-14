import asyncio
from abc import ABC, abstractmethod

from its_hub.api.lm import AbstractLanguageModel
from its_hub.api.types import ChatMessage


class AbstractOrchestrator(ABC):
    """
    Abstract base class for orchestration of LM calls.

    Orchestrator will manage parallel execution of language model requests,
    handling concurrency limits and batching strategies.
    """

    @abstractmethod
    async def agenerate(
        self,
        lm: AbstractLanguageModel,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Generate responses for a batch of messages asynchronously.

        Args:
            lm: Language model to use for generation
            messages_lst: List of conversations to process
            stop: (Optional) Stop sequence for generation
            **kwargs: Additional model-specific parameters (max_completion_tokens, temperature,
                      tools, tool_choice, response_format, etc.)

        Returns:
            List of response dicts in the same order as messages_lst
        """
        pass

    def generate(
        self,
        lm: AbstractLanguageModel,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Synchronous wrapper for agenerate. Runs async generation in event loop.

        Args:
            Same as agenerate

        Returns:
            List of response dicts in the same order as messages_lst
        """
        return asyncio.run(
            self.agenerate(
                lm,
                messages_lst,
                stop=stop,
                **kwargs,
            )
        )
