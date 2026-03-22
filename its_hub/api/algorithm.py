import asyncio
from abc import ABC, abstractmethod

from its_hub.api.lm import AbstractLanguageModel
from its_hub.api.types import ChatMessage, ChatMessages


class AbstractScalingResult(ABC):
    """
    Abstract base class for algorithm results.

    Algorithms return instances of this class when return_response_only=False.
    """

    @property
    @abstractmethod
    def the_one(self) -> dict:
        """
        Return the selected best response.

        Returns:
            The response message dict selected by the algorithm
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass


class AbstractScalingAlgorithm(ABC):
    """
    Abstract base class for inference-time scaling algorithms.

    All algorithms (Self-Consistency, Best-of-N, etc.) implement this interface.
    """

    @abstractmethod
    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | AbstractScalingResult:
        """
        Run inference asynchronously with the given language model and prompt.

        Args:
            lm: Language model instance implementing AbstractLanguageModel
            prompt_or_messages: User prompt (string or structured messages)
            budget: Computational budget (interpretation varies by algorithm)
            return_response_only: If True, return just the selected response;
                                   if False, return full result object
            tools: Optional OpenAI-style tool definitions
            tool_choice: Optional tool choice strategy ("auto", "none", or specific tool)

        Returns:
            Selected response dict (if return_response_only=True) or
            AbstractScalingResult instance with full details
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | AbstractScalingResult:
        """
        Run inference synchronously with the given language model and prompt.

        Default implementation wraps ainfer() using asyncio.run().
        """
        async def _run():
            try:
                return await self.ainfer(
                    lm, prompt_or_messages, budget, return_response_only, tools, tool_choice
                )
            finally:
                if hasattr(lm, 'close'):
                    await lm.close()

        return asyncio.run(_run())
