from abc import ABC, abstractmethod

from its_hub.api.types import ChatMessage


class AbstractLanguageModel(ABC):
    """
    Abstract base class for language models.

    Gateway integrators should implement this interface to use its_hub algorithms
    with their existing LM infrastructure. Only async implementation is required.
    """

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        **kwargs,
    ) -> dict | list[dict]:
        """
        Generate response(s) asynchronously.
        Batch processing has been moved to orchestrator. This method will be deprecated
        in favor of agenerate_single once all algorithms have been moved to using orchestrator.

        Args:
            messages: Single conversation or batch of conversations
            stop: Optional stop sequence for generation
            **kwargs: Additional model-specific parameters (tools, tool_choice, etc.)

        Returns:
            Single response dict or list of response dicts (for batched input)
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass

    @abstractmethod
    async def agenerate_single(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        **kwargs,
    ) -> dict | list[dict]:
        """
        Generate response asynchronously.

        Args:
            messages: Single conversation
            stop: Optional stop sequence for generation
            **kwargs: Additional model-specific parameters (tools, tool_choice, etc.)

        Returns:
            Single response dict
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
