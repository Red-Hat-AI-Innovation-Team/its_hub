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

        Args:
            messages: Single conversation or batch of conversations
            stop: Optional stop sequence for generation
            **kwargs: Additional model-specific parameters (tools, tool_choice, etc.)

        Returns:
            Single response dict or list of response dicts (for batched input)
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
