from abc import ABC, abstractmethod

from its_hub.api.types import ChatMessage, ChatMessages


class AbstractProcessRewardModel(ABC):
    """
    Abstract base class for process reward models.

    This class supports process reward models that evaluate steps.
    """

    @abstractmethod
    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """
        Score steps synchronously.

        Args:
            prompt_or_messages: Single conversation or batch of conversations
                Single: list[dict] (one conversation)
                Batch: list[list[dict]] (multiple conversations)
            steps: Intermediate steps

        Returns:
            List of scores (list[float])
            Higher score = better response
        """
        pass

    @abstractmethod
    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """
        Score steps asynchronously.

        Args:
            prompt_or_messages: Single conversation or batch of conversations
                Single: list[dict] (one conversation)
                Batch: list[list[dict]] (multiple conversations)
            steps: Intermediate steps
        """
        pass
