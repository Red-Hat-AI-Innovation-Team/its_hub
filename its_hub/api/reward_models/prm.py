from abc import ABC, abstractmethod

from its_hub.api.types import ChatMessage, ChatMessages


class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """score steps asynchronously"""
        pass

    @abstractmethod
    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """score steps synchronously"""
        pass
