from abc import ABC, abstractmethod

from its_hub.api.orchestrator import AbstractOrchestrator
from its_hub.api.types import ChatMessage, ChatMessages


class AbstractOutcomeRewardModel(ABC):
    """
    Abstract base class for outcome reward models and judge models.

    This class supports both traditional reward models and LLM-based judge models
    that evaluate conversation outcomes and quality.
    """

    @abstractmethod
    def score(
        self,
        messages: list[ChatMessage] | ChatMessages,
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations synchronously.

        Args:
            messages: Single conversation or batch of conversations
                Single: list[dict] (one conversation)
                Batch: list[list[dict]] (multiple conversations)
            **kwargs: Additional model-specific parameters

        Returns:
            Single score (float) or list of scores (list[float])
            Higher score = better response
        """
        pass

    async def ascore(
        self,
        messages: list[ChatMessage] | ChatMessages,
        orchestrator: AbstractOrchestrator,
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations asynchronously.

        Default implementation raises NotImplementedError.
        Override this for async-compatible reward models (e.g., LLM judges).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement async scoring. "
            "Override ascore() to support async scoring."
        )
