from abc import ABC, abstractmethod

from .types import ChatMessage, ChatMessages


class AbstractLanguageModel(ABC):
    """abstract base class for (autoregressive) language models"""

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """generate a response from the model asynchronously"""
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """generate a response from the model synchronously"""
        pass

    @abstractmethod
    async def aevaluate(self, prompt: str, generation: str) -> list[float]:
        """evaluate the likelihoods of the generation asynchronously"""
        pass

    @abstractmethod
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """evaluate the likelihoods of the generation synchronously"""
        pass


class AbstractScalingResult(ABC):
    """abstract base class for scaling result"""

    @property
    @abstractmethod
    def the_one(self) -> str:
        """the selected response"""
        pass


class AbstractScalingAlgorithm(ABC):
    """abstract base class for inference-time scaling algorithms"""

    @abstractmethod
    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | AbstractScalingResult:
        """run inference asynchronously with the given language model and prompt"""
        pass

    @abstractmethod
    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | AbstractScalingResult:
        """run inference synchronously with the given language model and prompt"""
        pass

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> str | AbstractScalingResult:
        """
        async version of infer - default implementation falls back to sync version

        Subclasses should override this for true async performance
        """
        import asyncio
        return await asyncio.to_thread(
            self.infer, lm, prompt_or_messages, budget, return_response_only, tools, tool_choice
        )


class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    async def ascore(
        self, prompt_or_messages: str | list[ChatMessage] | ChatMessages, response: str
    ) -> float:
        """score a response asynchronously"""
        pass

    @abstractmethod
    def score(
        self, prompt_or_messages: str | list[ChatMessage] | ChatMessages, response: str
    ) -> float:
        """score a response synchronously"""
        pass


# TODO(GX) deal with aggregation of PRM scores somehow in a common place, e.g. here
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
