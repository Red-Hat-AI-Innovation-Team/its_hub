from abc import ABC, abstractmethod

from .types import ChatMessage, ChatMessages


class AbstractLanguageModel(ABC):
    """abstract base class for (autoregressive) language models"""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """generate a response from the model synchronously"""
        pass

    @abstractmethod
    async def generate_async(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """generate a response from the model asynchronously"""
        pass

    @abstractmethod
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """evaluate the likelihoods of the generation synchronously"""
        pass

    @abstractmethod
    async def evaluate_async(self, prompt: str, generation: str) -> list[float]:
        """evaluate the likelihoods of the generation asynchronously"""
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

    @abstractmethod
    async def infer_async(
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


class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    def score(
        self, prompt_or_messages: str | list[ChatMessage] | ChatMessages, response: str
    ) -> float:
        """score a response synchronously"""
        pass

    @abstractmethod
    async def score_async(
        self, prompt_or_messages: str | list[ChatMessage] | ChatMessages, response: str
    ) -> float:
        """score a response asynchronously"""
        pass


# TODO(GX) deal with aggregation of PRM scores somehow in a common place, e.g. here
class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """score steps synchronously"""
        pass

    @abstractmethod
    async def score_async(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """score steps asynchronously"""
        pass
