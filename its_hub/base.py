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

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """evaluate the likelihoods of the generation synchronously"""
        raise NotImplementedError("evaluate method not implemented")


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
        """
        Run inference asynchronously with the given language model and prompt.

        This is the primary method that subclasses must implement.
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
    ) -> str | AbstractScalingResult:
        """
        Run inference synchronously with the given language model and prompt.

        Default implementation wraps ainfer() using asyncio.run().
        """
        import asyncio

        return asyncio.run(
            self.ainfer(
                lm, prompt_or_messages, budget, return_response_only, tools, tool_choice
            )
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


class AbstractTrajectoryAggregator(ABC):
    """abstract base class for trajectory aggregators

    Reduces a sequence of per-step PRM scores to a single trajectory score.
    Pluggable into ParticleFiltering and BeamSearch via the aggregator parameter.
    """

    @abstractmethod
    def aggregate(self, step_scores: list[float]) -> float:
        """reduce per-step scores to a single trajectory score"""
        pass

    async def aaggregate(self, step_scores: list[float]) -> float:
        """async variant; delegates to sync by default"""
        return self.aggregate(step_scores)
