from abc import ABC, abstractmethod

from .types import ChatMessage


class AbstractLanguageModel(ABC):
    """abstract base class for (autoregressive) language models"""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """
        generate a response from the model

        Args:
            messages: the input messages

        Returns:
            the generated output string
        """
        pass

    @abstractmethod
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """
        evaluate the likelihoods of the generation given the prompt

        Args:
            prompt: the input prompt
            generation: the generated output string

        Returns:
            the likelihoods of the generation per token
        """
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
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | AbstractScalingResult:
        """
        run inference with the given language model and prompt under the specified budget

        Args:
            lm: a language model that takes a prompt and returns a response
            prompt: the input prompt
            budget: the computational budget for inference
            return_response_only: whether to return only the selected response

        Returns:
            the generated output string or the complete scaling result
        """
        pass


class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    def score(self, prompt: str, response: str) -> float:
        """the score for a given prompt and response"""
        pass


# TODO(GX) deal with aggregation of PRM scores somehow in a common place, e.g. here
class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    def score(self, prompt: str, steps: list[str]) -> list[float]:
        """the score for a given prompt and steps"""
        pass
