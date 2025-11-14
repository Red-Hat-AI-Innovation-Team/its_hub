from abc import ABC, abstractmethod

from .types import ChatMessage, ChatMessages


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
    ) -> dict | list[dict]:
        """
        Generate response(s) asynchronously.

        Args:
            messages: Single conversation or batch of conversations
            stop: Optional stop sequence for generation

        Returns:
            Single response dict or list of response dicts (for batched input)
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass


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

        This is the primary method that subclasses must implement.

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
        Subclasses can override for custom sync implementations.

        Args:
            Same as ainfer()

        Returns:
            Same as ainfer()
        """
        import asyncio
        return asyncio.run(
            self.ainfer(
                lm, prompt_or_messages, budget, return_response_only, tools, tool_choice
            )
        )


class AbstractOutcomeRewardModel(ABC):
    """
    Abstract base class for outcome reward models and judge models.

    This class supports both traditional reward models and LLM-based judge models
    that evaluate conversation outcomes and quality.
    """

    @abstractmethod
    def score(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score responses/conversations using the OpenAI chat completion format.

        Args:
            messages: Either a single conversation (list[dict]) or multiple conversations (list[list[dict]])
                     Each message dict format: {"role": "user/assistant", "content": "...", "tool_calls": [...]}
            **kwargs: Additional parameters (e.g., max_input_tokens, top_n, return_judge_reasoning, etc.)

        Returns:
            For single conversation: float (single score)
            For multiple conversations: list[float] (list of scores)
        """
        pass

    async def ascore(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Async version of score method.

        Default implementation raises NotImplementedError. Subclasses that support async scoring
        (e.g., LLM judges) should override this method.

        Args:
            messages: Either a single conversation (list[dict]) or multiple conversations (list[list[dict]])
            **kwargs: Additional parameters (e.g., return_judge_reasoning, etc.)

        Returns:
            For single conversation: float (single score)
            For multiple conversations: list[float] (list of scores)

        Raises:
            NotImplementedError: If the subclass does not implement async scoring
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async scoring. "
            f"Use the synchronous score() method instead."
        )


# TODO(GX) deal with aggregation of PRM scores somehow in a common place, e.g. here
class AbstractProcessRewardModel(ABC):
    """
    Abstract base class for process reward models.

    Process reward models score step-by-step reasoning (e.g., math solution steps).
    Used by experimental algorithms: Beam Search, Particle Filtering.

    NOTE: This is not implemented in the MVP. For production use, see experimental extras.
    """

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """Score reasoning steps asynchronously."""
        raise NotImplementedError(
            "Process reward models are not implemented in MVP. "
            "Install with 'pip install its_hub[experimental]' for step-wise algorithms."
        )

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """Score reasoning steps synchronously."""
        raise NotImplementedError(
            "Process reward models are not implemented in MVP. "
            "Install with 'pip install its_hub[experimental]' for step-wise algorithms."
        )
