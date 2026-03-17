import asyncio
from abc import ABC, abstractmethod

from its_hub.api.lm import AbstractLanguageModel
from its_hub.api.types import ChatMessage


class AbstractOrchestrator(ABC):
    """
    Abstract base class for orchestration of LM calls.

    Orchestrator will manage parallel execution of language model requests,
    handling concurrency limits and batching strategies.
    """

    @abstractmethod
    async def agenerate(
        self,
        lm: AbstractLanguageModel,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[dict]:
        """
        Generate responses for a batch of messages asynchronously.

        Args:
            lm: Language model to use for generation
            messages_lst: List of conversations to process
            stop: (Optional) Stop sequence for generation
            max_tokens: (Optional) Maximum tokens to generate per response
            temperature: (Optional) Temperature value(s) for sampling. Can be single float or list of floats
            include_stop_str_in_output: (Optional) Whether to include stop string in output (vLLM only)
            tools: (Optional) Ist of available tools
            tool_choice: (Optional) Tool choice mode

        Returns:
            List of response dicts in the same order as messages_lst
        """
        pass

    def generate(
        self,
        lm: AbstractLanguageModel,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[dict]:
        """
        Synchronous wrapper for agenerate. Runs async generation in event loop.

        Args:
            Same as agenerate

        Returns:
            List of response dicts in the same order as messages_lst
        """
        return asyncio.run(
            self.agenerate(
                lm,
                messages_lst,
                stop=stop,
                max_tokens=max_tokens,
                temperature=temperature,
                include_stop_str_in_output=include_stop_str_in_output,
                tools=tools,
                tool_choice=tool_choice,
            )
        )
