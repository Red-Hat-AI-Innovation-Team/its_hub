import asyncio
import logging

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    ChatMessage,
)


class LMOrchestrator(AbstractOrchestrator):
    """
    LMOrchestrator is the inline implementation for managing parallel execution
    of language model requests, handling concurrency limits and batching strategies.

    This class implements the Singleton pattern to ensure only one orchestrator
    instance exists throughout the application lifetime.
    """

    __instance = None
    __initialized = False

    def __new__(cls, max_concurrency: int = 32):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, max_concurrency: int = 32):
        if self.__initialized:
            return

        assert max_concurrency == -1 or max_concurrency > 0, (
            "max_concurrency must be -1 (unlimited concurrency) or a positive integer"
        )

        self.max_concurrency = max_concurrency
        self._semaphore = None if max_concurrency == -1 else asyncio.Semaphore(max_concurrency)
        self.__initialized = True

    async def agenerate(
        self,
        lm: AbstractLanguageModel,
        messages_batch: list[list[ChatMessage]],
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

        if not messages_batch:
            return []

        logging.info(
            "LMOrchestrator: Processing batch of %d messages",
            len(messages_batch)
        )

        is_single = not isinstance(messages_batch[0], list)
        messages_lst = (
            [messages_batch] if is_single else messages_batch
        )

        # Prepare temperature list
        temperature_list = (
            temperature if isinstance(temperature, list)
            else [temperature] * len(messages_lst)
        )

        async def _gen_coro(messages, temp):
            if self._semaphore is None:
                return await lm.agenerate_single(
                    messages,
                    stop=stop,
                    max_tokens=max_tokens,
                    temperature=temp,
                    include_stop_str_in_output=include_stop_str_in_output,
                    tools=tools,
                    tool_choice=tool_choice,
                )
            else:
                async with self._semaphore:
                    return await lm.agenerate_single(
                        messages,
                        stop=stop,
                        max_tokens=max_tokens,
                        temperature=temp,
                        include_stop_str_in_output=include_stop_str_in_output,
                        tools=tools,
                        tool_choice=tool_choice,
                    )

        responses = []
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(_gen_coro(msgs, temp))
                for msgs, temp in zip(messages_lst, temperature_list)
            ]

        # Collect results in order
        responses = [task.result() for task in tasks]

        # Close LM session after all requests complete
        if hasattr(lm, 'close'):
            await lm.close()

        logging.info("LMOrchestrator: Completed batch generation")

        return responses
