import asyncio
import contextlib
import logging
import threading

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    ChatMessage,
    GenerationUsage,
)
from its_hub.core.utils import resolve_max_completion_tokens


class _ThreadSafeAsyncSemaphore:
    """A semaphore that works across event loops and threads.

    Uses a threading.Semaphore as the source of truth so the concurrency
    limit is respected across event loops and threads, and wraps acquire/release for use in
    async contexts without blocking the event loop.

    FIXME: When a task wants to acquire the semaphore, it submits self._sem.acquire (a blocking call)
    to the default ThreadPoolExecutor. The default thread pool can be exhausted when max_concurrency
    is low (e.g., 2-4) with large batches. Our default max_concurrency=32 should help avoid the issue
    but needs further investigation, if necessary.
    """

    def __init__(self, value: int):
        self._sem = threading.Semaphore(value)

    async def acquire(self):
        loop = asyncio.get_running_loop()
        # Run blocking acquire in the default executor so the event loop
        # stays responsive while waiting for a slot.
        await loop.run_in_executor(None, self._sem.acquire)

    def release(self):
        self._sem.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class LMOrchestrator(AbstractOrchestrator):
    """
    LMOrchestrator is the inline implementation for managing parallel execution
    of language model requests, handling concurrency limits and batching strategies.

    The concurrency limit is enforced across event loops and threads
    using a thread-safe semaphore.
    """

    def __init__(self, max_concurrency: int = 32):
        if max_concurrency < -1 or max_concurrency == 0:
            raise ValueError(
                "max_concurrency must be -1 (unlimited concurrency) or a positive integer"
            )

        self.max_concurrency = max_concurrency
        self._semaphore: _ThreadSafeAsyncSemaphore | None = (
            _ThreadSafeAsyncSemaphore(max_concurrency)
            if max_concurrency != -1
            else None
        )

    async def agenerate(
        self,
        lm: AbstractLanguageModel,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        usage_accumulator: GenerationUsage | None = None,
    ) -> list[dict]:
        """
        Generate responses for a batch of messages asynchronously.

        Args:
            lm: Language model to use for generation
            messages_lst: List of conversations to process
            stop: (Optional) Stop sequence for generation
            max_completion_tokens: (Optional) Maximum tokens to generate per response
            temperature: (Optional) Temperature value(s) for sampling. Can be single float or list of floats
            include_stop_str_in_output: (Optional) Whether to include stop string in output (vLLM only)
            tools: (Optional) List of available tools
            tool_choice: (Optional) Tool choice mode
            response_format: (Optional) Response format specification for structured outputs

        Returns:
            List of response dicts in the same order as messages_lst
        """

        max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, max_tokens
        )

        if not messages_lst:
            return []

        logging.debug(
            "LMOrchestrator: Processing batch of %d messages", len(messages_lst)
        )

        # Prepare temperature list
        temperature_list = (
            temperature
            if isinstance(temperature, list)
            else [temperature] * len(messages_lst)
        )

        current_loop = asyncio.get_running_loop()

        async def _gen_coro(messages, temp):
            ctx = (
                self._semaphore
                if self._semaphore is not None
                else contextlib.nullcontext()
            )
            async with ctx:
                return await lm.agenerate_single(
                    messages,
                    stop=stop,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temp,
                    include_stop_str_in_output=include_stop_str_in_output,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    loop=current_loop,
                    usage_accumulator=usage_accumulator,
                )

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(_gen_coro(msgs, temp))
                    for msgs, temp in zip(messages_lst, temperature_list)
                ]
        except ExceptionGroup as eg:
            cancelled = sum(1 for t in tasks if t.cancelled())
            error_types = {}
            for exc in eg.exceptions:
                name = type(exc).__name__
                error_types[name] = error_types.get(name, 0) + 1
            summary = ", ".join(f"{v}x {k}" for k, v in error_types.items())
            msg = (
                f"LMOrchestrator: {len(eg.exceptions)} error(s), "
                f"{cancelled} cancelled out of {len(messages_lst)} "
                f"generation(s) ({summary})"
            )
            raise RuntimeError(msg) from eg

        # Collect results in order
        responses = [task.result() for task in tasks]

        logging.debug("LMOrchestrator: Completed batch generation")

        return responses
