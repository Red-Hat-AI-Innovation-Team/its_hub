import asyncio
import logging
import ssl
import threading
import warnings
import weakref

import aiohttp
import backoff
import certifi

from its_hub.api import (
    RETRYABLE_ERRORS,
    AbstractLanguageModel,
    APIError,
    ChatMessage,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)


class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str,
        system_prompt: str | None = None,
        is_async: bool = False,  # Deprecated: parameter is ignored (always async internally)
        # default runtime parameters
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tries: int = 8,
        max_concurrency: int = -1,
        replace_error_with_message: str | None = None,
        # SSL configuration
        verify_ssl: bool = True,
        ssl_context: ssl.SSLContext | None = None,
    ):
        assert max_concurrency == -1 or max_concurrency > 0, (
            "max_concurrency must be -1 (unlimited concurrency) or a positive integer"
        )

        # Warn about deprecated is_async parameter
        if is_async is not False:
            warnings.warn(
                "The 'is_async' parameter is deprecated and will be removed in a future version. "
                "The implementation now always uses async internally. "
                "Sync methods (generate, evaluate) automatically wrap async calls with asyncio.run().",
                DeprecationWarning,
                stacklevel=2,
            )

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        # Keep is_async for backward compatibility but it's no longer used
        self.is_async = is_async
        self.max_tries = max_tries
        self.max_concurrency = max_concurrency
        self.replace_error_with_message = replace_error_with_message

        # runtime parameters
        self.stop = stop
        self.max_tokens = max_tokens
        self.temperature = temperature

        # SSL configuration
        self.verify_ssl = verify_ssl
        if ssl_context is not None:
            self.ssl_context = ssl_context
        elif not verify_ssl:
            # Create an SSL context that doesn't verify certificates
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
        else:
            # For async requests, create SSL context using the same CA bundle as requests
            # This ensures aiohttp uses the same certificates as requests library
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        # set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # endpoint type
        self.endpoint_type = "openai" if "openai" in self.endpoint else "vllm"

        # Session cache: one session per event loop, entry is auto-cleaned via weak references.
        # Session(s) need to be closed via close or close_session function calls.
        self._sessions: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, aiohttp.ClientSession
        ] = weakref.WeakKeyDictionary()
        self._session_lock = threading.Lock()

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _get_session(self, loop: asyncio.AbstractEventLoop) -> aiohttp.ClientSession:
        """Get or create an HTTP session for the given event loop.

        Sessions are cached per event loop and automatically cleaned up
        when the loop is garbage collected.
        """
        with self._session_lock:
            session = self._sessions.get(loop)
            if session is not None and not session.closed:
                return session

            # Create new session for the event loop
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            session = aiohttp.ClientSession(connector=connector)
            self._sessions[loop] = session

            return session

    async def close(self) -> None:
        """Close all cached HTTP sessions."""
        with self._session_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            if not session.closed:
                await session.close()

    async def close_session(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Close the cached session for the given event loop.

        Args:
            loop: Event loop whose session to close. Defaults to the current running loop.
        """
        if loop is None:
            loop = asyncio.get_running_loop()
        with self._session_lock:
            session = self._sessions.get(loop)
        if session and not session.closed:
            await session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures sessions are closed."""
        await self.close()
        return False

    def _prepare_request_data(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> dict:
        # helper method to prepare request data for both sync and async methods
        # Convert dict messages to Message objects if needed
        messages = [
            msg if isinstance(msg, ChatMessage) else ChatMessage(**msg)
            for msg in messages
        ]

        if self.system_prompt:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *messages,
            ]

        request_data = {
            "model": self.model_name,
            "messages": [msg.to_dict() for msg in messages],
        }

        if self.endpoint_type == "vllm":
            request_data["extra_body"] = {}
            if messages[-1].role == "assistant":
                request_data["extra_body"]["add_generation_prompt"] = False
                request_data["extra_body"]["continue_final_message"] = True
                request_data["add_generation_prompt"] = False
                request_data["continue_final_message"] = True
            if include_stop_str_in_output is not None:
                request_data["extra_body"]["include_stop_str_in_output"] = (
                    include_stop_str_in_output
                )
                request_data["include_stop_str_in_output"] = include_stop_str_in_output
        else:
            logging.info(
                "openai endpoint does not support add_generation_prompt, continue_final_message, or include_stop_str_in_output"
            )
            if include_stop_str_in_output is not None:
                logging.warning(
                    "include_stop_str_in_output parameter is not supported with OpenAI endpoints and will be ignored"
                )

        # set default runtime parameters
        if self.stop is not None:
            request_data["stop"] = self.stop
        if self.max_tokens is not None:
            request_data["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            request_data["temperature"] = self.temperature

        # override runtime parameters
        if stop is not None:
            request_data["stop"] = stop
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature

        # add tools and tool_choice if provided
        if tools is not None:
            request_data["tools"] = tools
        if tool_choice is not None:
            request_data["tool_choice"] = tool_choice

        # add response_format for structured outputs
        if response_format is not None:
            request_data["response_format"] = response_format

        return request_data

    async def _agenerate(
        self,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> list[dict]:
        # limit concurrency to max_concurrency using a semaphore
        semaphore = asyncio.Semaphore(
            len(messages_lst) if self.max_concurrency == -1 else self.max_concurrency
        )

        # create a single session for all requests in this call
        # Use the same SSL behavior as requests library
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            async def fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                async with semaphore:
                    request_data = self._prepare_request_data(
                        messages,
                        stop,
                        max_tokens,
                        _temperature,
                        include_stop_str_in_output,
                        tools,
                        tool_choice,
                        response_format,
                    )

                    async with session.post(
                        self._chat_completion_endpoint,
                        headers=self.headers,
                        json=request_data,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            api_error = parse_api_error(response.status, error_text)
                            if not should_retry(api_error):
                                logging.error(format_non_retryable_error(api_error))
                            raise api_error
                        response_json = await response.json()
                        # Return the full message object to preserve tool calls
                        return response_json["choices"][0]["message"]

            async def safe_fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                if self.replace_error_with_message is not None:
                    try:
                        return await fetch_response(messages, _temperature)
                    except (aiohttp.ClientError, TimeoutError) as e:
                        logging.error(f"Network error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                    except APIError as e:
                        logging.error(f"API error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                else:
                    return await fetch_response(messages, _temperature)

            # gather all responses asynchronously, with concurrency limited to max_concurrency
            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            return await asyncio.gather(
                *(
                    safe_fetch_response(messages, _temperature)
                    for messages, _temperature in zip(messages_lst, temperature_lst)
                )
            )

    async def agenerate(
        self,
        messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> dict | list[dict]:
        """
        generate response(s) asynchronously

        FIXME: Batch processing has been moved to orchestrator. This function will be fully
        replaced by agenerate_single once all algorithms have been moved to using orchestrator.
        """

        warnings.warn(
            "agenerate() is deprecated and will be removed in a future version. "
            "Use agenerate_single() with the orchestrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )
        response_or_responses = await self._agenerate(
            messages_lst,
            stop,
            max_tokens,
            temperature,
            include_stop_str_in_output,
            tools,
            tool_choice,
            response_format,
        )
        return response_or_responses[0] if is_single else response_or_responses

    async def agenerate_single(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> dict:
        # Fallback to the current event loop
        if loop is None:
            loop = asyncio.get_running_loop()
        # Get or create session for the event loop
        session = self._get_session(loop)

        @backoff.on_exception(
            backoff.expo,
            RETRYABLE_ERRORS,
            max_tries=self.max_tries,
            on_backoff=enhanced_on_backoff,
            giveup=lambda e: not should_retry(e),
        )
        async def fetch_response(
            messages: list[ChatMessage], _temperature: float | None
        ) -> dict:
            request_data = self._prepare_request_data(
                messages,
                stop,
                max_tokens,
                _temperature,
                include_stop_str_in_output,
                tools,
                tool_choice,
                response_format,
            )

            async with session.post(
                self._chat_completion_endpoint,
                headers=self.headers,
                json=request_data,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    api_error = parse_api_error(response.status, error_text)
                    if not should_retry(api_error):
                        logging.error(format_non_retryable_error(api_error))
                    raise api_error
                response_json = await response.json()
                # Return the full message object to preserve tool calls
                return response_json["choices"][0]["message"]

        async def safe_fetch_response(
            messages: list[ChatMessage], _temperature: float | None
        ) -> dict:
            if self.replace_error_with_message is not None:
                try:
                    return await fetch_response(messages, _temperature)
                except (aiohttp.ClientError, TimeoutError) as e:
                    logging.error(f"Network error during async generation: {e}")
                    return {
                        "role": "assistant",
                        "content": self.replace_error_with_message,
                    }
                except APIError as e:
                    logging.error(f"API error during async generation: {e}")
                    return {
                        "role": "assistant",
                        "content": self.replace_error_with_message,
                    }
            else:
                return await fetch_response(messages, _temperature)

        return await safe_fetch_response(messages, temperature)
