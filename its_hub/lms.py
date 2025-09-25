import asyncio
import logging
import ssl

import aiohttp
import backoff
import certifi
import requests

from .base import AbstractLanguageModel
from .error_handling import (
    RETRYABLE_ERRORS,
    APIError,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)
from .types import ChatMessage
from .utils import extract_content_from_lm_response


def rstrip_iff_entire(s: str, subs: str) -> str:
    if s.endswith(subs):
        # If s ends with subs, return the string without the length of subs at the end
        return s[: -len(subs)]
    else:
        # Otherwise, return the original string
        return s


# TODO make it robust such that one of the particle dead (e.g. due to max tokens), the whole generation is not stopped
# TODO change stop_token to be a function called is_stopped
class StepGeneration:
    def __init__(
        self,
        max_steps: int,
        step_token: str | list[str] | None = None,
        stop_token: str | None = None,
        temperature: float = 0.8,
        include_stop_str_in_output: bool = False,  # If True, keep stop strings in output; if False, strip them
        temperature_switch: tuple[float, str, str]
        | None = None,  # (temperature, open_token, close_token)
        tokens_per_step: int
        | None = None,  # Maximum tokens per step when step_token is None
    ):
        # Validate that exactly one of step_token or tokens_per_step is set
        if step_token is None and tokens_per_step is None:
            raise ValueError("Either step_token or tokens_per_step must be provided")
        if step_token is not None and tokens_per_step is not None:
            raise ValueError("Cannot specify both step_token and tokens_per_step")

        if step_token is not None and not include_stop_str_in_output:
            assert isinstance(step_token, str), (
                "step_token must be a string if include_stop_str_in_output is False"
            )

        if tokens_per_step is not None and tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be a positive integer")

        self.step_token = step_token
        self.tokens_per_step = tokens_per_step
        self.max_steps = max_steps
        self.stop_token = stop_token
        self.temperature = temperature
        self.include_stop_str_in_output = include_stop_str_in_output
        self.temperature_switch = temperature_switch

    def _post_process(self, steps: list[str], stopped: bool = False) -> str:
        if self.include_stop_str_in_output:
            if stopped and self.stop_token is not None:
                last_step = steps[-1]
                last_step = rstrip_iff_entire(last_step, self.stop_token)
                steps = [*steps[:-1], last_step]
            return "".join(steps)
        else:
            if self.tokens_per_step is not None:
                # Using tokens_per_step: simply concatenate all steps
                response = "".join(steps)
            elif isinstance(self.step_token, str):
                response = self.step_token.join(steps)
            else:
                response = "".join(steps)
            if not stopped and isinstance(self.step_token, str):
                response += self.step_token
            return response

    def _get_temperature(
        self, messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]]
    ) -> float | list[float]:
        if self.temperature_switch is None:
            return self.temperature
        else:
            is_single = isinstance(messages_or_messages_lst[0], ChatMessage)
            if is_single:
                messages = messages_or_messages_lst
                if (
                    isinstance(messages, list)
                    and len(messages) > 0
                    and hasattr(messages[-1], "role")
                    and messages[-1].role == "assistant"
                ):
                    temperature, open_token, close_token = self.temperature_switch
                    if (
                        hasattr(messages[-1], "content")
                        and messages[-1].content is not None
                        and open_token in messages[-1].content
                        and close_token not in messages[-1].content
                    ):
                        return temperature
                    else:
                        return self.temperature
                else:
                    return self.temperature
            else:
                return [
                    self._get_temperature(messages)
                    for messages in messages_or_messages_lst
                ]

    def forward(
        self,
        lm: AbstractLanguageModel,
        prompt_or_prompts: str | list[str],
        steps_so_far: list[str] | list[list[str]] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, bool] | list[tuple[str, bool]]:
        if steps_so_far is None:
            steps_so_far = []
        is_single_prompt = isinstance(prompt_or_prompts, str)
        if is_single_prompt:
            prompt = prompt_or_prompts
            current_step = len(steps_so_far) + 1
            logging.info("Generating step %s/%s", current_step, self.max_steps)

            messages = [
                ChatMessage(role="user", content=prompt),
            ]
            if steps_so_far:
                messages.append(
                    ChatMessage(
                        role="assistant", content=self._post_process(steps_so_far)
                    )
                )
            next_step_response = lm.generate(
                messages,
                stop=self.step_token,
                max_tokens=self.tokens_per_step,
                temperature=self._get_temperature(messages),
                include_stop_str_in_output=self.include_stop_str_in_output,
                tools=tools,
                tool_choice=tool_choice,
            )
            next_step = extract_content_from_lm_response(next_step_response)
            is_stopped = len(steps_so_far) >= self.max_steps
            if self.stop_token:
                is_stopped = is_stopped or self.stop_token in next_step
            return next_step, is_stopped
        else:
            prompts = prompt_or_prompts
            step_numbers = [
                len(steps_so_far_per_prompt) + 1
                for steps_so_far_per_prompt in steps_so_far
            ]
            logging.info(
                "Generating steps (batch): %s / %s", step_numbers, self.max_steps
            )

            messages_lst = []
            for prompt, steps_so_far_per_prompt in zip(prompts, steps_so_far):
                messages = [
                    ChatMessage(role="user", content=prompt),
                ]
                if steps_so_far_per_prompt:
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=self._post_process(steps_so_far_per_prompt),
                        )
                    )
                messages_lst.append(messages)
            next_steps_responses = lm.generate(
                messages_lst,
                stop=self.step_token,
                max_tokens=self.tokens_per_step,
                temperature=self._get_temperature(messages_lst),
                include_stop_str_in_output=self.include_stop_str_in_output,
                tools=tools,
                tool_choice=tool_choice,
            )
            next_steps = [
                extract_content_from_lm_response(r) for r in next_steps_responses
            ]
            is_stopped = [
                len(steps_so_far_per_prompt) >= self.max_steps
                for steps_so_far_per_prompt in steps_so_far
            ]
            if self.stop_token:
                is_stopped = [
                    is_stopped_per_prompt or self.stop_token in next_step
                    for is_stopped_per_prompt, next_step in zip(is_stopped, next_steps)
                ]
            return list(zip(next_steps, is_stopped))


class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str,
        system_prompt: str | None = None,
        is_async: bool = False,
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

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
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

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _prepare_request_data(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
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

        return request_data

    async def _generate(
        self,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
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

    def generate(
        self,
        messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | list[dict]:
        # Check if we have a single list of messages or a list of message lists
        # Single list: [{"role": "user", "content": "..."}] or [Message(...)]
        # Multiple lists: [[{"role": "user", "content": "..."}], [{"role": "user", "content": "..."}]]
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )
        if self.is_async:
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # Run async code in a new thread to avoid "event loop already running" error
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    response_or_responses = executor.submit(
                        lambda: asyncio.run(self._generate(
                            messages_lst, stop, max_tokens, temperature,
                            include_stop_str_in_output, tools, tool_choice
                        ))
                    ).result()
            except RuntimeError:
                # No running loop, safe to use run_until_complete
                response_or_responses = asyncio.get_event_loop().run_until_complete(
                    self._generate(messages_lst, stop, max_tokens, temperature,
                                 include_stop_str_in_output, tools, tool_choice)
                )
        else:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            def fetch_single_response(
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
                )

                response = requests.post(
                    self._chat_completion_endpoint,
                    headers=self.headers,
                    json=request_data,
                    verify=self.verify_ssl,
                )

                if response.status_code != 200:
                    api_error = parse_api_error(response.status_code, response.text)
                    if not should_retry(api_error):
                        logging.error(format_non_retryable_error(api_error))
                    raise api_error

                response_json = response.json()
                # Return the full message object to preserve tool calls
                return response_json["choices"][0]["message"]

            def safe_fetch_single_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                if self.replace_error_with_message is not None:
                    try:
                        return fetch_single_response(messages, _temperature)
                    except requests.RequestException as e:
                        logging.error(f"Network error during sync generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                    except APIError as e:
                        logging.error(f"API error during sync generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                else:
                    return fetch_single_response(messages, _temperature)

            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            response_or_responses = [
                safe_fetch_single_response(messages, _temperature)
                for messages, _temperature in zip(messages_lst, temperature_lst)
            ]
        return response_or_responses[0] if is_single else response_or_responses

    async def agenerate(
        self,
        messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | list[dict]:
        """Async version of generate method for use in async contexts."""
        # Check if we have a single list of messages or a list of message lists
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )

        # Always use async implementation for agenerate
        response_or_responses = await self._generate(
            messages_lst,
            stop,
            max_tokens,
            temperature,
            include_stop_str_in_output,
            tools,
            tool_choice,
        )

        return response_or_responses[0] if is_single else response_or_responses

    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        raise NotImplementedError("evaluate method not implemented")


# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass


# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass
