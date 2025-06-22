import asyncio
import logging

import aiohttp
import backoff
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


def rstrip_iff_entire(s, subs):
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
        step_token: str | list[str],
        max_steps: int,
        stop_token: str | None = None,
        temperature: float = 0.8,
        include_stop_str_in_output: bool = False,  # If True, keep stop strings in output; if False, strip them
        temperature_switch: tuple[float, str, str]
        | None = None,  # (temperature, open_token, close_token)
    ):
        if not include_stop_str_in_output:
            assert isinstance(step_token, str), (
                "step_token must be a string if include_stop_str_in_output is False"
            )
        else:
            assert step_token is not None, (
                "step_token must be provided if include_stop_str_in_output is True"
            )
        self.step_token = step_token
        self.max_steps = max_steps
        self.stop_token = stop_token
        self.temperature = temperature
        self.include_stop_str_in_output = include_stop_str_in_output
        self.temperature_switch = temperature_switch

    def _post_process(self, steps: list[str], stopped: bool = False) -> str:
        if self.include_stop_str_in_output:
            if stopped:
                last_step = steps[-1]
                last_step = rstrip_iff_entire(last_step, self.stop_token)
                steps = [*steps[:-1], last_step]
            return "".join(steps)
        else:
            if isinstance(self.step_token, str):
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
            next_step = lm.generate(
                messages,
                stop=self.step_token,
                temperature=self._get_temperature(messages),
                include_stop_str_in_output=self.include_stop_str_in_output,
            )
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
            next_steps = lm.generate(
                messages_lst,
                stop=self.step_token,
                temperature=self._get_temperature(messages_lst),
                include_stop_str_in_output=self.include_stop_str_in_output,
            )
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

        # set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _prepare_request_data(
        self,
        messages,
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
    ):
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
            "messages": [msg.__dict__ for msg in messages],
            "extra_body": {},
        }
        if messages[-1].role == "assistant":
            request_data["extra_body"]["add_generation_prompt"] = False
            request_data["extra_body"]["continue_final_message"] = True
            request_data["add_generation_prompt"] = False
            request_data["continue_final_message"] = True

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
        if include_stop_str_in_output is not None:
            request_data["extra_body"]["include_stop_str_in_output"] = (
                include_stop_str_in_output
            )
            request_data["include_stop_str_in_output"] = include_stop_str_in_output

        return request_data

    async def _generate(
        self,
        messages_lst,
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
    ) -> list[str]:
        # limit concurrency to max_concurrency using a semaphore
        semaphore = asyncio.Semaphore(
            len(messages_lst) if self.max_concurrency == -1 else self.max_concurrency
        )

        # create a single session for all requests in this call
        async with aiohttp.ClientSession() as session:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            async def fetch_response(messages, _temperature):
                async with semaphore:
                    request_data = self._prepare_request_data(
                        messages,
                        stop,
                        max_tokens,
                        _temperature,
                        include_stop_str_in_output,
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
                        return response_json["choices"][0]["message"]["content"]

            async def safe_fetch_response(messages, _temperature):
                if self.replace_error_with_message is not None:
                    try:
                        return await fetch_response(messages, _temperature)
                    except (aiohttp.ClientError, TimeoutError) as e:
                        logging.error(f"Network error during async generation: {e}")
                        return self.replace_error_with_message
                    except APIError as e:
                        logging.error(f"API error during async generation: {e}")
                        return self.replace_error_with_message
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
        messages_or_messages_lst,
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool
        | None = None,  # If True, keep stop strings in generated text; if False, strip them
    ) -> str | list[str]:
        # Check if we have a single list of messages or a list of message lists
        # Single list: [{"role": "user", "content": "..."}] or [Message(...)]
        # Multiple lists: [[{"role": "user", "content": "..."}], [{"role": "user", "content": "..."}]]
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )
        if self.is_async:
            loop = asyncio.get_event_loop()
            response_or_responses = loop.run_until_complete(
                self._generate(
                    messages_lst,
                    stop,
                    max_tokens,
                    temperature,
                    include_stop_str_in_output,
                )
            )
        else:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            def fetch_single_response(messages, _temperature):
                request_data = self._prepare_request_data(
                    messages, stop, max_tokens, _temperature, include_stop_str_in_output
                )

                response = requests.post(
                    self._chat_completion_endpoint,
                    headers=self.headers,
                    json=request_data,
                )

                if response.status_code != 200:
                    api_error = parse_api_error(response.status_code, response.text)
                    if not should_retry(api_error):
                        logging.error(format_non_retryable_error(api_error))
                    raise api_error

                response_json = response.json()
                return response_json["choices"][0]["message"]["content"]

            def safe_fetch_single_response(messages, _temperature):
                if self.replace_error_with_message is not None:
                    try:
                        return fetch_single_response(messages, _temperature)
                    except requests.RequestException as e:
                        logging.error(f"Network error during sync generation: {e}")
                        return self.replace_error_with_message
                    except APIError as e:
                        logging.error(f"API error during sync generation: {e}")
                        return self.replace_error_with_message
                else:
                    return fetch_single_response(messages, _temperature)

            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            responses = [
                safe_fetch_single_response(messages, _temperature)
                for messages, _temperature in zip(messages_lst, temperature_lst)
            ]
            response_or_responses = responses
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
