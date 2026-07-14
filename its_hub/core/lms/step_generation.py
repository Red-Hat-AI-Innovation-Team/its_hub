import asyncio
import logging

from its_hub.api import AbstractLanguageModel, ChatMessage
from its_hub.core.utils import extract_content_from_lm_response


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

    async def aforward(
        self,
        lm: AbstractLanguageModel,
        prompt_or_prompts: str | list[str],
        steps_so_far: list[str] | list[list[str]] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, bool] | list[tuple[str, bool]]:
        """generate next step(s) asynchronously"""
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
            next_step_response = await lm.agenerate(
                messages,
                stop=self.step_token,
                max_completion_tokens=self.tokens_per_step,
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
            next_steps_responses = await lm.agenerate(
                messages_lst,
                stop=self.step_token,
                max_completion_tokens=self.tokens_per_step,
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

    def forward(
        self,
        lm: AbstractLanguageModel,
        prompt_or_prompts: str | list[str],
        steps_so_far: list[str] | list[list[str]] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> tuple[str, bool] | list[tuple[str, bool]]:
        """generate next step(s) synchronously"""
        return asyncio.run(
            self.aforward(lm, prompt_or_prompts, steps_so_far, tools, tool_choice)
        )
