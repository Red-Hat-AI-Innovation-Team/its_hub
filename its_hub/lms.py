from typing import Union, List, Tuple
import asyncio
import backoff
from openai import OpenAI, AsyncOpenAI
from .base import AbstractLanguageModel

def rstrip_iff_entire(s, subs):
  if s.endswith(subs):
    # If s ends with subs, return the string without the length of subs at the end
    return s[:-len(subs)]
  else:
    # Otherwise, return the original string
    return s

# TODO make it robust such that one of the particle dead (e.g. due to max tokens), the whole generation is not stopped
# TODO change stop_token to be a function called is_stopped
class StepGeneration:
    def __init__(self, step_token: Union[str, List[str]], max_steps: int, stop_token: str = None, temperature: float = 0.8, include_stop_str_in_output: bool = False):
        if not include_stop_str_in_output:
            assert isinstance(step_token, str), "step_token must be a string if include_stop_str_in_output is False"
        else:
            assert step_token is not None, "step_token must be provided if include_stop_str_in_output is True"
        self.step_token = step_token
        self.max_steps = max_steps
        self.stop_token = stop_token
        self.temperature = temperature
        self.include_stop_str_in_output = include_stop_str_in_output

    def _post_process(self, steps: str, stopped: bool = False) -> str:
        if self.include_stop_str_in_output:
            return "".join(steps)
        else:
            response = self.step_token.join(steps)
            if not stopped:
                response += self.step_token
            return response
    
    def forward(
        self, 
        lm: AbstractLanguageModel, 
        prompt_or_prompts: Union[str, List[str]], 
        steps_so_far: Union[List[str],List[List[str]]] = []
    ) -> Tuple[str, bool]:
        is_single_prompt = isinstance(prompt_or_prompts, str)
        if is_single_prompt:
            prompt = prompt_or_prompts
            messages = [
                {"role": "user", "content": prompt},
            ]
            if steps_so_far:
                messages.append({"role": "assistant", 
                                 "content": self._post_process(steps_so_far)})
            next_step = lm.generate(
                messages, stop=self.step_token, temperature=self.temperature, include_stop_str_in_output=self.include_stop_str_in_output
            )
            is_stopped = len(steps_so_far) >= self.max_steps
            if self.stop_token:
                is_stopped = is_stopped or self.stop_token in next_step
                if self.include_stop_str_in_output:
                    next_step = rstrip_iff_entire(next_step, self.stop_token)
            return next_step, is_stopped
        else:
            prompts = prompt_or_prompts
            messages_lst = []
            for prompt, steps_so_far_per_prompt in zip(prompts, steps_so_far):
                messages = [
                    {"role": "user", "content": prompt},
                ]
                if steps_so_far_per_prompt:
                    messages.append({"role": "assistant", 
                                     "content": self._post_process(steps_so_far_per_prompt)})
                messages_lst.append(messages)
            next_steps = lm.generate(
                messages_lst, stop=self.step_token, temperature=self.temperature, include_stop_str_in_output=self.include_stop_str_in_output
            )
            is_stopped = [len(steps_so_far_per_prompt) >= self.max_steps
                          for steps_so_far_per_prompt, next_step in zip(steps_so_far, next_steps)]
            if self.stop_token:
                is_stopped = [is_stopped_per_prompt or self.stop_token in next_step
                             for is_stopped_per_prompt, next_step in zip(is_stopped, next_steps)]
                if self.include_stop_str_in_output:
                    next_steps = [rstrip_iff_entire(next_step, self.stop_token) for next_step in next_steps]
            return list(zip(next_steps, is_stopped))

def _on_backoff(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))

class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self, 
        endpoint: str, 
        api_key: str, 
        model_name: str, 
        system_prompt: str = None, 
        is_async: bool = False,
        # default runtime parameters
        stop: str = None,
        max_tokens: int = None,
        temperature: float = None,
        max_tries: int = 8,
        max_concurrency: int = 8,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.is_async = is_async
        self.max_tries = max_tries
        self.max_concurrency = max_concurrency

        # runtime parameters
        self.stop = stop
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # set up openai clients for sync and async
        if self.is_async:
            self._openai_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.endpoint.rstrip("/"),
            )
        else:
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.endpoint.rstrip("/"),
            )

    @property
    def _chat_completion_endpoint(self) -> str:
        # not used with openai client, but kept for compatibility
        return self.endpoint.rstrip("/") + "/chat/completions"
    
    def _prepare_request_data(
        self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None
):
        # helper method to prepare request data for both sync and async methods
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": {},
        }
        if "assistant" == messages[-1]["role"]:
            request_data["extra_body"]["add_generation_prompt"] = False
            request_data["extra_body"]["continue_final_message"] = True
        
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
            request_data["extra_body"]["include_stop_str_in_output"] = include_stop_str_in_output
        
        return request_data

    async def _generate(
        self, messages_lst, stop: str = None, max_tokens: int = None, temperature: float = None, include_stop_str_in_output: bool = None
    ) -> List[str]:
        # use openai's async client for batch requests
        # limit concurrency to max_concurrency using a semaphore
        semaphore = asyncio.Semaphore(self.max_concurrency)

        @backoff.on_exception(backoff.expo, Exception, max_tries=self.max_tries, on_backoff=_on_backoff)
        async def fetch_response(messages):
            async with semaphore:
                request_data = self._prepare_request_data(
                    messages, stop, max_tokens, temperature, include_stop_str_in_output
                )
                response = await self._openai_client.chat.completions.create(
                    model=request_data["model"],
                    messages=request_data["messages"],
                    stop=request_data.get("stop"),
                    max_tokens=request_data.get("max_tokens"),
                    temperature=request_data.get("temperature"),
                    extra_body=request_data.get("extra_body", {}),
                )
                return response.choices[0].message.content

        # gather all responses asynchronously, with concurrency limited to max_concurrency
        return await asyncio.gather(*(fetch_response(messages) for messages in messages_lst))
    
    def generate(
        self, messages_or_messages_lst, stop: str = None, max_tokens: int = None, temperature: float = None, include_stop_str_in_output: bool = None
    ) -> Union[str, List[str]]:
        is_single = isinstance(messages_or_messages_lst[0], dict)
        messages_lst = [messages_or_messages_lst] if is_single else messages_or_messages_lst
        if self.is_async:
            response_or_responses = asyncio.run(self._generate(messages_lst, stop, max_tokens, temperature, include_stop_str_in_output))
        else:
            @backoff.on_exception(backoff.expo, Exception, max_tries=self.max_tries, on_backoff=_on_backoff)
            def fetch_single_response(messages):
                request_data = self._prepare_request_data(
                    messages, stop, max_tokens, temperature, include_stop_str_in_output
                )
                response = self._openai_client.chat.completions.create(
                    model=request_data["model"],
                    messages=request_data["messages"],
                    stop=request_data.get("stop"),
                    max_tokens=request_data.get("max_tokens"),
                    temperature=request_data.get("temperature"),
                    extra_body=request_data.get("extra_body", {}),
                )
                return response.choices[0].message.content
            
            responses = [fetch_single_response(messages) for messages in messages_lst]
            response_or_responses = responses
        return response_or_responses[0] if is_single else response_or_responses
    
    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        raise NotImplementedError("evaluate method not implemented")

# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass

# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass