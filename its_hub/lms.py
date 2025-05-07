from typing import Union, List, Tuple
import asyncio
from openai import OpenAI, AsyncOpenAI
from .base import AbstractLanguageModel

class StepGeneration:
    def __init__(self, step_token: str, max_steps: int, stop_token: str):
        self.step_token = step_token
        self.max_steps = max_steps
        self.stop_token = stop_token

    def _forward(
        self, lm: AbstractLanguageModel, prompt: str, steps_so_far: List[str] = []
    ) -> Tuple[str, bool]:
        next_step = lm.generate(
            self.step_token.join([prompt] + steps_so_far), stop=self.step_token, temperature=0.8
        )
        is_stopped = self.stop_token in next_step or len(steps_so_far) >= self.max_steps
        return next_step, is_stopped
    
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
                                 "content": self.step_token.join(steps_so_far) + self.step_token})
            next_step = lm.generate(
                messages, stop=self.step_token, temperature=0.8
            )
            is_stopped = self.stop_token in next_step or len(steps_so_far) >= self.max_steps
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
                                     "content": self.step_token.join(steps_so_far_per_prompt) + self.step_token})
                messages_lst.append(messages)
            next_steps = lm.generate(
                messages_lst, stop=self.step_token, temperature=0.8
            )
            is_stopped = [self.stop_token in next_step or len(steps_so_far_per_prompt) >= self.max_steps 
                          for next_step, steps_so_far_per_prompt in zip(next_steps, steps_so_far)]
            return list(zip(next_steps, is_stopped))

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
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.is_async = is_async
        
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
    
    def _prepare_request_data(self, messages, stop=None, max_tokens=None, temperature=None):
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
        return request_data

    async def _generate(
        self, messages_lst, stop: str = None, max_tokens: int = None, temperature: float = None, max_retries: int = 3
    ) -> List[str]:
        # use openai's async client for batch requests
        async def fetch_response(messages):
            request_data = self._prepare_request_data(messages, stop, max_tokens, temperature)
            retries = 0
            while retries <= max_retries:
                try:
                    response = await self._openai_client.chat.completions.create(
                        model=request_data["model"],
                        messages=request_data["messages"],
                        stop=request_data.get("stop"),
                        max_tokens=request_data.get("max_tokens"),
                        temperature=request_data.get("temperature"),
                        extra_body=request_data.get("extra_body", {}),
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"[OpenAICompatibleLanguageModel] failed after {max_retries} retries: {e}")
                        raise e
                    wait_time = 2 ** retries  # exponential backoff
                    print(f"[OpenAICompatibleLanguageModel] retry {retries}/{max_retries} after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)

        # gather all responses asynchronously
        return await asyncio.gather(*(fetch_response(messages) for messages in messages_lst))
    
    def generate(
        self, messages_or_messages_lst, stop: str = None, max_tokens: int = None, temperature: float = None, max_retries: int = 3
    ) -> Union[str, List[str]]:
        is_single = isinstance(messages_or_messages_lst[0], dict)
        messages_lst = [messages_or_messages_lst] if is_single else messages_or_messages_lst
        if self.is_async:
            response_or_responses = asyncio.run(self._generate(messages_lst, stop, max_tokens, temperature, max_retries))
        else:
            responses = []
            for messages in messages_lst:
                request_data = self._prepare_request_data(messages, stop, max_tokens, temperature)
                retries = 0
                while retries <= max_retries:
                    try:
                        response = self._openai_client.chat.completions.create(
                            model=request_data["model"],
                            messages=request_data["messages"],
                            stop=request_data.get("stop"),
                            max_tokens=request_data.get("max_tokens"),
                            temperature=request_data.get("temperature"),
                            extra_body=request_data.get("extra_body", {}),
                        )
                        responses.append(response.choices[0].message.content)
                        break
                    except Exception as e:
                        retries += 1
                        if retries > max_retries:
                            print(f"[OpenAICompatibleLanguageModel] failed after {max_retries} retries: {e}")
                            raise e
                        wait_time = 2 ** retries  # exponential backoff
                        print(f"[OpenAICompatibleLanguageModel] retry {retries}/{max_retries} after {wait_time}s: {e}")
                        import time
                        time.sleep(wait_time)
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