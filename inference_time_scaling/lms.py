from typing import Union, List, Tuple
import requests
import asyncio
import aiohttp
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
            prompt = self.step_token.join([prompt] + steps_so_far)
            next_step = lm.generate(
                prompt, stop=self.step_token, temperature=0.8
            )
            is_stopped = self.stop_token in next_step or len(steps_so_far) >= self.max_steps
            return next_step, is_stopped
        else:
            prompts = prompt_or_prompts
            prompts = [self.step_token.join([prompt] + steps_so_far_per_prompt) 
                       for prompt, steps_so_far_per_prompt in zip(prompts, steps_so_far)]
            next_steps = lm.generate(
                prompts, stop=self.step_token, temperature=0.8
            )
            is_stopped = [self.stop_token in next_step or len(steps_so_far_per_prompt) >= self.max_steps 
                          for next_step, steps_so_far_per_prompt in zip(next_steps, steps_so_far)]
            return list(zip(next_steps, is_stopped))

class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self, endpoint: str, api_key: str, model_name: str, system_prompt: str = None
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"
    
    def _prepare_request_data(self, prompt, stop=None, max_tokens=None, temperature=None):
        # helper method to prepare request data for both sync and async methods
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        request_data = {
            "model": self.model_name,
            "messages": messages,
        }
        if stop is not None:
            request_data["stop"] = stop
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        return request_data

    async def _generate(
        self, prompts: List[str], stop: str = None, max_tokens: int = None, temperature: float = None
    ) -> List[str]:
        async def fetch_response(session, prompt):
            request_data = self._prepare_request_data(prompt, stop, max_tokens, temperature)
            
            async with session.post(
                self._chat_completion_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=request_data,
            ) as response:
                response_json = None
                try:
                    response_json = await response.json()
                    return response_json["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Cannot decode response:\n{response_json=}")
                    raise e
        
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_response(session, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
    
    def generate(
        self, prompt_or_prompts: Union[str, List[str]], stop: str = None, max_tokens: int = None, temperature: float = None
    ) -> Union[str, List[str]]:
        is_single_prompt = isinstance(prompt_or_prompts, str)
        prompts = [prompt_or_prompts] if is_single_prompt else prompt_or_prompts
        response_or_responses = asyncio.run(self._generate(prompts, stop, max_tokens, temperature))
        return response_or_responses[0] if is_single_prompt else response_or_responses
    
    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        raise NotImplementedError("evaluate method not implemented")

# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass

# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass