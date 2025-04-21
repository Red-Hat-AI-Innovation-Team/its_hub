from typing import List, Tuple
import requests
from .base import AbstractLanguageModel

class StepGeneration:
    def __init__(self, step_token: str, max_steps: int, stop_token: str):
        self.step_token = step_token
        self.max_steps = max_steps
        self.stop_token = stop_token

    def forward(self, lm: AbstractLanguageModel, prompt: str, steps_so_far: List[str] = []) -> Tuple[str, bool]:
        next_step = lm.generate(self.step_token.join([prompt] + steps_so_far), stop=self.step_token)
        is_stopped = self.stop_token in next_step or len(steps_so_far) >= self.max_steps
        return next_step, is_stopped

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

    def generate(self, prompt: str, stop: str = None) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            self._chat_completion_endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": messages,
                "stop": stop, 
            },
        )
        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Cannot decode response:\n{response.json()=}")
            raise e
    
    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        raise NotImplementedError("evaluate method not implemented")

# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass

# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass