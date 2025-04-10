from typing import List
import requests
from .base import AbstractLanguageModel

class StepGeneration:
    def __init__(self, step: str, max_steps: int):
        self.step = step
        self.max_steps = max_steps

    def forward(self, lm: AbstractLanguageModel, prompt: str, steps_so_far: List[str] = []) -> str:
        pass

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
        return response.json()["choices"][0]["message"]["content"]
    
    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        raise NotImplementedError("evaluate method not implemented")

# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass

# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass