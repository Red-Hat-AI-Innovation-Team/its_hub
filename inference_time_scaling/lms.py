import requests
from .base import AbstractLanguageModel


class OpenAIAPILanguageModel(AbstractLanguageModel):
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

    def generate(self, prompt: str) -> str:
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
            },
        )
        return response.json()["choices"][0]["message"]["content"]