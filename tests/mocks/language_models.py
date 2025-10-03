"""Mock language models for testing."""


from its_hub.base import AbstractLanguageModel


class SimpleMockLanguageModel:
    """Simple mock language model for basic testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, tools=None, tool_choice=None):
        if isinstance(messages[0], list):
            # Multiple message lists
            content_responses = self.responses[self.call_count:self.call_count + len(messages)]
            self.call_count += len(messages)
            return [{"role": "assistant", "content": content} for content in content_responses]
        else:
            # Single message list
            content = self.responses[self.call_count]
            self.call_count += 1
            return {"role": "assistant", "content": content}


class StepMockLanguageModel(AbstractLanguageModel):
    """Mock language model for step-by-step generation testing."""

    def __init__(self, step_responses: list[str]):
        self.step_responses = step_responses
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, tools=None, tool_choice=None):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation
            num_requests = len(messages)
            responses = []
            for i in range(num_requests):
                response_idx = (self.call_count + i) % len(self.step_responses)
                content = self.step_responses[response_idx]
                responses.append({"role": "assistant", "content": content})
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            content = self.step_responses[self.call_count % len(self.step_responses)]
            self.call_count += 1
            return {"role": "assistant", "content": content}

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """Return mock evaluation scores."""
        return [0.1] * len(generation.split())

    async def generate_async(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, tools=None, tool_choice=None):
        return self.generate(messages, stop, max_tokens, temperature, include_stop_str_in_output, tools, tool_choice)

    async def evaluate_async(self, prompt: str, generation: str) -> list[float]:
        return self.evaluate(prompt, generation)


class ErrorMockLanguageModel(AbstractLanguageModel):
    """Mock language model that can simulate errors."""

    def __init__(self, responses: list[str], error_on_calls: list[int] | None = None):
        self.responses = responses
        self.error_on_calls = error_on_calls or []
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, tools=None, tool_choice=None):
        if self.call_count in self.error_on_calls:
            self.call_count += 1
            raise Exception("Simulated LM error")

        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation
            num_requests = len(messages)
            responses = []
            for i in range(num_requests):
                if (self.call_count + i) in self.error_on_calls:
                    raise Exception("Simulated LM error in batch")
                response_idx = (self.call_count + i) % len(self.responses)
                content = self.responses[response_idx]
                responses.append({"role": "assistant", "content": content})
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            content = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return {"role": "assistant", "content": content}

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        return [0.1] * len(generation.split())

    async def generate_async(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, tools=None, tool_choice=None):
        return self.generate(messages, stop, max_tokens, temperature, include_stop_str_in_output, tools, tool_choice)

    async def evaluate_async(self, prompt: str, generation: str) -> list[float]:
        return self.evaluate(prompt, generation)
