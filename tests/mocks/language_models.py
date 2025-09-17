"""Mock language models for testing."""


from its_hub.base import AbstractLanguageModel


class SimpleMockLanguageModel:
    """Simple mock language model for basic testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, messages_output=False):
        if isinstance(messages[0], list):
            # Multiple message lists
            base_responses = self.responses[self.call_count:self.call_count + len(messages)]
            self.call_count += len(messages)
            if messages_output:
                return [{"role": "assistant", "content": resp} for resp in base_responses]
            else:
                return base_responses
        else:
            # Single message list
            base_response = self.responses[self.call_count]
            self.call_count += 1
            if messages_output:
                return {"role": "assistant", "content": base_response}
            else:
                return base_response


class StepMockLanguageModel(AbstractLanguageModel):
    """Mock language model for step-by-step generation testing."""

    def __init__(self, step_responses: list[str]):
        self.step_responses = step_responses
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, messages_output=False):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation
            num_requests = len(messages)
            responses = []
            for i in range(num_requests):
                response_idx = (self.call_count + i) % len(self.step_responses)
                base_response = self.step_responses[response_idx]
                if messages_output:
                    responses.append({"role": "assistant", "content": base_response})
                else:
                    responses.append(base_response)
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            base_response = self.step_responses[self.call_count % len(self.step_responses)]
            self.call_count += 1
            if messages_output:
                return {"role": "assistant", "content": base_response}
            else:
                return base_response

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """Return mock evaluation scores."""
        return [0.1] * len(generation.split())


class ErrorMockLanguageModel(AbstractLanguageModel):
    """Mock language model that can simulate errors."""

    def __init__(self, responses: list[str], error_on_calls: list[int] = None):
        self.responses = responses
        self.error_on_calls = error_on_calls or []
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, temperature=None, include_stop_str_in_output=None, messages_output=False):
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
                base_response = self.responses[response_idx]
                if messages_output:
                    responses.append({"role": "assistant", "content": base_response})
                else:
                    responses.append(base_response)
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            base_response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            if messages_output:
                return {"role": "assistant", "content": base_response}
            else:
                return base_response

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        return [0.1] * len(generation.split())
