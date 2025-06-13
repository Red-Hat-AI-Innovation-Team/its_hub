"""Mock language models for testing."""

from typing import List
from its_hub.base import AbstractLanguageModel


class SimpleMockLanguageModel:
    """Simple mock language model for basic testing."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
        if isinstance(messages[0], list):
            # Multiple message lists
            responses = self.responses[self.call_count:self.call_count + len(messages)]
            self.call_count += len(messages)
            return responses
        else:
            # Single message list
            response = self.responses[self.call_count]
            self.call_count += 1
            return response


class StepMockLanguageModel(AbstractLanguageModel):
    """Mock language model for step-by-step generation testing."""
    
    def __init__(self, step_responses: List[str]):
        self.step_responses = step_responses
        self.call_count = 0
        
    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation
            num_requests = len(messages)
            responses = []
            for i in range(num_requests):
                response_idx = (self.call_count + i) % len(self.step_responses)
                responses.append(self.step_responses[response_idx])
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            response = self.step_responses[self.call_count % len(self.step_responses)]
            self.call_count += 1
            return response
            
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        """Return mock evaluation scores."""
        return [0.1] * len(generation.split())


class ErrorMockLanguageModel(AbstractLanguageModel):
    """Mock language model that can simulate errors."""
    
    def __init__(self, responses: List[str], error_on_calls: List[int] = None):
        self.responses = responses
        self.error_on_calls = error_on_calls or []
        self.call_count = 0
        
    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
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
                responses.append(self.responses[response_idx])
            self.call_count += num_requests
            return responses
        else:
            # Single generation
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return response
            
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        return [0.1] * len(generation.split())