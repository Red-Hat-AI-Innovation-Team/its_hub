"""Test data factories for consistent test data generation."""

from typing import List, Dict, Any
from its_hub.integration.iaas import ChatMessage


class TestDataFactory:
    """Factory for creating consistent test data."""
    
    @staticmethod
    def create_chat_messages(user_content: str = "Hello", system_content: str = None) -> List[Dict[str, str]]:
        """Create standard chat messages for testing."""
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return messages
    
    @staticmethod
    def create_chat_completion_request(
        model: str = "test-model",
        user_content: str = "Hello",
        budget: int = 4,
        system_content: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a standard chat completion request."""
        request = {
            "model": model,
            "messages": TestDataFactory.create_chat_messages(user_content, system_content),
            "budget": budget
        }
        request.update(kwargs)
        return request
    
    @staticmethod
    def create_config_request(
        endpoint: str = "http://localhost:8000",
        api_key: str = "test-key",
        model: str = "test-model",
        alg: str = "best-of-n",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a standard configuration request."""
        config = {
            "endpoint": endpoint,
            "api_key": api_key,
            "model": model,
            "alg": alg,
            "rm_name": "test-rm",
            "rm_device": "cpu"
        }
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_error_trigger_request(trigger: str = "trigger_error") -> Dict[str, Any]:
        """Create a request that triggers errors for testing."""
        return TestDataFactory.create_chat_completion_request(user_content=trigger)
    
    @staticmethod
    def create_multiple_responses(base: str = "response", count: int = 3) -> List[str]:
        """Create multiple response strings for testing."""
        return [f"{base}{i+1}" for i in range(count)]
    
    @staticmethod
    def create_score_sequence(base_score: float = 0.5, count: int = 3, increment: float = 0.1) -> List[float]:
        """Create a sequence of scores for testing."""
        return [base_score + (i * increment) for i in range(count)]


# Common test scenarios
TEST_SCENARIOS = {
    "simple_chat": {
        "user_content": "Hello, world!",
        "expected_response": "Response to: Hello, world!"
    },
    "math_problem": {
        "user_content": "Solve 2+2",
        "expected_response": "Response to: Solve 2+2"
    },
    "error_trigger": {
        "user_content": "trigger_error",
        "should_error": True
    },
    "vllm_error_trigger": {
        "user_content": "error",
        "should_error": True
    },
    "with_system_prompt": {
        "system_content": "You are a helpful assistant",
        "user_content": "How can I help you?",
        "expected_response": "Response to: How can I help you?"
    }
}

# Algorithm test configurations
ALGORITHM_CONFIGS = {
    "best_of_n": {
        "alg": "best-of-n",
        "requires_outcome_rm": True,
        "supports_batching": True
    },
    "beam_search": {
        "alg": "beam-search",
        "requires_process_rm": True,
        "requires_step_generation": True,
        "budget_constraints": "divisible_by_beam_width"
    },
    "particle_filtering": {
        "alg": "particle-filtering",
        "requires_process_rm": True,
        "requires_step_generation": True,
        "supports_selection_methods": ["argmax", "sample"]
    },
    "particle_gibbs": {
        "alg": "particle-gibbs",
        "requires_process_rm": True,
        "requires_step_generation": True,
        "supports_selection_methods": ["argmax", "sample"],
        "budget_constraints": "divisible_by_iterations"
    }
}