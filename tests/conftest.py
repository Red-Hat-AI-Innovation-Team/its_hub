"""Shared test configuration and fixtures."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from its_hub.base import AbstractLanguageModel, AbstractOutcomeRewardModel
from its_hub.integration.iaas import app


def find_free_port() -> int:
    """Find a free port to use for test servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class DummyVLLMHandler(BaseHTTPRequestHandler):
    """A dummy HTTP handler that mimics a vLLM server."""
    
    def do_POST(self):
        """Handle POST requests to the /v1/chat/completions endpoint."""
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Simulate some processing time
            time.sleep(0.01)
            
            # Extract the user message
            messages = request_data.get("messages", [])
            user_content = messages[-1]['content'] if messages else "unknown"
            
            # Check for error triggers
            if "error" in user_content.lower():
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": {
                        "message": "Simulated vLLM error",
                        "type": "server_error",
                        "code": 500
                    }
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                return
            
            # Create a response that includes the request content for testing
            response_content = f"vLLM response to: {user_content}"
            
            # Check if we should include stop tokens
            stop = request_data.get("stop")
            include_stop = request_data.get("include_stop_str_in_output", False)
            
            if stop and include_stop:
                response_content += stop
            
            # Create vLLM-like response
            response = {
                "id": "vllm-test-id",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        """Suppress log messages to keep test output clean."""
        pass


class DummyOpenAIHandler(BaseHTTPRequestHandler):
    """A dummy HTTP handler that mimics the OpenAI API."""
    
    # Class-level variables to track concurrent requests
    active_requests = 0
    max_concurrent_requests = 0
    request_lock = threading.Lock()
    
    @classmethod
    def reset_stats(cls):
        """Reset the request statistics."""
        with cls.request_lock:
            cls.active_requests = 0
            cls.max_concurrent_requests = 0
    
    def do_POST(self):
        """Handle POST requests to the /chat/completions endpoint."""
        if self.path == "/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Track concurrent requests
            with self.__class__.request_lock:
                self.__class__.active_requests += 1
                self.__class__.max_concurrent_requests = max(
                    self.__class__.max_concurrent_requests, 
                    self.__class__.active_requests
                )
            
            # Simulate some processing time
            time.sleep(0.1)
            
            # Check if we should simulate an error
            if "trigger_error" in request_data.get("messages", [{}])[-1].get("content", ""):
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": {
                        "message": "Simulated API error",
                        "type": "server_error",
                        "code": 500
                    }
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                
                # Decrement active requests
                with self.__class__.request_lock:
                    self.__class__.active_requests -= 1
                
                return
            
            # Extract the messages from the request
            messages = request_data.get("messages", [])
            
            # Prepare a response based on the messages
            response_content = f"Response to: {messages[-1]['content']}"
            
            # Check if there's a stop sequence and we should include it
            stop = request_data.get("stop")
            include_stop = request_data.get("include_stop_str_in_output", False)
            
            if stop and include_stop:
                response_content += stop
            
            # Create an OpenAI-like response
            response = {
                "id": "dummy-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": request_data.get("model", "dummy-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            
            # Send the response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            # Decrement active requests
            with self.__class__.request_lock:
                self.__class__.active_requests -= 1
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        """Suppress log messages to keep test output clean."""
        pass


class MockLanguageModel(AbstractLanguageModel):
    """Mock language model for testing."""
    
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        
    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation - messages is List[List[ChatMessage]]
            num_requests = len(messages)
            if self.call_count + num_requests > len(self.responses):
                # Cycle through responses if we run out
                responses = []
                for i in range(num_requests):
                    responses.append(self.responses[(self.call_count + i) % len(self.responses)])
            else:
                responses = self.responses[self.call_count:self.call_count + num_requests]
            self.call_count += num_requests
            return responses
        else:
            # Single generation - messages is List[ChatMessage]
            if self.call_count >= len(self.responses):
                # Cycle through responses if we run out
                response = self.responses[self.call_count % len(self.responses)]
            else:
                response = self.responses[self.call_count]
            self.call_count += 1
            return response
            
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        return [0.1] * len(generation.split())


class MockOutcomeRewardModel(AbstractOutcomeRewardModel):
    """Mock outcome reward model for testing."""
    
    def __init__(self, scores: List[float]):
        if isinstance(scores, float):
            self.scores = [scores]
        else:
            self.scores = scores
        self.call_count = 0
        
    def score(self, prompt: str, response) -> float:
        if isinstance(response, list):
            scores = self.scores[self.call_count:self.call_count + len(response)]
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


class MockProcessRewardModel:
    """Mock process reward model for testing."""
    
    def __init__(self, scores: List[float]):
        if isinstance(scores[0], list):
            self.scores = [score for sublist in scores for score in sublist]
        else:
            self.scores = scores
        self.call_count = 0
        
    def score(self, prompt: str, response) -> float:
        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                scores.append(self.scores[(self.call_count + i) % len(self.scores)])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


# Pytest fixtures

@pytest.fixture(scope="session")
def vllm_server():
    """Start a vLLM mock server for the test session."""
    port = find_free_port()
    server = HTTPServer(('localhost', port), DummyVLLMHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(0.1)
    
    yield f"http://localhost:{port}"
    
    server.shutdown()
    server_thread.join()


@pytest.fixture(scope="session")
def openai_server():
    """Start an OpenAI mock server for the test session."""
    port = find_free_port()
    server = HTTPServer(('localhost', port), DummyOpenAIHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(0.1)
    
    yield f"http://localhost:{port}"
    
    server.shutdown()
    server_thread.join()


@pytest.fixture
def iaas_client():
    """Create a test client for the IaaS API."""
    # Reset global state before each test
    import its_hub.integration.iaas as iaas_module
    iaas_module.LM_DICT.clear()
    iaas_module.SCALING_ALG = None
    
    return TestClient(app)


@pytest.fixture
def mock_language_model():
    """Create a mock language model with default responses."""
    return MockLanguageModel(["response1", "response2", "response3"])


@pytest.fixture
def mock_outcome_reward_model():
    """Create a mock outcome reward model with default scores."""
    return MockOutcomeRewardModel([0.8, 0.6, 0.9])


@pytest.fixture
def mock_process_reward_model():
    """Create a mock process reward model with default scores."""
    return MockProcessRewardModel([0.7, 0.6, 0.8, 0.5])


# Test constants
TEST_CONSTANTS = {
    "DEFAULT_BUDGET": 4,
    "DEFAULT_TEMPERATURE": 0.7,
    "DEFAULT_MODEL_NAME": "test-model",
    "DEFAULT_API_KEY": "test-key",
    "DEFAULT_TIMEOUT": 0.1,
    "ERROR_TRIGGER": "trigger_error",
    "VLLM_ERROR_TRIGGER": "error",
}