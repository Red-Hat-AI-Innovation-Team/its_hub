"""Tests for the Inference-as-a-Service (IaaS) integration."""

import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from its_hub.integration.iaas import app, ConfigRequest, ChatCompletionRequest, ChatMessage


def find_free_port():
    """Find a free port to use for the test server."""
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


class MockRewardModel:
    """Mock reward model for testing."""
    
    def __init__(self, scores: List[float] = None):
        self.scores = scores or [0.8, 0.6, 0.9]  # Default scores
        self.call_count = 0
    
    def score(self, prompt: str, response: str) -> float:
        """Return a mock score."""
        score = self.scores[self.call_count % len(self.scores)]
        self.call_count += 1
        return score


class TestIaaSAPI(unittest.TestCase):
    """Test the IaaS API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the dummy vLLM server."""
        cls.vllm_port = find_free_port()
        cls.vllm_server = HTTPServer(('localhost', cls.vllm_port), DummyVLLMHandler)
        cls.vllm_server_thread = threading.Thread(target=cls.vllm_server.serve_forever)
        cls.vllm_server_thread.daemon = True
        cls.vllm_server_thread.start()
        
        # Give the server a moment to start
        time.sleep(0.1)
    
    @classmethod
    def tearDownClass(cls):
        """Shut down the dummy vLLM server."""
        cls.vllm_server.shutdown()
        cls.vllm_server_thread.join()
    
    def setUp(self):
        """Set up the test client for each test."""
        self.client = TestClient(app)
        self.vllm_endpoint = f"http://localhost:{self.vllm_port}"
        
        # Reset global state before each test
        import its_hub.integration.iaas as iaas_module
        iaas_module.LM_DICT.clear()
        iaas_module.SCALING_ALG = None
    
    def test_models_endpoint_empty(self):
        """Test /v1/models endpoint when no models are configured."""
        response = self.client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data == {"data": []}
    
    def test_chat_completions_without_configuration(self):
        """Test chat completions endpoint without prior configuration."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
    
    def test_configuration_validation(self):
        """Test configuration request validation."""
        # Missing required fields
        invalid_config = {
            "endpoint": self.vllm_endpoint,
            "model": "test-model"
            # Missing required fields
        }
        
        response = self.client.post("/configure", json=invalid_config)
        assert response.status_code == 422  # Validation error
    
    def test_configuration_invalid_algorithm(self):
        """Test configuration with invalid algorithm."""
        invalid_config = {
            "endpoint": self.vllm_endpoint,
            "api_key": "test-key",
            "model": "test-model",
            "alg": "invalid-algorithm",  # Invalid algorithm
            "rm_name": "test-rm",
            "rm_device": "cpu"
        }
        
        response = self.client.post("/configure", json=invalid_config)
        assert response.status_code == 422  # Validation error
        assert "not supported" in str(response.json())
    
    @patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel')
    @patch('its_hub.integration.reward_hub.AggregationMethod')
    def test_configuration_best_of_n_success(self, mock_agg_method, mock_reward_model):
        """Test successful configuration with best-of-n algorithm."""
        # Mock the reward model
        mock_rm_instance = MagicMock()
        mock_reward_model.return_value = mock_rm_instance
        mock_agg_method.return_value = MagicMock()
        
        config_data = {
            "endpoint": self.vllm_endpoint,
            "api_key": "test-key",
            "model": "test-model",
            "alg": "best-of-n",
            "rm_name": "test-rm",
            "rm_device": "cpu",
            "rm_agg_method": "model"
        }
        
        response = self.client.post("/configure", json=config_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "best-of-n" in data["message"]
        
        # Verify reward model was created with correct parameters
        mock_reward_model.assert_called_once()
        call_args = mock_reward_model.call_args
        assert call_args[1]["model_name"] == "test-rm"
        assert call_args[1]["device"] == "cpu"
    
    @patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel')
    @patch('its_hub.integration.reward_hub.AggregationMethod')
    def test_configuration_particle_filtering_success(self, mock_agg_method, mock_reward_model):
        """Test successful configuration with particle-filtering algorithm."""
        # Mock the reward model
        mock_rm_instance = MagicMock()
        mock_reward_model.return_value = mock_rm_instance
        mock_agg_method.return_value = MagicMock()
        
        config_data = {
            "endpoint": self.vllm_endpoint,
            "api_key": "test-key",
            "model": "test-model",
            "alg": "particle-filtering",
            "step_token": "\\n",
            "stop_token": "<end>",
            "rm_name": "test-rm",
            "rm_device": "cuda:0",
            "rm_agg_method": "model"
        }
        
        response = self.client.post("/configure", json=config_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "particle-filtering" in data["message"]
    
    @patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel')
    @patch('its_hub.integration.reward_hub.AggregationMethod')
    def test_models_endpoint_after_configuration(self, mock_agg_method, mock_reward_model):
        """Test /v1/models endpoint after configuration."""
        # Mock dependencies
        mock_reward_model.return_value = MagicMock()
        mock_agg_method.return_value = MagicMock()
        
        # Configure the service
        config_data = {
            "endpoint": self.vllm_endpoint,
            "api_key": "test-key",
            "model": "test-model",
            "alg": "best-of-n",
            "rm_name": "test-rm",
            "rm_device": "cpu"
        }
        
        config_response = self.client.post("/configure", json=config_data)
        assert config_response.status_code == 200
        
        # Check models endpoint
        response = self.client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "its_hub"
    
    def test_chat_completions_validation(self):
        """Test chat completions request validation."""
        # Empty messages
        invalid_request = {
            "model": "test-model",
            "messages": [],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422
        
        # Too many messages
        invalid_request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Too many messages"}
            ],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422
        
        # Wrong role order
        invalid_request = {
            "model": "test-model", 
            "messages": [
                {"role": "user", "content": "User first"},
                {"role": "system", "content": "System second"}
            ],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422
        
        # Invalid budget
        invalid_request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test"}],
            "budget": 0  # Too low
        }
        
        response = self.client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422
    
    def test_chat_completions_streaming_not_implemented(self):
        """Test that streaming is not yet implemented."""
        # Configure first
        self._configure_service()
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "budget": 4,
            "stream": True
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"]
    
    def test_chat_completions_model_not_found(self):
        """Test chat completions with non-existent model."""
        # Configure first
        self._configure_service()
        
        request_data = {
            "model": "non-existent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_chat_completions_success(self):
        """Test successful chat completion."""
        # Configure first
        self._configure_service()
        
        # Mock the scaling algorithm
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.return_value = "Mocked scaling response"
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Solve 2+2"}],
            "budget": 8,
            "temperature": 0.7
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Mocked scaling response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data
        
        # Verify the scaling algorithm was called correctly
        mock_scaling_alg.infer.assert_called_once()
        call_args = mock_scaling_alg.infer.call_args
        assert call_args[0][1] == "Solve 2+2"  # prompt
        assert call_args[0][2] == 8  # budget
    
    def test_chat_completions_with_system_message(self):
        """Test chat completion with system message."""
        # Configure first
        self._configure_service()
        
        # Mock the scaling algorithm
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.return_value = "Response with system prompt"
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor"},
                {"role": "user", "content": "Explain algebra"}
            ],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Response with system prompt"
        
        # Verify the scaling algorithm was called
        mock_scaling_alg.infer.assert_called_once()
    
    def test_chat_completions_algorithm_error(self):
        """Test chat completion when scaling algorithm raises an error."""
        # Configure first
        self._configure_service()
        
        # Mock the scaling algorithm to raise an error
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.side_effect = Exception("Algorithm failed")
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Test"}],
            "budget": 4
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]
    
    def test_api_documentation_available(self):
        """Test that API documentation is available."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        # Check that it's returning HTML (OpenAPI docs)
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_spec_available(self):
        """Test that OpenAPI specification is available."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert spec["info"]["title"] == "its_hub Inference-as-a-Service"
        assert spec["info"]["version"] == "0.1.0-alpha"
        
        # Check that our endpoints are documented
        paths = spec["paths"]
        assert "/configure" in paths
        assert "/v1/models" in paths
        assert "/v1/chat/completions" in paths
    
    def _configure_service(self):
        """Helper method to configure the service for testing."""
        with patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel') as mock_rm:
            with patch('its_hub.integration.reward_hub.AggregationMethod') as mock_agg:
                mock_rm.return_value = MagicMock()
                mock_agg.return_value = MagicMock()
                return self._do_configure()
    
    def _do_configure(self):
        """Actually perform the configuration."""
        config_data = {
            "endpoint": self.vllm_endpoint,
            "api_key": "test-key",
            "model": "test-model",
            "alg": "best-of-n",
            "rm_name": "test-rm",
            "rm_device": "cpu"
        }
        
        response = self.client.post("/configure", json=config_data)
        assert response.status_code == 200
        return response


class TestConfigRequestModel(unittest.TestCase):
    """Test the ConfigRequest Pydantic model."""
    
    def test_valid_config_request(self):
        """Test creating a valid ConfigRequest."""
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key="test-key",
            model="test-model",
            alg="particle-filtering",
            step_token="\\n",
            stop_token="END",
            rm_name="reward-model",
            rm_device="cuda:0",
            rm_agg_method="model"
        )
        
        assert config.endpoint == "http://localhost:8000"
        assert config.alg == "particle-filtering"
        assert config.step_token == "\\n"
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithms are rejected."""
        with pytest.raises(ValueError) as exc_info:
            ConfigRequest(
                endpoint="http://localhost:8000",
                api_key="test-key", 
                model="test-model",
                alg="invalid-algorithm",  # Invalid
                rm_name="reward-model",
                rm_device="cuda:0"
            )
        
        assert "not supported" in str(exc_info.value)


class TestChatCompletionRequestModel(unittest.TestCase):
    """Test the ChatCompletionRequest Pydantic model."""
    
    def test_valid_chat_request(self):
        """Test creating a valid ChatCompletionRequest."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello")
            ],
            budget=8,
            temperature=0.7
        )
        
        assert request.model == "test-model"
        assert len(request.messages) == 2
        assert request.budget == 8
        assert request.temperature == 0.7
    
    def test_budget_validation(self):
        """Test budget parameter validation."""
        # Budget too low
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Test")],
                budget=0  # Too low
            )
        
        # Budget too high  
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="test-model",
                messages=[ChatMessage(role="user", content="Test")],
                budget=1001  # Too high
            )
    
    def test_message_validation(self):
        """Test message validation."""
        # Too many messages
        with pytest.raises(ValueError) as exc_info:
            ChatCompletionRequest(
                model="test-model",
                messages=[
                    ChatMessage(role="system", content="System"),
                    ChatMessage(role="user", content="User"),
                    ChatMessage(role="assistant", content="Too many")
                ],
                budget=4
            )
        assert "Maximum 2 messages" in str(exc_info.value)
        
        # Last message not from user
        with pytest.raises(ValueError) as exc_info:
            ChatCompletionRequest(
                model="test-model",
                messages=[
                    ChatMessage(role="user", content="User"),
                    ChatMessage(role="assistant", content="Assistant last")
                ],
                budget=4
            )
        assert "Last message must be from user" in str(exc_info.value)


if __name__ == "__main__":
    unittest.main()