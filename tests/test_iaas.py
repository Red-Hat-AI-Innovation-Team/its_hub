"""Clean tests for the Inference-as-a-Service (IaaS) integration."""

from unittest.mock import patch, MagicMock
import pytest

from its_hub.integration.iaas import ConfigRequest, ChatCompletionRequest, ChatMessage
from tests.mocks.test_data import TestDataFactory, ALGORITHM_CONFIGS
from tests.conftest import TEST_CONSTANTS


class TestIaaSAPIEndpoints:
    """Test the IaaS API endpoints with improved organization."""
    
    def test_models_endpoint_empty(self, iaas_client):
        """Test /v1/models endpoint when no models are configured."""
        response = iaas_client.get("/v1/models")
        assert response.status_code == 200
        assert response.json() == {"data": []}

    def test_chat_completions_without_configuration(self, iaas_client):
        """Test chat completions endpoint without prior configuration."""
        request_data = TestDataFactory.create_chat_completion_request()
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_api_documentation_available(self, iaas_client):
        """Test that API documentation is available."""
        response = iaas_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_spec_available(self, iaas_client):
        """Test that OpenAPI specification is available."""
        response = iaas_client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert spec["info"]["title"] == "its_hub Inference-as-a-Service"
        assert spec["info"]["version"] == "0.1.0-alpha"
        
        # Check that our endpoints are documented
        paths = spec["paths"]
        assert "/configure" in paths
        assert "/v1/models" in paths
        assert "/v1/chat/completions" in paths


class TestConfiguration:
    """Test the configuration endpoint and validation."""
    
    def test_configuration_validation_missing_fields(self, iaas_client, vllm_server):
        """Test configuration request validation with missing fields."""
        invalid_config = {
            "endpoint": vllm_server,
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
            # Missing required fields
        }
        
        response = iaas_client.post("/configure", json=invalid_config)
        assert response.status_code == 422

    @pytest.mark.parametrize("invalid_algorithm", [
        "invalid-algorithm",
        "beam-search",  # Not implemented in IaaS
        "particle-gibbs",  # Not implemented in IaaS
    ])
    def test_configuration_invalid_algorithm(self, iaas_client, vllm_server, invalid_algorithm):
        """Test configuration with invalid or unsupported algorithms."""
        invalid_config = TestDataFactory.create_config_request(
            endpoint=vllm_server,
            alg=invalid_algorithm
        )
        
        response = iaas_client.post("/configure", json=invalid_config)
        assert response.status_code == 422
        assert "not supported" in str(response.json())

    @pytest.mark.parametrize("algorithm,config_override", [
        ("best-of-n", {}),
        ("particle-filtering", {"step_token": "\n", "stop_token": "<end>"}),
    ])
    @patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel')
    @patch('its_hub.integration.reward_hub.AggregationMethod')
    def test_configuration_success(self, mock_agg_method, mock_reward_model, 
                                 iaas_client, vllm_server, algorithm, config_override):
        """Test successful configuration with different algorithms."""
        mock_reward_model.return_value = MagicMock()
        mock_agg_method.return_value = MagicMock()
        
        config_data = TestDataFactory.create_config_request(
            endpoint=vllm_server,
            alg=algorithm,
            **config_override
        )
        
        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert algorithm in data["message"]

    @patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel')
    @patch('its_hub.integration.reward_hub.AggregationMethod')
    def test_models_endpoint_after_configuration(self, mock_agg_method, mock_reward_model, 
                                                iaas_client, vllm_server):
        """Test /v1/models endpoint after configuration."""
        mock_reward_model.return_value = MagicMock()
        mock_agg_method.return_value = MagicMock()
        
        config_data = TestDataFactory.create_config_request(endpoint=vllm_server)
        config_response = iaas_client.post("/configure", json=config_data)
        assert config_response.status_code == 200
        
        response = iaas_client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "its_hub"


class TestChatCompletions:
    """Test the chat completions endpoint."""
    
    @pytest.mark.parametrize("invalid_request", [
        {"model": "test-model", "messages": [], "budget": 4},
        {"model": "test-model", "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Too many messages"}
        ], "budget": 4},
        {"model": "test-model", "messages": [
            {"role": "user", "content": "User first"},
            {"role": "system", "content": "System second"}
        ], "budget": 4},
        {"model": "test-model", "messages": [{"role": "user", "content": "Test"}], "budget": 0},
    ])
    def test_chat_completions_validation(self, iaas_client, invalid_request):
        """Test chat completions request validation with various invalid inputs."""
        response = iaas_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422

    def test_chat_completions_streaming_not_implemented(self, iaas_client, vllm_server):
        """Test that streaming is not yet implemented."""
        self._configure_service(iaas_client, vllm_server)
        
        request_data = TestDataFactory.create_chat_completion_request(stream=True)
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"]

    def test_chat_completions_model_not_found(self, iaas_client, vllm_server):
        """Test chat completions with non-existent model."""
        self._configure_service(iaas_client, vllm_server)
        
        request_data = TestDataFactory.create_chat_completion_request(model="non-existent-model")
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_chat_completions_success(self, iaas_client, vllm_server):
        """Test successful chat completion."""
        self._configure_service(iaas_client, vllm_server)
        
        # Mock the scaling algorithm
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.return_value = "Mocked scaling response"
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = TestDataFactory.create_chat_completion_request(
            user_content="Solve 2+2",
            budget=8,
            temperature=TEST_CONSTANTS["DEFAULT_TEMPERATURE"]
        )
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
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

    def test_chat_completions_with_system_message(self, iaas_client, vllm_server):
        """Test chat completion with system message."""
        self._configure_service(iaas_client, vllm_server)
        
        # Mock the scaling algorithm
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.return_value = "Response with system prompt"
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = TestDataFactory.create_chat_completion_request(
            user_content="Explain algebra",
            system_content="You are a helpful math tutor"
        )
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Response with system prompt"
        
        # Verify the scaling algorithm was called
        mock_scaling_alg.infer.assert_called_once()

    def test_chat_completions_algorithm_error(self, iaas_client, vllm_server):
        """Test chat completion when scaling algorithm raises an error."""
        self._configure_service(iaas_client, vllm_server)
        
        # Mock the scaling algorithm to raise an error
        import its_hub.integration.iaas as iaas_module
        mock_scaling_alg = MagicMock()
        mock_scaling_alg.infer.side_effect = Exception("Algorithm failed")
        iaas_module.SCALING_ALG = mock_scaling_alg
        
        request_data = TestDataFactory.create_chat_completion_request()
        
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 500
        assert "Generation failed" in response.json()["detail"]

    def _configure_service(self, iaas_client, vllm_server):
        """Helper method to configure the service for testing."""
        with patch('its_hub.integration.reward_hub.LocalVllmProcessRewardModel') as mock_rm:
            with patch('its_hub.integration.reward_hub.AggregationMethod') as mock_agg:
                mock_rm.return_value = MagicMock()
                mock_agg.return_value = MagicMock()
                
                config_data = TestDataFactory.create_config_request(endpoint=vllm_server)
                response = iaas_client.post("/configure", json=config_data)
                assert response.status_code == 200
                return response


class TestPydanticModels:
    """Test the Pydantic model validation."""
    
    def test_valid_config_request(self):
        """Test creating a valid ConfigRequest."""
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="particle-filtering",
            step_token="\n",
            stop_token="END",
            rm_name="reward-model",
            rm_device="cuda:0",
            rm_agg_method="model"
        )
        
        assert config.endpoint == "http://localhost:8000"
        assert config.alg == "particle-filtering"
        assert config.step_token == "\n"

    def test_invalid_algorithm_in_config(self):
        """Test that invalid algorithms are rejected in ConfigRequest."""
        with pytest.raises(ValueError) as exc_info:
            ConfigRequest(
                endpoint="http://localhost:8000",
                api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                alg="invalid-algorithm",
                rm_name="reward-model",
                rm_device="cuda:0"
            )
        
        assert "not supported" in str(exc_info.value)

    def test_valid_chat_completion_request(self):
        """Test creating a valid ChatCompletionRequest."""
        request = ChatCompletionRequest(
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello")
            ],
            budget=8,
            temperature=TEST_CONSTANTS["DEFAULT_TEMPERATURE"]
        )
        
        assert request.model == TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
        assert len(request.messages) == 2
        assert request.budget == 8
        assert request.temperature == TEST_CONSTANTS["DEFAULT_TEMPERATURE"]

    @pytest.mark.parametrize("invalid_budget", [0, 1001])
    def test_budget_validation_in_chat_request(self, invalid_budget):
        """Test budget parameter validation in ChatCompletionRequest."""
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                messages=[ChatMessage(role="user", content="Test")],
                budget=invalid_budget
            )

    @pytest.mark.parametrize("invalid_messages,expected_error", [
        ([
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="User"),
            ChatMessage(role="assistant", content="Too many")
        ], "Maximum 2 messages"),
        ([
            ChatMessage(role="user", content="User"),
            ChatMessage(role="assistant", content="Assistant last")
        ], "Last message must be from user"),
    ])
    def test_message_validation_in_chat_request(self, invalid_messages, expected_error):
        """Test message validation in ChatCompletionRequest."""
        with pytest.raises(ValueError) as exc_info:
            ChatCompletionRequest(
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                messages=invalid_messages,
                budget=4
            )
        assert expected_error in str(exc_info.value)