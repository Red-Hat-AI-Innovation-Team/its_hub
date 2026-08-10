"""Clean tests for the Inference-as-a-Service (IaaS) integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from its_hub.integration.iaas import ChatCompletionRequest, ChatMessage, ConfigRequest
from tests.conftest import TEST_CONSTANTS
from tests.mocks.test_data import TestDataFactory


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
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

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
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            # Missing required fields
        }

        response = iaas_client.post("/configure", json=invalid_config)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "invalid_algorithm",
        [
            "invalid-algorithm",
            "beam-search",  # Not implemented in IaaS
            "particle-gibbs",  # Not implemented in IaaS
        ],
    )
    def test_configuration_invalid_algorithm(
        self, iaas_client, vllm_server, invalid_algorithm
    ):
        """Test configuration with invalid or unsupported algorithms."""
        invalid_config = TestDataFactory.create_config_request(
            endpoint=vllm_server, alg=invalid_algorithm
        )

        response = iaas_client.post("/configure", json=invalid_config)
        assert response.status_code == 422
        assert "not supported" in str(response.json())


class TestSelfConsistencyToolVote:
    """Test self-consistency configuration with tool-vote functionality."""

    def test_self_consistency_basic_configuration(self, iaas_client, vllm_server):
        """Test basic self-consistency configuration without tool-vote."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_tool_name_vote(self, iaas_client, vllm_server):
        """Test self-consistency configuration with tool_name voting."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_name",
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_tool_args_vote(self, iaas_client, vllm_server):
        """Test self-consistency configuration with tool_args voting."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_args",
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_hierarchical_vote(self, iaas_client, vllm_server):
        """Test self-consistency configuration with tool_hierarchical voting."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_hierarchical",
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_exclude_tool_args(self, iaas_client, vllm_server):
        """Test self-consistency configuration with exclude_tool_args."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_args",
            "exclude_tool_args": ["timestamp", "request_id"]
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_all_tool_vote_options(
        self, iaas_client, vllm_server
    ):
        """Test self-consistency configuration with all tool-vote options."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_hierarchical",
            "exclude_tool_args": ["timestamp", "id", "session_id"]
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_invalid_tool_vote_value(self, iaas_client, vllm_server):
        """Test that invalid tool_vote values are rejected by SelfConsistency class."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "invalid_option",
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 500
        assert "tool_vote must be one of" in response.json()["detail"]

    def test_tool_vote_algorithm_usage_verification(self, iaas_client, vllm_server):
        """Test that configured tool-vote parameters are passed to SelfConsistency."""
        # Mock the SelfConsistency constructor to verify parameters
        with patch("its_hub.integration.iaas.SelfConsistency") as mock_sc:
            mock_sc.return_value = MagicMock()

            config_data = {
                "endpoint": vllm_server,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "self-consistency",
                "regex_patterns": [r"\\boxed{([^}]+)}"],
                "tool_vote": "tool_hierarchical",
                "exclude_tool_args": ["timestamp", "id"]
            }

            response = iaas_client.post("/configure", json=config_data)
            assert response.status_code == 200

            # Verify SelfConsistency was called with correct parameters
            mock_sc.assert_called_once()
            call_args = mock_sc.call_args

            # Check keyword arguments
            assert call_args.kwargs["tool_vote"] == "tool_hierarchical"
            assert call_args.kwargs["exclude_args"] == ["timestamp", "id"]

    def test_tool_vote_with_chat_completion(self, iaas_client, vllm_server):
        """Test chat completion with tool-vote configured algorithm."""
        # Configure service with tool voting
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_name",
        }

        config_response = iaas_client.post("/configure", json=config_data)
        assert config_response.status_code == 200

        # Mock the scaling algorithm
        import its_hub.integration.iaas as iaas_module

        mock_scaling_alg = MagicMock()
        mock_scaling_alg.ainfer = AsyncMock(return_value={"role": "assistant", "content": "Tool voting response"})
        iaas_module.SCALING_ALG = mock_scaling_alg

        # Make chat completion request
        request_data = TestDataFactory.create_chat_completion_request(
            user_content="Use a calculator tool to solve 15 * 23", budget=8
        )

        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Tool voting response"

        # Verify the scaling algorithm was called
        mock_scaling_alg.ainfer.assert_called_once()

    @pytest.mark.parametrize("tool_vote_config", [
        {"tool_vote": "tool_name"},
        {"tool_vote": "tool_args", "exclude_tool_args": ["id"]},
        {"tool_vote": "tool_hierarchical", "exclude_tool_args": ["timestamp", "session_id"]},
    ])
    def test_various_tool_vote_configurations(self, iaas_client, vllm_server, tool_vote_config):
        """Test various valid tool-vote configurations."""
        config_data = {
            "endpoint": vllm_server,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            **tool_vote_config,
        }

        response = iaas_client.post("/configure", json=config_data)
        assert response.status_code == 200
        assert "success" in response.json()["status"]


class TestChatCompletions:
    """Test the chat completions endpoint."""

    @pytest.mark.parametrize(
        "invalid_request",
        [
            {"model": "test-model", "messages": [], "budget": 4},
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "budget": 0,
            },
        ],
    )
    def test_chat_completions_validation(self, iaas_client, invalid_request):
        """Test chat completions request validation with various invalid inputs."""
        response = iaas_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422

    def _configure_service(self, iaas_client, vllm_server):
        """Helper method to configure the service for testing."""
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
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
            step_token="\n",
            stop_token="END",
        )

        assert config.endpoint == "http://localhost:8000"
        assert config.alg == "self-consistency"
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
                rm_device="cuda:0",
            )

        assert "not supported" in str(exc_info.value)

    def test_valid_chat_completion_request(self):
        """Test creating a valid ChatCompletionRequest."""
        request = ChatCompletionRequest(
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
            budget=8,
            temperature=TEST_CONSTANTS["DEFAULT_TEMPERATURE"],
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
                budget=invalid_budget,
            )

    def test_message_validation_in_chat_request_empty_messages(self):
        """Test message validation in ChatCompletionRequest for empty messages."""
        with pytest.raises(ValueError) as exc_info:
            ChatCompletionRequest(
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"], messages=[], budget=4
            )
        assert "At least one message is required" in str(exc_info.value)

    def test_config_request_with_tool_vote_parameters(self):
        """Test ConfigRequest with tool-vote parameters."""
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
            tool_vote="tool_hierarchical",
            exclude_tool_args=["timestamp", "id"]
        )

        assert config.tool_vote == "tool_hierarchical"
        assert config.exclude_tool_args == ["timestamp", "id"]
        assert config.regex_patterns == [r"\\boxed{([^}]+)}"]

    def test_config_request_tool_vote_optional(self):
        """Test that tool_vote and exclude_tool_args are optional."""
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
        )

        assert config.tool_vote is None
        assert config.exclude_tool_args is None

    def test_chat_completion_request_with_return_response_only(self):
        """Test ChatCompletionRequest with return_response_only parameter."""
        request = ChatCompletionRequest(
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            messages=[ChatMessage(role="user", content="Test")],
            budget=4,
            return_response_only=False,
        )

        assert request.return_response_only is False

    def test_chat_completion_request_defaults(self):
        """Test ChatCompletionRequest default values."""
        request = ChatCompletionRequest(
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            messages=[ChatMessage(role="user", content="Test")],
            budget=4,
        )

        assert request.return_response_only is True  # Default value
