"""Tests for the Inference-as-a-Service (IaaS) integration."""

import logging
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from its_hub.api.types import ChatMessage
from its_hub.integration.iaas.app import _state, app
from its_hub.integration.iaas.models import (
    ChatCompletionRequest,
    ConfigRequest,
)
from tests.conftest import TEST_CONSTANTS
from tests.mocks.test_data import TestDataFactory


@pytest.fixture
def iaas_client():
    """Create a test client for the IaaS API, resetting state between tests."""
    _state.reset()
    yield TestClient(app)
    _state.reset()


@pytest.fixture(scope="session")
def vllm_endpoint(vllm_server):
    """Alias the vllm_server fixture for clarity in IaaS tests."""
    return vllm_server


def _mock_gateway_result(content="answer", usage=None):
    """Build a dict matching ITSGateway.arun_chat_completion return (response_only=True)."""
    if usage is None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
            "num_calls": 1,
        }
    return {"message": {"role": "assistant", "content": content}, "usage": usage}


def _mock_gateway_full_result(content="answer", usage=None):
    """Build a dict matching ITSGateway.arun_chat_completion return (response_only=False)."""
    if usage is None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
            "num_calls": 1,
        }
    return {
        "the_one": {"role": "assistant", "content": content},
        "responses": [
            {"role": "assistant", "content": content},
            {"role": "assistant", "content": "other"},
        ],
        "response_counts": Counter({content: 2, "other": 1}),
        "selected_index": 0,
        "usage": usage,
    }


class TestIaaSAPIEndpoints:
    def test_models_endpoint_empty(self, iaas_client):
        response = iaas_client.get("/v1/models")
        assert response.status_code == 200
        assert response.json() == {"data": []}

    def test_chat_completions_without_configuration(self, iaas_client):
        request_data = TestDataFactory.create_chat_completion_request()
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 400
        assert "api_endpoint" in response.json()["detail"]

    def test_api_documentation_available(self, iaas_client):
        response = iaas_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_spec_available(self, iaas_client):
        response = iaas_client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "its_hub Inference-as-a-Service"
        assert spec["info"]["version"] == "0.1.0-alpha"
        paths = spec["paths"]
        assert "/configure" in paths
        assert "/v1/models" in paths
        assert "/v1/chat/completions" in paths


class TestConfiguration:
    def test_configuration_validation_missing_fields(self, iaas_client, vllm_endpoint):
        invalid_config = {
            "endpoint": vllm_endpoint,
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
        }
        response = iaas_client.post("/configure", json=invalid_config)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "invalid_algorithm",
        [
            "invalid-algorithm",
            "beam-search",
            "particle-gibbs",
        ],
    )
    def test_configuration_invalid_algorithm(
        self, iaas_client, vllm_endpoint, invalid_algorithm
    ):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": invalid_algorithm,
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 422
        assert "not supported" in str(response.json())

    def test_configure_creates_gateway(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 200
        assert _state.gateway is not None
        assert _state.config.model == TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
        assert _state.config.endpoint == vllm_endpoint

    def test_models_endpoint_after_configure(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)
        response = iaas_client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["data"]
        assert len(models) == 1
        assert models[0]["id"] == TEST_CONSTANTS["DEFAULT_MODEL_NAME"]


class TestSelfConsistencyToolVote:
    def test_self_consistency_basic_configuration(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    @pytest.mark.parametrize(
        "tool_vote",
        ["tool_name", "tool_args", "tool_hierarchical"],
    )
    def test_self_consistency_with_tool_vote(
        self, iaas_client, vllm_endpoint, tool_vote
    ):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": tool_vote,
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 200
        assert "success" in response.json()["status"]

    def test_self_consistency_with_exclude_tool_args(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "tool_args",
            "exclude_tool_args": ["timestamp", "request_id"],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 200

    def test_invalid_tool_vote_value(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "tool_vote": "invalid_option",
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 400
        assert "tool_vote must be one of" in response.json()["detail"]

    def test_tool_vote_algorithm_usage_verification(self, iaas_client, vllm_endpoint):
        with patch("its_hub.core.gateway.SelfConsistency") as mock_sc:
            mock_sc.return_value = MagicMock()
            config = {
                "endpoint": vllm_endpoint,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "self-consistency",
                "regex_patterns": [r"\\boxed{([^}]+)}"],
                "tool_vote": "tool_hierarchical",
                "exclude_tool_args": ["timestamp", "id"],
            }
            response = iaas_client.post("/configure", json=config)
            assert response.status_code == 200

            mock_sc.assert_called_once()
            call_args = mock_sc.call_args
            assert call_args.kwargs["tool_vote"] == "tool_hierarchical"
            assert call_args.kwargs["exclude_args"] == ["timestamp", "id"]


class TestAdaptiveAndBetaSelfConsistency:
    """Configuration of the adaptive/beta self-consistency variants via /configure."""

    @pytest.mark.parametrize(
        "alg",
        ["adaptive-self-consistency", "beta-self-consistency"],
    )
    def test_family_basic_configuration(self, iaas_client, vllm_endpoint, alg):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": alg,
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 200
        assert "success" in response.json()["status"]
        assert _state.config.alg == alg

    def test_adaptive_threshold_forwarded_to_algorithm(
        self, iaas_client, vllm_endpoint
    ):
        with patch("its_hub.core.gateway.AdaptiveSelfConsistency") as mock_alg:
            mock_alg.return_value = MagicMock()
            config = {
                "endpoint": vllm_endpoint,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "adaptive-self-consistency",
                "regex_patterns": [r"\\boxed{([^}]+)}"],
                "threshold": 0.9,
            }
            response = iaas_client.post("/configure", json=config)
            assert response.status_code == 200
            mock_alg.assert_called_once()
            assert mock_alg.call_args.kwargs["threshold"] == 0.9

    def test_beta_confidence_threshold_forwarded_to_algorithm(
        self, iaas_client, vllm_endpoint
    ):
        with patch("its_hub.core.gateway.BetaSelfConsistency") as mock_alg:
            mock_alg.return_value = MagicMock()
            config = {
                "endpoint": vllm_endpoint,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "beta-self-consistency",
                "regex_patterns": [r"\\boxed{([^}]+)}"],
                "confidence_threshold": 0.8,
            }
            response = iaas_client.post("/configure", json=config)
            assert response.status_code == 200
            mock_alg.assert_called_once()
            assert mock_alg.call_args.kwargs["confidence_threshold"] == 0.8

    def test_threshold_omitted_uses_algorithm_default(self, iaas_client, vllm_endpoint):
        """When threshold is unset, it is not forwarded so the class default applies."""
        with patch("its_hub.core.gateway.AdaptiveSelfConsistency") as mock_alg:
            mock_alg.return_value = MagicMock()
            config = {
                "endpoint": vllm_endpoint,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "adaptive-self-consistency",
                "regex_patterns": [r"\\boxed{([^}]+)}"],
            }
            response = iaas_client.post("/configure", json=config)
            assert response.status_code == 200
            assert "threshold" not in mock_alg.call_args.kwargs

    @pytest.mark.parametrize(
        ("field", "alg"),
        [
            ("threshold", "adaptive-self-consistency"),
            ("confidence_threshold", "beta-self-consistency"),
        ],
    )
    def test_threshold_out_of_range_rejected(
        self, iaas_client, vllm_endpoint, field, alg
    ):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": alg,
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            field: 0.5,  # not > 0.5
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 422

    def test_family_tool_vote_forwarded(self, iaas_client, vllm_endpoint):
        with patch("its_hub.core.gateway.BetaSelfConsistency") as mock_alg:
            mock_alg.return_value = MagicMock()
            config = {
                "endpoint": vllm_endpoint,
                "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
                "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                "alg": "beta-self-consistency",
                "tool_vote": "tool_hierarchical",
                "exclude_tool_args": ["timestamp"],
            }
            response = iaas_client.post("/configure", json=config)
            assert response.status_code == 200
            assert mock_alg.call_args.kwargs["tool_vote"] == "tool_hierarchical"
            assert mock_alg.call_args.kwargs["exclude_args"] == ["timestamp"]


class TestChatCompletions:
    def test_chat_completion_with_gateway(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(
            return_value=_mock_gateway_result(
                "Tool voting response",
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                    "num_calls": 6,
                },
            )
        )
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(
            user_content="What is 2+2?", budget=8
        )
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Tool voting response"
        assert data["usage"]["prompt_tokens"] == 10
        # num_calls is surfaced so clients can see how many LM calls the
        # (possibly adaptive) algorithm actually made for this ITS request.
        assert data["usage"]["num_calls"] == 6
        mock_gw.arun_chat_completion.assert_called_once()

    def test_chat_completion_full_result(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(
            return_value=_mock_gateway_full_result()
        )
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["return_response_only"] = False
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["metadata"] is not None
        assert data["metadata"]["algorithm"] == "self-consistency"
        assert data["metadata"]["selected_index"] == 0

    def test_chat_completion_metadata_reports_configured_algorithm(
        self, iaas_client, vllm_endpoint
    ):
        """Metadata reflects the configured algorithm, not a hardcoded value."""
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "beta-self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(
            return_value=_mock_gateway_full_result()
        )
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["return_response_only"] = False
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.json()["metadata"]["algorithm"] == "beta-self-consistency"

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
        response = iaas_client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422


class TestStreamingChatCompletions:
    """Tests for the SSE streaming path (_stream_chat_completions)."""

    def _parse_sse(self, response):
        """Parse SSE lines from a streaming response into data payloads."""
        import json

        chunks = []
        for line in response.text.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    chunks.append("[DONE]")
                else:
                    chunks.append(json.loads(payload))
        return chunks

    def _configure_and_mock(
        self, iaas_client, endpoint, mock_return=None, side_effect=None
    ):
        _state.config.endpoint = endpoint
        _state.config.model = TEST_CONSTANTS["DEFAULT_MODEL_NAME"]
        _state.config.api_key = TEST_CONSTANTS["DEFAULT_API_KEY"]
        mock_gw = MagicMock()
        if side_effect:
            mock_gw.arun_chat_completion = AsyncMock(side_effect=side_effect)
        else:
            mock_gw.arun_chat_completion = AsyncMock(return_value=mock_return)
        _state.gateway = mock_gw
        return mock_gw

    def test_stream_content_response(self, iaas_client, vllm_endpoint):
        self._configure_and_mock(
            iaas_client, vllm_endpoint, mock_return=_mock_gateway_result("Hello world")
        )

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["stream"] = True
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        chunks = self._parse_sse(response)
        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello world"
        assert chunks[1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[2] == "[DONE]"

    def test_stream_tool_call_response(self, iaas_client, vllm_endpoint):
        tool_call_result = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "function": {"name": "calculator", "arguments": '{"x": 1}'},
                    }
                ],
            },
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
                "num_calls": 1,
            },
        }
        self._configure_and_mock(
            iaas_client, vllm_endpoint, mock_return=tool_call_result
        )

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["stream"] = True
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        chunks = self._parse_sse(response)
        assert len(chunks) == 3
        tc_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc_delta["function"]["name"] == "calculator"
        assert chunks[1]["choices"][0]["finish_reason"] == "tool_calls"
        assert chunks[2] == "[DONE]"

    def test_stream_not_configured(self, iaas_client):
        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["stream"] = True
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        chunks = self._parse_sse(response)
        assert chunks[0]["error"] == "Service not configured"
        assert chunks[1] == "[DONE]"

    def test_stream_gateway_error_sanitized(self, iaas_client, vllm_endpoint):
        self._configure_and_mock(
            iaas_client,
            vllm_endpoint,
            side_effect=RuntimeError("Connection to http://internal:8100 refused"),
        )

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        request_data["stream"] = True
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        chunks = self._parse_sse(response)
        assert "internal:8100" not in chunks[0]["error"]
        assert "Check server logs" in chunks[0]["error"]
        assert chunks[1] == "[DONE]"


class TestPydanticModels:
    def test_valid_config_request(self):
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
        )
        assert config.endpoint == "http://localhost:8000"
        assert config.alg == "self-consistency"

    def test_invalid_algorithm_in_config(self):
        with pytest.raises(ValueError, match="not supported"):
            ConfigRequest(
                endpoint="http://localhost:8000",
                api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                alg="invalid-algorithm",
            )

    def test_valid_chat_completion_request(self):
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

    @pytest.mark.parametrize("invalid_budget", [0, 1001])
    def test_budget_validation_in_chat_request(self, invalid_budget):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                messages=[ChatMessage(role="user", content="Test")],
                budget=invalid_budget,
            )

    def test_message_validation_empty_messages(self):
        with pytest.raises(ValueError, match="At least one message is required"):
            ChatCompletionRequest(
                model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"], messages=[], budget=4
            )

    def test_config_request_with_tool_vote_parameters(self):
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
            tool_vote="tool_hierarchical",
            exclude_tool_args=["timestamp", "id"],
        )
        assert config.tool_vote == "tool_hierarchical"
        assert config.exclude_tool_args == ["timestamp", "id"]

    def test_config_request_tool_vote_optional(self):
        config = ConfigRequest(
            endpoint="http://localhost:8000",
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            alg="self-consistency",
            regex_patterns=[r"\\boxed{([^}]+)}"],
        )
        assert config.tool_vote is None
        assert config.exclude_tool_args is None

    def test_chat_completion_request_defaults(self):
        request = ChatCompletionRequest(
            model=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            messages=[ChatMessage(role="user", content="Test")],
            budget=4,
        )
        assert request.return_response_only is True


class TestHeaderConfig:
    """Tests for per-request configuration via X-ITS-* headers."""

    def test_budget_from_header_overrides_body(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-Budget": "16"},
        )
        assert response.status_code == 200

        call_kwargs = mock_gw.arun_chat_completion.call_args.kwargs
        assert call_kwargs["config"].budget == 16

    def test_endpoint_from_header_overrides_service_default(
        self, iaas_client, vllm_endpoint
    ):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-Endpoint": "http://override:9999/v1"},
        )
        assert response.status_code == 200

        call_kwargs = mock_gw.arun_chat_completion.call_args.kwargs
        assert call_kwargs["config"].api_endpoint == "http://override:9999/v1"

    def test_api_key_from_header(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-API-Key": "header-key"},
        )
        assert response.status_code == 200

        call_kwargs = mock_gw.arun_chat_completion.call_args.kwargs
        assert call_kwargs["config"].api_key == "header-key"

    def test_headers_without_service_config(self, iaas_client):
        """Headers alone can configure a request — no /configure needed."""
        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={
                "X-ITS-Budget": "8",
                "X-ITS-Endpoint": "http://llm:8100/v1",
                "X-ITS-API-Key": "sk-test",
            },
        )
        assert response.status_code == 200

        call_kwargs = mock_gw.arun_chat_completion.call_args.kwargs
        assert call_kwargs["config"].budget == 8
        assert call_kwargs["config"].api_endpoint == "http://llm:8100/v1"
        assert call_kwargs["config"].api_key == "sk-test"

    def test_400_when_no_endpoint_from_any_source(self, iaas_client):
        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-Budget": "8"},
        )
        assert response.status_code == 400
        assert "api_endpoint" in response.json()["detail"]


class TestRegexValidation:
    """Tests for regex pattern validation at configure time."""

    def test_rejects_invalid_regex(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": ["[invalid("],
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 400
        assert "Invalid regex pattern" in response.json()["detail"]


class TestBudgetValidation:
    """Tests for budget validation edge cases."""

    def test_negative_budget_header_rejected(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-Budget": "-5"},
        )
        assert response.status_code == 400
        assert "budget" in response.json()["detail"].lower()

    def test_over_max_budget_header_rejected(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post(
            "/v1/chat/completions",
            json=request_data,
            headers={"X-ITS-Budget": "1001"},
        )
        assert response.status_code == 400
        assert "budget" in response.json()["detail"].lower()

    def test_configure_rejects_invalid_budget(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
            "budget": 0,
        }
        response = iaas_client.post("/configure", json=config)
        assert response.status_code == 422


class TestErrorSanitization:
    """Tests that internal errors don't leak details to clients."""

    def test_generation_error_is_sanitized(self, iaas_client, vllm_endpoint):
        config = {
            "endpoint": vllm_endpoint,
            "api_key": TEST_CONSTANTS["DEFAULT_API_KEY"],
            "model": TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            "alg": "self-consistency",
            "regex_patterns": [r"\\boxed{([^}]+)}"],
        }
        iaas_client.post("/configure", json=config)

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(
            side_effect=RuntimeError("Connection to http://internal:8100 refused")
        )
        _state.gateway = mock_gw

        request_data = TestDataFactory.create_chat_completion_request(budget=4)
        response = iaas_client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "internal:8100" not in detail
        assert "Check server logs" in detail


class TestModelValidatorFixes:
    """Tests that model validators fire even when fields are omitted."""

    @pytest.mark.parametrize(
        "alg",
        [
            "self-consistency",
            "adaptive-self-consistency",
            "beta-self-consistency",
        ],
    )
    def test_family_accepts_neither_regex_nor_tool_vote(self, alg):
        """Neither projection field is required; the algorithm defaults handle voting."""
        config = ConfigRequest(
            endpoint="http://example.com:8000",
            api_key="sk-test",
            model="test-model",
            alg=alg,
        )
        assert config.regex_patterns is None
        assert config.tool_vote is None

    def test_self_consistency_accepts_tool_vote_without_regex(self):
        """tool_vote alone is a valid projection surface."""
        config = ConfigRequest(
            endpoint="http://example.com:8000",
            api_key="sk-test",
            model="test-model",
            alg="self-consistency",
            tool_vote="tool_hierarchical",
        )
        assert config.regex_patterns is None
        assert config.tool_vote == "tool_hierarchical"

    def test_openai_requires_api_key_when_omitted(self):
        with pytest.raises(ValueError, match="api_key is required"):
            ConfigRequest(
                endpoint="http://example.com:8000",
                model="test-model",
                alg="self-consistency",
                regex_patterns=[r"\\boxed{([^}]+)}"],
            )


class TestExtProcessor:
    """Tests for the IaaS ext_proc."""

    @pytest.fixture
    def _make_headers_request(self):
        """Build an ext_proc ProcessingRequest with request_headers."""
        from envoy.config.core.v3 import base_pb2
        from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2

        from its_hub.integration.proto import envoy  # noqa: F401

        def _build(headers: dict[str, str]):
            header_list = [
                base_pb2.HeaderValue(key=k, raw_value=v.encode())
                for k, v in headers.items()
            ]
            return ext_proc_pb2.ProcessingRequest(
                request_headers=ext_proc_pb2.HttpHeaders(
                    headers=base_pb2.HeaderMap(headers=header_list)
                )
            )

        return _build

    @pytest.fixture
    def ext_proc(self):
        from its_hub.integration.iaas.ext_processor import ExternalProcessorService

        return ExternalProcessorService()

    @pytest.mark.asyncio
    async def test_routes_and_preserves_headers_when_budget_present(
        self, ext_proc, _make_headers_request
    ):
        request = _make_headers_request(
            {
                ":path": "/v1/chat/completions",
                "x-its-budget": "4",
                "x-its-endpoint": "http://localhost:8100/v1",
                "x-its-api-key": "secret",
            }
        )

        async def _stream():
            yield request

        responses = []
        async for resp in ext_proc.Process(_stream(), MagicMock(peer=lambda: "test")):
            responses.append(resp)

        assert len(responses) == 1
        resp = responses[0]
        headers_resp = resp.request_headers.response

        assert headers_resp.clear_route_cache is True

        set_headers = {
            h.header.key: h.header.raw_value.decode()
            for h in headers_resp.header_mutation.set_headers
        }
        assert set_headers["X-ITS-Route"] == "its-service"

        removed = list(headers_resp.header_mutation.remove_headers)
        assert not removed

    @pytest.mark.asyncio
    async def test_passes_through_without_budget_header(
        self, ext_proc, _make_headers_request
    ):
        request = _make_headers_request(
            {
                ":path": "/v1/chat/completions",
                "content-type": "application/json",
            }
        )

        async def _stream():
            yield request

        responses = []
        async for resp in ext_proc.Process(_stream(), MagicMock(peer=lambda: "test")):
            responses.append(resp)

        assert len(responses) == 1
        resp = responses[0]
        headers_resp = resp.request_headers.response

        assert headers_resp.clear_route_cache is False
        assert not headers_resp.header_mutation.set_headers
        assert not list(headers_resp.header_mutation.remove_headers)

    @pytest.mark.asyncio
    async def test_strips_stray_its_headers_on_pass_through(
        self, ext_proc, _make_headers_request
    ):
        """Stray ITS headers without budget are stripped on pass-through."""
        request = _make_headers_request(
            {
                ":path": "/v1/chat/completions",
                "x-its-endpoint": "http://localhost:8100/v1",
                "x-its-api-key": "secret",
            }
        )

        async def _stream():
            yield request

        responses = []
        async for resp in ext_proc.Process(_stream(), MagicMock(peer=lambda: "test")):
            responses.append(resp)

        resp = responses[0]
        headers_resp = resp.request_headers.response

        assert headers_resp.clear_route_cache is False
        assert not headers_resp.header_mutation.set_headers

        removed = list(headers_resp.header_mutation.remove_headers)
        assert "x-its-endpoint" in removed
        assert "x-its-api-key" in removed


class TestAppServer:
    """Tests for app_server.py entry point."""

    def test_configure_logging(self):
        from its_hub.integration.iaas.app_server import _configure_logging

        _configure_logging("DEBUG")
        assert logging.getLogger("its_hub.integration.iaas.app").level == logging.DEBUG

    def test_configure_logging_invalid_level(self):
        from its_hub.integration.iaas.app_server import _configure_logging

        with pytest.raises(ValueError, match="Invalid log level"):
            _configure_logging("BOGUS")

    def test_parse_args_defaults(self):
        from its_hub.integration.iaas.app_server import _parse_args

        with patch("sys.argv", ["its-iaas"]):
            args = _parse_args()
        assert args.host == "127.0.0.1"
        assert args.port == 8109
        assert args.log_level == "INFO"
        assert args.dev is False
        assert args.print_config is False

    def test_print_config(self, capsys):
        from its_hub.integration.iaas.app_server import _print_config

        _print_config()
        output = capsys.readouterr().out
        assert "envoy" in output.lower()

    def test_serve_calls_uvicorn(self):
        from its_hub.integration.iaas.app_server import serve

        with patch("its_hub.integration.iaas.app_server.uvicorn") as mock_uvicorn:
            serve(host="127.0.0.1", port=9999)
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs.kwargs["host"] == "127.0.0.1"
            assert call_kwargs.kwargs["port"] == 9999

    def test_main_print_config(self, capsys):
        from its_hub.integration.iaas.app_server import main

        with patch("sys.argv", ["its-iaas", "--print-config"]):
            main()
        output = capsys.readouterr().out
        assert "envoy" in output.lower()

    def test_main_missing_uvicorn(self):
        from its_hub.integration.iaas import app_server

        with (
            patch("sys.argv", ["its-iaas"]),
            patch.object(app_server, "uvicorn", None),
            pytest.raises(SystemExit),
        ):
            app_server.main()


class TestGrpcServer:
    """Tests for grpc_server.py entry point."""

    def test_configure_logging(self):
        from its_hub.integration.iaas.grpc_server import _configure_logging

        _configure_logging("WARNING")
        assert (
            logging.getLogger("its_hub.integration.iaas.ext_processor").level
            == logging.WARNING
        )

    def test_configure_logging_invalid_level(self):
        from its_hub.integration.iaas.grpc_server import _configure_logging

        with pytest.raises(ValueError, match="Invalid log level"):
            _configure_logging("BOGUS")

    def test_parse_args_defaults(self):
        from its_hub.integration.iaas.grpc_server import _parse_args

        with patch("sys.argv", ["its-iaas-ext-proc"]):
            args = _parse_args()
        assert args.port == 50051
        assert args.log_level == "INFO"

    def test_main_missing_grpc(self):
        from its_hub.integration.iaas import grpc_server

        with (
            patch("sys.argv", ["its-iaas-ext-proc"]),
            patch.object(grpc_server, "grpc", None),
            pytest.raises(SystemExit),
        ):
            grpc_server.main()
