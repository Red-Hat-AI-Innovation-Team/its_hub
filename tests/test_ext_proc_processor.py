"""Tests for the Envoy ext_proc processor (integration/ext_proc/processor.py)."""

# ruff: noqa: I001
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import grpc
    import its_hub.integration.ext_proc.proto  # noqa: F401
    from envoy.config.core.v3 import base_pb2
    from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
    from envoy.type.v3 import http_status_pb2

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

pytestmark = pytest.mark.skipif(not HAS_GRPC, reason="ext_proc deps not installed")


# ---------------------------------------------------------------------------
# Proto helpers (adapted from scripts/test_envoy_grpc.py)
# ---------------------------------------------------------------------------


def _header(key, value):
    return base_pb2.HeaderValue(key=key, raw_value=value.encode("utf-8"))


def _request_headers(
    path="/v1/chat/completions",
    method="POST",
    its_budget=None,
    its_endpoint=None,
    its_api_key=None,
):
    headers = [
        _header(":path", path),
        _header(":method", method),
        _header("content-type", "application/json"),
    ]
    if its_budget is not None:
        headers.append(_header("x-its-budget", str(its_budget)))
    if its_endpoint is not None:
        headers.append(_header("x-its-endpoint", its_endpoint))
    if its_api_key is not None:
        headers.append(_header("x-its-api-key", its_api_key))
    return ext_proc_pb2.ProcessingRequest(
        request_headers=ext_proc_pb2.HttpHeaders(
            headers=base_pb2.HeaderMap(headers=headers)
        )
    )


def _request_body(body_dict, end_of_stream=True):
    return ext_proc_pb2.ProcessingRequest(
        request_body=ext_proc_pb2.HttpBody(
            body=json.dumps(body_dict).encode("utf-8"),
            end_of_stream=end_of_stream,
        )
    )


def _raw_body(raw_bytes, end_of_stream=True):
    return ext_proc_pb2.ProcessingRequest(
        request_body=ext_proc_pb2.HttpBody(body=raw_bytes, end_of_stream=end_of_stream)
    )


async def _async_iter(items):
    for item in items:
        yield item


async def _run_process(processor, requests, context=None):
    if context is None:
        context = MagicMock(spec=grpc.ServicerContext)
        context.peer.return_value = "test-peer"
    responses = []
    async for resp in processor.Process(_async_iter(requests), context):
        responses.append(resp)
    return responses


def _mock_gateway_result():
    return {
        "the_one": {"role": "assistant", "content": "42"},
        "responses": [
            {"role": "assistant", "content": "42"},
            {"role": "assistant", "content": "4"},
        ],
        "response_counts": {"42": 2, "4": 1},
        "selected_index": 0,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "num_calls": 3,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _parse_its_headers
# ---------------------------------------------------------------------------


class TestParseITSHeaders:
    def _make_processor(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            return ExternalProcessorService()

    def test_valid_headers(self):
        p = self._make_processor()
        config = p._parse_its_headers(
            {"x-its-budget": "5", "x-its-endpoint": "http://llm/v1"}
        )
        assert config is not None
        assert config.budget == 5
        assert config.api_endpoint == "http://llm/v1"
        assert config.api_key is None
        assert config.model is None

    def test_with_api_key(self):
        p = self._make_processor()
        config = p._parse_its_headers(
            {
                "x-its-budget": "3",
                "x-its-endpoint": "http://llm/v1",
                "x-its-api-key": "sk-test",
            }
        )
        assert config.api_key == "sk-test"

    def test_missing_budget(self):
        p = self._make_processor()
        assert p._parse_its_headers({"x-its-endpoint": "http://llm/v1"}) is None

    def test_missing_endpoint(self):
        p = self._make_processor()
        assert p._parse_its_headers({"x-its-budget": "5"}) is None

    def test_budget_not_integer(self):
        p = self._make_processor()
        assert (
            p._parse_its_headers(
                {"x-its-budget": "abc", "x-its-endpoint": "http://llm/v1"}
            )
            is None
        )

    def test_budget_zero(self):
        p = self._make_processor()
        assert (
            p._parse_its_headers(
                {"x-its-budget": "0", "x-its-endpoint": "http://llm/v1"}
            )
            is None
        )

    def test_budget_over_1000(self):
        p = self._make_processor()
        assert (
            p._parse_its_headers(
                {"x-its-budget": "1001", "x-its-endpoint": "http://llm/v1"}
            )
            is None
        )

    def test_budget_boundary_1(self):
        p = self._make_processor()
        config = p._parse_its_headers(
            {"x-its-budget": "1", "x-its-endpoint": "http://llm/v1"}
        )
        assert config is not None
        assert config.budget == 1

    def test_budget_boundary_1000(self):
        p = self._make_processor()
        config = p._parse_its_headers(
            {"x-its-budget": "1000", "x-its-endpoint": "http://llm/v1"}
        )
        assert config is not None
        assert config.budget == 1000


# ---------------------------------------------------------------------------
# Tests: header stripping
# ---------------------------------------------------------------------------


class TestHeaderStripping:
    def test_strips_its_headers(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        resp = ExternalProcessorService._headers_continue(
            ["x-its-budget", "x-its-endpoint", "x-its-api-key"]
        )
        mutation = resp.request_headers.response.header_mutation
        assert set(mutation.remove_headers) == {
            "x-its-budget",
            "x-its-endpoint",
            "x-its-api-key",
        }

    def test_no_mutation_without_its_headers(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        resp = ExternalProcessorService._headers_continue([])
        mutation = resp.request_headers.response.header_mutation
        assert len(mutation.remove_headers) == 0

    def test_partial_headers(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        resp = ExternalProcessorService._headers_continue(["x-its-budget"])
        mutation = resp.request_headers.response.header_mutation
        assert list(mutation.remove_headers) == ["x-its-budget"]


# ---------------------------------------------------------------------------
# Tests: route filtering
# ---------------------------------------------------------------------------


class TestRouteFiltering:
    @pytest.mark.asyncio
    async def test_non_chat_path_passes_through(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        responses = await _run_process(
            p, [_request_headers(path="/v1/embeddings", its_budget=3, its_endpoint="http://x/v1")]
        )
        assert len(responses) == 1
        assert responses[0].HasField("request_headers")
        assert (
            responses[0].request_headers.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )

    @pytest.mark.asyncio
    async def test_models_path_passes_through(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        responses = await _run_process(
            p, [_request_headers(path="/v1/models")]
        )
        assert len(responses) == 1
        assert (
            responses[0].request_headers.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )


# ---------------------------------------------------------------------------
# Tests: pass-through
# ---------------------------------------------------------------------------


class TestPassThrough:
    @pytest.mark.asyncio
    async def test_no_its_headers(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        responses = await _run_process(
            p, [_request_headers(), _request_body(body)]
        )
        assert len(responses) == 2
        assert responses[1].HasField("request_body")
        assert (
            responses[1].request_body.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )

    @pytest.mark.asyncio
    async def test_missing_model_in_body(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        body = {"messages": [{"role": "user", "content": "hi"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        assert len(responses) == 2
        assert (
            responses[1].request_body.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )

    @pytest.mark.asyncio
    async def test_malformed_json_body(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _raw_body(b"not json{{{"),
            ],
        )
        assert len(responses) == 2
        assert (
            responses[1].request_body.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )


# ---------------------------------------------------------------------------
# Tests: ITS applied
# ---------------------------------------------------------------------------


class TestITSApplied:
    @pytest.mark.asyncio
    async def test_valid_request_returns_immediate_response(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "2+2?"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        # headers response + immediate_response (no body CONTINUE after)
        assert len(responses) == 2
        assert responses[1].HasField("immediate_response")
        ir = responses[1].immediate_response
        assert ir.status.code == http_status_pb2.OK

        resp_body = json.loads(ir.body.decode("utf-8"))
        assert resp_body["object"] == "chat.completion"
        assert resp_body["model"] == "gpt-4"
        assert resp_body["choices"][0]["message"]["content"] == "42"
        assert resp_body["id"].startswith("chatcmpl-its-")
        assert resp_body["usage"]["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_response_headers(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "2+2?"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        ir = responses[1].immediate_response
        header_dict = {
            h.header.key: h.header.raw_value for h in ir.headers.set_headers
        }
        assert header_dict["content-type"] == b"application/json"
        assert header_dict["x-its-applied"] == b"true"

    @pytest.mark.asyncio
    async def test_gateway_called_with_correct_args(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        messages = [{"role": "user", "content": "2+2?"}]
        tools = [{"type": "function", "function": {"name": "calc"}}]
        body = {"model": "gpt-4", "messages": messages, "tools": tools, "tool_choice": "auto"}
        await _run_process(
            p,
            [
                _request_headers(its_budget=5, its_endpoint="http://llm/v1", its_api_key="sk-x"),
                _request_body(body),
            ],
        )
        call_kwargs = mock_gw.arun_chat_completion.call_args.kwargs
        assert call_kwargs["config"].budget == 5
        assert call_kwargs["config"].api_endpoint == "http://llm/v1"
        assert call_kwargs["config"].api_key == "sk-x"
        assert call_kwargs["config"].model == "gpt-4"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Tests: failure fallback
# ---------------------------------------------------------------------------


class TestFailureFallback:
    @pytest.mark.asyncio
    async def test_gateway_exception_falls_back(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        assert len(responses) == 2
        assert responses[1].HasField("request_body")
        assert (
            responses[1].request_body.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )

    @pytest.mark.asyncio
    async def test_response_headers_pass_through(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        req = ext_proc_pb2.ProcessingRequest(
            response_headers=ext_proc_pb2.HttpHeaders(
                headers=base_pb2.HeaderMap(headers=[])
            )
        )
        responses = await _run_process(p, [req])
        assert len(responses) == 1
        assert responses[0].HasField("response_headers")
        assert (
            responses[0].response_headers.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )

    @pytest.mark.asyncio
    async def test_response_body_pass_through(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        with patch("its_hub.integration.ext_proc.processor.ITSGateway"):
            p = ExternalProcessorService()
        req = ext_proc_pb2.ProcessingRequest(
            response_body=ext_proc_pb2.HttpBody(body=b"ok", end_of_stream=True)
        )
        responses = await _run_process(p, [req])
        assert len(responses) == 1
        assert responses[0].HasField("response_body")
        assert (
            responses[0].response_body.response.status
            == ext_proc_pb2.CommonResponse.CONTINUE
        )


# ---------------------------------------------------------------------------
# Tests: usage
# ---------------------------------------------------------------------------


class TestUsage:
    @pytest.mark.asyncio
    async def test_usage_included_in_response(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=_mock_gateway_result())
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        resp_body = json.loads(responses[1].immediate_response.body.decode("utf-8"))
        assert resp_body["usage"]["prompt_tokens"] == 10
        assert resp_body["usage"]["completion_tokens"] == 20
        assert resp_body["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_empty_usage(self):
        from its_hub.integration.ext_proc.processor import ExternalProcessorService

        result = _mock_gateway_result()
        result["usage"] = {}
        mock_gw = MagicMock()
        mock_gw.arun_chat_completion = AsyncMock(return_value=result)
        with patch(
            "its_hub.integration.ext_proc.processor.ITSGateway", return_value=mock_gw
        ):
            p = ExternalProcessorService()
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        responses = await _run_process(
            p,
            [
                _request_headers(its_budget=3, its_endpoint="http://llm/v1"),
                _request_body(body),
            ],
        )
        resp_body = json.loads(responses[1].immediate_response.body.decode("utf-8"))
        assert resp_body["usage"] == {}


# ---------------------------------------------------------------------------
# Tests: _preview_message_content
# ---------------------------------------------------------------------------


class TestPreviewMessageContent:
    def test_short_string(self):
        from its_hub.integration.ext_proc.processor import _preview_message_content

        assert _preview_message_content({"content": "hello"}) == "hello"

    def test_long_string_truncated(self):
        from its_hub.integration.ext_proc.processor import _preview_message_content

        long = "a" * 200
        result = _preview_message_content({"content": long}, limit=10)
        assert result == "a" * 10 + "…"

    def test_structured_content(self):
        from its_hub.integration.ext_proc.processor import _preview_message_content

        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert _preview_message_content({"content": content}) == "hello world"

    def test_none_content(self):
        from its_hub.integration.ext_proc.processor import _preview_message_content

        assert _preview_message_content({"content": None}) == "<empty>"

    def test_empty_string(self):
        from its_hub.integration.ext_proc.processor import _preview_message_content

        assert _preview_message_content({"content": ""}) == "<empty>"


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_main_exits_without_grpc(self):
        from its_hub.integration.ext_proc.processor import main

        with patch("its_hub.integration.ext_proc.processor.grpc", None):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
