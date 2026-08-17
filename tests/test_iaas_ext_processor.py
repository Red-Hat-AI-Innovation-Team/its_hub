"""Tests for the IaaS ext_proc router (integration/iaas/ext_processor.py).

The IaaS ext_proc is a lightweight gRPC service that makes routing decisions:
- X-ITS-Budget present → route to IaaS service (set X-ITS-Route header)
- No X-ITS-Budget → pass through to upstream LLM (strip stray ITS headers)
"""

# ruff: noqa: I001
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    import grpc
    import its_hub.integration.proto  # noqa: F401
    from envoy.config.core.v3 import base_pb2
    from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

pytestmark = pytest.mark.skipif(not HAS_GRPC, reason="ext_proc deps not installed")


# ---------------------------------------------------------------------------
# Proto helpers
# ---------------------------------------------------------------------------


def _header(key, value):
    return base_pb2.HeaderValue(key=key, raw_value=value.encode("utf-8"))


def _request_headers(*extra_headers):
    """Build a ProcessingRequest with request_headers."""
    headers = [
        _header(":path", "/v1/chat/completions"),
        _header(":method", "POST"),
        _header("content-type", "application/json"),
    ]
    for k, v in extra_headers:
        headers.append(_header(k, v))
    return ext_proc_pb2.ProcessingRequest(
        request_headers=ext_proc_pb2.HttpHeaders(
            headers=base_pb2.HeaderMap(headers=headers)
        )
    )


def _response_headers():
    return ext_proc_pb2.ProcessingRequest(
        response_headers=ext_proc_pb2.HttpHeaders(
            headers=base_pb2.HeaderMap(headers=[])
        )
    )


def _response_body():
    return ext_proc_pb2.ProcessingRequest(
        response_body=ext_proc_pb2.HttpBody(body=b'{"choices": []}', end_of_stream=True)
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


def _make_processor():
    from its_hub.integration.iaas.ext_processor import ExternalProcessorService

    return ExternalProcessorService()


# ---------------------------------------------------------------------------
# Tests: routing decisions
# ---------------------------------------------------------------------------


class TestIaaSExtProcRouting:
    @pytest.mark.asyncio
    async def test_budget_header_routes_to_iaas(self):
        """X-ITS-Budget present → route to IaaS with X-ITS-Route header."""
        from its_hub.integration.iaas.ext_processor import _ROUTE_TO_IAAS

        processor = _make_processor()
        request = _request_headers(("x-its-budget", "5"))
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        assert responses[0] == _ROUTE_TO_IAAS

        resp_headers = responses[0].request_headers.response
        assert resp_headers.clear_route_cache is True

        set_headers = resp_headers.header_mutation.set_headers
        assert len(set_headers) == 1
        assert set_headers[0].header.key == "X-ITS-Route"
        assert set_headers[0].header.raw_value == b"its-service"

    @pytest.mark.asyncio
    async def test_budget_with_other_its_headers_routes_to_iaas(self):
        """Multiple ITS headers with budget → still routes to IaaS."""
        from its_hub.integration.iaas.ext_processor import _ROUTE_TO_IAAS

        processor = _make_processor()
        request = _request_headers(
            ("x-its-budget", "3"),
            ("x-its-endpoint", "http://llm/v1"),
            ("x-its-api-key", "sk-test"),
        )
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        assert responses[0] == _ROUTE_TO_IAAS

    @pytest.mark.asyncio
    async def test_no_its_headers_passes_through(self):
        """No ITS headers → pass through unchanged."""
        from its_hub.integration.iaas.ext_processor import _PASS_THROUGH

        processor = _make_processor()
        request = _request_headers()
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        assert responses[0] == _PASS_THROUGH

    @pytest.mark.asyncio
    async def test_its_headers_without_budget_strips_and_passes_through(self):
        """ITS headers present but no X-ITS-Budget → strip stray headers, pass through."""
        processor = _make_processor()
        request = _request_headers(
            ("x-its-endpoint", "http://llm/v1"),
            ("x-its-api-key", "sk-test"),
        )
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        resp = responses[0]
        mutation = resp.request_headers.response.header_mutation
        assert "x-its-endpoint" in mutation.remove_headers
        assert "x-its-api-key" in mutation.remove_headers
        assert resp.request_headers.response.clear_route_cache is False

    @pytest.mark.asyncio
    async def test_single_stray_its_header_stripped(self):
        """A single stray X-ITS-Endpoint (no budget) → strip it."""
        processor = _make_processor()
        request = _request_headers(("x-its-endpoint", "http://llm/v1"))
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        mutation = responses[0].request_headers.response.header_mutation
        assert "x-its-endpoint" in mutation.remove_headers


# ---------------------------------------------------------------------------
# Tests: response phases
# ---------------------------------------------------------------------------


class TestIaaSExtProcResponsePhases:
    @pytest.mark.asyncio
    async def test_response_headers_continue(self):
        """response_headers phase → CONTINUE."""
        processor = _make_processor()
        responses = await _run_process(processor, [_response_headers()])

        assert len(responses) == 1
        status = responses[0].response_headers.response.status
        assert status == ext_proc_pb2.CommonResponse.CONTINUE

    @pytest.mark.asyncio
    async def test_response_body_continue(self):
        """response_body phase → CONTINUE."""
        processor = _make_processor()
        responses = await _run_process(processor, [_response_body()])

        assert len(responses) == 1
        status = responses[0].response_body.response.status
        assert status == ext_proc_pb2.CommonResponse.CONTINUE


# ---------------------------------------------------------------------------
# Tests: full request lifecycle
# ---------------------------------------------------------------------------


class TestIaaSExtProcLifecycle:
    @pytest.mark.asyncio
    async def test_full_its_request_lifecycle(self):
        """Full lifecycle: request_headers → response_headers → response_body."""
        from its_hub.integration.iaas.ext_processor import _ROUTE_TO_IAAS

        processor = _make_processor()
        requests = [
            _request_headers(("x-its-budget", "5")),
            _response_headers(),
            _response_body(),
        ]
        responses = await _run_process(processor, requests)

        assert len(responses) == 3
        assert responses[0] == _ROUTE_TO_IAAS
        assert responses[1].response_headers.response.status == ext_proc_pb2.CommonResponse.CONTINUE
        assert responses[2].response_body.response.status == ext_proc_pb2.CommonResponse.CONTINUE

    @pytest.mark.asyncio
    async def test_full_passthrough_lifecycle(self):
        """Full lifecycle without ITS headers: all phases CONTINUE."""
        from its_hub.integration.iaas.ext_processor import _PASS_THROUGH

        processor = _make_processor()
        requests = [
            _request_headers(),
            _response_headers(),
            _response_body(),
        ]
        responses = await _run_process(processor, requests)

        assert len(responses) == 3
        assert responses[0] == _PASS_THROUGH
        assert responses[1].HasField("response_headers")
        assert responses[2].HasField("response_body")


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestIaaSExtProcErrors:
    @pytest.mark.asyncio
    async def test_stream_error_aborts_with_internal(self):
        """Exception in request stream → abort with INTERNAL status."""
        processor = _make_processor()
        context = AsyncMock(spec=grpc.aio.ServicerContext)
        context.peer.return_value = "test-peer"

        async def _error_iter():
            raise RuntimeError("connection lost")
            yield  # noqa: unreachable — makes this an async generator

        responses = []
        async for resp in processor.Process(_error_iter(), context):
            responses.append(resp)

        assert len(responses) == 0
        context.abort.assert_awaited_once_with(
            grpc.StatusCode.INTERNAL, "connection lost"
        )

    @pytest.mark.asyncio
    async def test_stream_error_logged(self, caplog):
        """Stream error is logged at ERROR level."""
        processor = _make_processor()
        context = AsyncMock(spec=grpc.aio.ServicerContext)
        context.peer.return_value = "test-peer"

        async def _error_iter():
            raise ValueError("bad request")
            yield  # noqa: unreachable

        with caplog.at_level("ERROR"):
            async for _ in processor.Process(_error_iter(), context):
                pass

        assert any("Stream error" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests: case insensitivity
# ---------------------------------------------------------------------------


class TestIaaSExtProcCaseHandling:
    @pytest.mark.asyncio
    async def test_mixed_case_budget_header_routes(self):
        """X-ITS-Budget with mixed case still routes (keys are lowered)."""
        from its_hub.integration.iaas.ext_processor import _ROUTE_TO_IAAS

        processor = _make_processor()
        request = _request_headers(("X-ITS-Budget", "5"))
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        assert responses[0] == _ROUTE_TO_IAAS

    @pytest.mark.asyncio
    async def test_uppercase_its_headers_stripped_on_passthrough(self):
        """Mixed-case ITS headers without budget → lowered keys are stripped."""
        processor = _make_processor()
        request = _request_headers(("X-ITS-Endpoint", "http://llm/v1"))
        responses = await _run_process(processor, [request])

        assert len(responses) == 1
        mutation = responses[0].request_headers.response.header_mutation
        assert "x-its-endpoint" in mutation.remove_headers
