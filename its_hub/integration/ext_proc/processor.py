"""
Envoy External Processor for Inference-Time Scaling.

This gRPC service implements the Envoy External Processor protocol to intercept
HTTP requests to LLM endpoints and apply inference-time scaling algorithms.
"""

# ruff: noqa: I001
import json
import logging
import time
import uuid

import grpc
import its_hub.integration.proto  # noqa: F401
from envoy.config.core.v3 import base_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_proc_grpc
from envoy.type.v3 import http_status_pb2

from its_hub.api import ChatMessage, ITSRequestConfig
from its_hub.core.gateway import ITSGateway

_ITS_HEADER_PREFIX = "x-its-"

logger = logging.getLogger(__name__)


def _preview_message_content(message: dict, limit: int = 160) -> str:
    """Return a single-line preview of message content for logging."""
    content = ChatMessage.from_dict(message).extract_text_content()
    preview = " ".join(content.split())
    if not preview:
        return "<empty>"
    return preview[:limit] + ("…" if len(preview) > limit else "")


class ExternalProcessorService(ext_proc_grpc.ExternalProcessorServicer):
    """
    Implements Envoy External Processor protocol for ITS.

    This service intercepts HTTP requests to /v1/chat/completions and applies
    inference-time scaling when X-ITS-* headers are present.

    Required Configuration (per-request):
    - Headers:
      * X-ITS-Budget: Computational budget for scaling (required, 1-1000)
      * X-ITS-Endpoint: LLM API endpoint URL (required)
      * X-ITS-API-Key: API key for LLM authentication (optional)
    - Body:
      * model: Model name in standard OpenAI request format (required)

    When ITS headers are absent, the request passes through without modification.
    If model is missing from body, the request passes through to upstream.
    """

    def __init__(self):
        """Initialize the external processor with gateway."""
        self.gateway = ITSGateway()
        logger.info("External Processor initialized")

    async def Process(  # noqa: N802 — gRPC servicer interface requires this name
        self,
        request_iterator,
        context: grpc.ServicerContext,
    ):
        """
        Process bidirectional stream of requests from Envoy.

        This is the main entry point called by Envoy for each HTTP request.
        """
        peer = context.peer()
        request_id = f"{peer}-{uuid.uuid4().hex[:8]}"
        logger.info("[%s] New stream started", request_id)

        request_path = None
        request_headers = {}
        request_body_chunks = []
        its_config = None

        try:
            async for request in request_iterator:
                if request.HasField("request_headers"):
                    logger.debug("[%s] Received request headers", request_id)
                    headers_msg = request.request_headers

                    its_header_keys = []
                    for header in headers_msg.headers.headers:
                        key = header.key.lower()
                        value = header.raw_value.decode("utf-8", errors="ignore")
                        request_headers[key] = value

                        if key == ":path":
                            request_path = value
                        if key.startswith(_ITS_HEADER_PREFIX):
                            its_header_keys.append(key)

                    logger.info("[%s] Request path: %s", request_id, request_path)

                    if request_path and not request_path.startswith(
                        "/v1/chat/completions"
                    ):
                        logger.info(
                            "[%s] Not a chat completions request, passing through",
                            request_id,
                        )
                        yield self._headers_continue(its_header_keys)
                        continue

                    its_config = self._parse_its_headers(request_headers)

                    if its_config:
                        logger.info(
                            "[%s] ITS headers detected: budget=%s, endpoint=%s",
                            request_id,
                            its_config.budget,
                            its_config.api_endpoint,
                        )
                    else:
                        logger.info(
                            "[%s] No ITS headers, will pass through", request_id
                        )

                    yield self._headers_continue(its_header_keys)

                elif request.HasField("request_body"):
                    logger.debug("[%s] Received request body chunk", request_id)
                    body_chunk = request.request_body
                    request_body_chunks.append(body_chunk.body)

                    if not body_chunk.end_of_stream:
                        yield self._body_continue()
                        continue

                    logger.info("[%s] Request body complete", request_id)
                    full_body = b"".join(request_body_chunks)

                    if not its_config:
                        logger.debug("[%s] Passing through without ITS", request_id)
                        yield self._body_continue()
                        continue

                    try:
                        yield await self._apply_its(its_config, full_body, request_id)
                        return
                    except Exception as e:
                        logger.error(
                            "[%s] Error processing ITS request: %s",
                            request_id,
                            e,
                            exc_info=True,
                        )
                        yield self._body_continue()

                elif request.HasField("response_headers"):
                    logger.debug(
                        "[%s] Received response headers, passing through",
                        request_id,
                    )
                    yield ext_proc_pb2.ProcessingResponse(
                        response_headers=ext_proc_pb2.HeadersResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

                elif request.HasField("response_body"):
                    logger.debug(
                        "[%s] Received response body, passing through",
                        request_id,
                    )
                    yield ext_proc_pb2.ProcessingResponse(
                        response_body=ext_proc_pb2.BodyResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

        except Exception as e:
            logger.error("[%s] Stream error: %s", request_id, e, exc_info=True)

        logger.info("[%s] Stream completed", request_id)

    async def _apply_its(
        self,
        its_config: ITSRequestConfig,
        body: bytes,
        request_id: str,
    ) -> ext_proc_pb2.ProcessingResponse:
        """Run ITS scaling on the request body and return an immediate or continue response."""
        request_data = json.loads(body.decode("utf-8"))

        if request_data.get("stream"):
            raise ValueError("Streaming requests are not supported by ITS")

        model = request_data.get("model")
        if not model:
            raise ValueError("No 'model' field in request body")

        its_config.model = model
        messages = request_data.get("messages", [])
        tools = request_data.get("tools")
        tool_choice = request_data.get("tool_choice")

        logger.info(
            "[%s] Processing %d messages with ITS (model=%s, budget=%s)",
            request_id,
            len(messages),
            model,
            its_config.budget,
        )

        result = await self.gateway.arun_chat_completion(
            config=its_config,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            return_response_only=False,
            request_id=request_id,
        )

        response_message = dict(result["the_one"])
        response_message.pop("usage", None)
        usage = result["usage"]

        preview = _preview_message_content(response_message)
        logger.info(
            "[%s] ITS selected candidate #%s with usage=%s preview='%s'",
            request_id,
            result["selected_index"],
            usage,
            preview,
        )
        logger.debug(
            "[%s] Candidate vote summary: %s",
            request_id,
            result["response_counts"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            for idx, candidate in enumerate(result["responses"]):
                logger.debug(
                    "[%s] Candidate %s content preview: %s",
                    request_id,
                    idx,
                    _preview_message_content(candidate),
                )

        openai_response = {
            "id": f"chatcmpl-its-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": its_config.model,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

        response_body = json.dumps(openai_response).encode("utf-8")

        logger.info(
            "[%s] ITS complete, returning response (%d bytes, usage=%s)",
            request_id,
            len(response_body),
            usage,
        )

        return ext_proc_pb2.ProcessingResponse(
            immediate_response=ext_proc_pb2.ImmediateResponse(
                status=http_status_pb2.HttpStatus(code=http_status_pb2.OK),
                body=response_body,
                headers=ext_proc_pb2.HeaderMutation(
                    set_headers=[
                        base_pb2.HeaderValueOption(
                            header=base_pb2.HeaderValue(
                                key="content-type",
                                raw_value=b"application/json",
                            )
                        ),
                        base_pb2.HeaderValueOption(
                            header=base_pb2.HeaderValue(
                                key="x-its-applied",
                                raw_value=b"true",
                            )
                        ),
                    ]
                ),
            )
        )

    @staticmethod
    def _body_continue() -> ext_proc_pb2.ProcessingResponse:
        """Build a CONTINUE BodyResponse for pass-through."""
        return ext_proc_pb2.ProcessingResponse(
            request_body=ext_proc_pb2.BodyResponse(
                response=ext_proc_pb2.CommonResponse(
                    status=ext_proc_pb2.CommonResponse.CONTINUE
                )
            )
        )

    @staticmethod
    def _headers_continue(
        its_header_keys: list[str],
    ) -> ext_proc_pb2.ProcessingResponse:
        """Build a CONTINUE HeadersResponse, stripping any X-ITS-* headers."""
        mutation = None
        if its_header_keys:
            mutation = ext_proc_pb2.HeaderMutation(
                remove_headers=its_header_keys,
            )
        return ext_proc_pb2.ProcessingResponse(
            request_headers=ext_proc_pb2.HeadersResponse(
                response=ext_proc_pb2.CommonResponse(
                    status=ext_proc_pb2.CommonResponse.CONTINUE,
                    header_mutation=mutation,
                )
            )
        )

    def _parse_its_headers(self, headers: dict[str, str]) -> ITSRequestConfig | None:
        """Parse ITS configuration from request headers.

        Model is NOT extracted from headers - it will be set from request body later.
        """
        budget_str = headers.get("x-its-budget")
        endpoint = headers.get("x-its-endpoint")

        if not all([budget_str, endpoint]):
            return None

        try:
            budget = int(budget_str)
            api_key = headers.get("x-its-api-key")

            return ITSRequestConfig(
                budget=budget,
                api_endpoint=endpoint,
                api_key=api_key,
                model=None,
            )

        except (ValueError, TypeError) as e:
            logger.error("Failed to parse ITS headers: %s", e)
            return None
