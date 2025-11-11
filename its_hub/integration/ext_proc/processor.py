"""
Envoy External Processor for Inference-Time Scaling.

This gRPC service implements the Envoy External Processor protocol to intercept
HTTP requests to LLM endpoints and apply inference-time scaling algorithms.
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional
import time

import grpc

# Import proto after setting up path
import its_hub.integration.ext_proc.proto
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_proc_grpc
from envoy.config.core.v3 import base_pb2

from its_hub.integration.orchestrator import ITSOrchestrator, ITSRequestConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        """Initialize the external processor with orchestrator."""
        self.orchestrator = ITSOrchestrator()
        logger.info("External Processor initialized")

    async def Process(
        self,
        request_iterator: AsyncIterator[ext_proc_pb2.ProcessingRequest],
        context: grpc.ServicerContext,
    ) -> AsyncIterator[ext_proc_pb2.ProcessingResponse]:
        """
        Process bidirectional stream of requests from Envoy.

        This is the main entry point called by Envoy for each HTTP request.
        """
        request_id = context.peer()  # Use peer info as request ID
        logger.info(f"[{request_id}] New stream started")

        # Request state
        request_path = None
        request_headers = {}
        request_body_chunks = []
        its_config = None

        try:
            async for request in request_iterator:
                # Handle request headers
                if request.HasField("request_headers"):
                    logger.debug(f"[{request_id}] Received request headers")
                    headers_msg = request.request_headers

                    # Extract headers into dict
                    for header in headers_msg.headers.headers:
                        key = header.key.lower()
                        value = header.raw_value.decode('utf-8', errors='ignore')
                        request_headers[key] = value

                        # Capture request path
                        if key == ":path":
                            request_path = value

                    logger.info(f"[{request_id}] Request path: {request_path}")

                    # Check if this is a /v1/chat/completions request
                    if request_path and not request_path.startswith("/v1/chat/completions"):
                        logger.info(f"[{request_id}] Not a chat completions request, passing through")
                        # Not a chat completion, pass through entire request
                        yield ext_proc_pb2.ProcessingResponse(
                            request_headers=ext_proc_pb2.HeadersResponse(
                                response=ext_proc_pb2.CommonResponse(
                                    status=ext_proc_pb2.CommonResponse.CONTINUE
                                )
                            )
                        )
                        # Continue processing but don't apply ITS
                        continue

                    # Try to parse ITS configuration from headers
                    its_config = self._parse_its_headers(request_headers)

                    if its_config:
                        logger.info(
                            f"[{request_id}] ITS headers detected: budget={its_config.budget}, "
                            f"endpoint={its_config.api_endpoint}"
                        )
                    else:
                        logger.info(f"[{request_id}] No ITS headers, will pass through")

                    # Continue to receive body
                    yield ext_proc_pb2.ProcessingResponse(
                        request_headers=ext_proc_pb2.HeadersResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

                # Handle request body
                elif request.HasField("request_body"):
                    logger.debug(f"[{request_id}] Received request body chunk")
                    body_chunk = request.request_body

                    # Collect body data
                    request_body_chunks.append(body_chunk.body)

                    # If this is the end of the stream, process it
                    if body_chunk.end_of_stream:
                        logger.info(f"[{request_id}] Request body complete")

                        # Combine all body chunks
                        full_body = b''.join(request_body_chunks)

                        # Apply ITS if configured
                        if its_config:
                            try:
                                # Parse request body
                                request_data = json.loads(full_body.decode('utf-8'))

                                # Extract model from body (standard OpenAI format)
                                model = request_data.get("model")
                                if not model:
                                    logger.error(f"[{request_id}] No 'model' field in request body")
                                    # Pass through to upstream
                                    yield ext_proc_pb2.ProcessingResponse(
                                        request_body=ext_proc_pb2.BodyResponse(
                                            response=ext_proc_pb2.CommonResponse(
                                                status=ext_proc_pb2.CommonResponse.CONTINUE
                                            )
                                        )
                                    )
                                    continue

                                # Set model on config
                                its_config.model = model

                                messages = request_data.get("messages", [])
                                tools = request_data.get("tools")
                                tool_choice = request_data.get("tool_choice")

                                logger.info(
                                    f"[{request_id}] Processing {len(messages)} messages with ITS "
                                    f"(model={model}, budget={its_config.budget})"
                                )

                                # Run ITS orchestration
                                result = await self.orchestrator.run_chat_completion(
                                    config=its_config,
                                    messages=messages,
                                    tools=tools,
                                    tool_choice=tool_choice,
                                    return_response_only=True
                                )

                                # Extract message and usage from result
                                response_message = result["message"]
                                usage = result["usage"]

                                # Create OpenAI-compatible response
                                openai_response = {
                                    "id": f"chatcmpl-its-{hash(request_id)}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": its_config.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": response_message,
                                            "finish_reason": "stop"
                                        }
                                    ],
                                    "usage": usage  # Aggregated usage from all LLM calls
                                }

                                # Convert to JSON
                                response_body = json.dumps(openai_response).encode('utf-8')

                                logger.info(
                                    f"[{request_id}] ITS complete, returning response "
                                    f"({len(response_body)} bytes, usage={usage})"
                                )

                                # Return immediate response (replaces upstream call)
                                yield ext_proc_pb2.ProcessingResponse(
                                    request_body=ext_proc_pb2.BodyResponse(
                                        response=ext_proc_pb2.CommonResponse(
                                            status=ext_proc_pb2.CommonResponse.CONTINUE_AND_REPLACE,
                                            body_mutation=ext_proc_pb2.BodyMutation(
                                                body=response_body
                                            ),
                                            header_mutation=ext_proc_pb2.HeaderMutation(
                                                set_headers=[
                                                    base_pb2.HeaderValueOption(
                                                        header=base_pb2.HeaderValue(
                                                            key="content-type",
                                                            raw_value=b"application/json"
                                                        )
                                                    ),
                                                    base_pb2.HeaderValueOption(
                                                        header=base_pb2.HeaderValue(
                                                            key="x-its-applied",
                                                            raw_value=b"true"
                                                        )
                                                    )
                                                ]
                                            )
                                        )
                                    )
                                )

                            except Exception as e:
                                logger.error(
                                    f"[{request_id}] Error processing ITS request: {e}",
                                    exc_info=True
                                )
                                # On error, pass through to upstream
                                yield ext_proc_pb2.ProcessingResponse(
                                    request_body=ext_proc_pb2.BodyResponse(
                                        response=ext_proc_pb2.CommonResponse(
                                            status=ext_proc_pb2.CommonResponse.CONTINUE
                                        )
                                    )
                                )

                        else:
                            # No ITS config, pass through
                            logger.debug(f"[{request_id}] Passing through without ITS")
                            yield ext_proc_pb2.ProcessingResponse(
                                request_body=ext_proc_pb2.BodyResponse(
                                    response=ext_proc_pb2.CommonResponse(
                                        status=ext_proc_pb2.CommonResponse.CONTINUE
                                    )
                                )
                            )

                    else:
                        # More body chunks coming, continue
                        yield ext_proc_pb2.ProcessingResponse(
                            request_body=ext_proc_pb2.BodyResponse(
                                response=ext_proc_pb2.CommonResponse(
                                    status=ext_proc_pb2.CommonResponse.CONTINUE
                                )
                            )
                        )

                # Handle response headers (pass through)
                elif request.HasField("response_headers"):
                    logger.debug(f"[{request_id}] Received response headers, passing through")
                    yield ext_proc_pb2.ProcessingResponse(
                        response_headers=ext_proc_pb2.HeadersResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

                # Handle response body (pass through)
                elif request.HasField("response_body"):
                    logger.debug(f"[{request_id}] Received response body, passing through")
                    yield ext_proc_pb2.ProcessingResponse(
                        response_body=ext_proc_pb2.BodyResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            # Let Envoy handle the error

        logger.info(f"[{request_id}] Stream completed")

    def _parse_its_headers(self, headers: dict[str, str]) -> Optional[ITSRequestConfig]:
        """
        Parse ITS configuration from request headers.

        Note: Model is NOT extracted from headers - it will be set from request body later.

        Args:
            headers: Dictionary of lowercase header names to values

        Returns:
            ITSRequestConfig (without model) if required headers present, None otherwise
        """
        # Check for required ITS headers
        budget_str = headers.get("x-its-budget")
        endpoint = headers.get("x-its-endpoint")

        # If any required header is missing, return None (no ITS)
        if not all([budget_str, endpoint]):
            return None

        try:
            # Parse budget as integer
            budget = int(budget_str)

            # Validate budget range (Pydantic will also validate, but fail early)
            if budget < 1 or budget > 1000:
                logger.error(f"Budget out of range (1-1000): {budget}")
                return None

            # Optional headers
            api_key = headers.get("x-its-api-key")

            # Create config without model (will be set from body later)
            config = ITSRequestConfig(
                budget=budget,
                api_endpoint=endpoint,
                api_key=api_key,
                model=None  # Will be populated from request body
            )

            return config

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse ITS headers: {e}")
            return None

    def shutdown(self):
        """Cleanup resources on service shutdown."""
        logger.info("External Processor shutting down")
        self.orchestrator.shutdown()


async def serve(port: int = 50051):
    """Start the gRPC server."""
    server = grpc.aio.server()

    # Create the processor
    processor = ExternalProcessorService()

    # Add to server
    ext_proc_grpc.add_ExternalProcessorServicer_to_server(processor, server)
    server.add_insecure_port(f'[::]:{port}')

    logger.info(f"Starting External Processor on port {port}")
    await server.start()

    logger.info("External Processor is ready to receive requests from Envoy")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        processor.shutdown()
        await server.stop(grace=5)


def main():
    """Entry point for the external processor service."""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Service stopped")


if __name__ == "__main__":
    main()