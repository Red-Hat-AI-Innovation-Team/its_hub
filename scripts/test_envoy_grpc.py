#!/usr/bin/env python3
"""
Test client for Envoy External Processor gRPC service.

This script simulates how Envoy communicates with the ext_proc service,
allowing manual testing without running a full Envoy proxy.

Usage:
    python scripts/test_envoy_grpc.py
"""

import asyncio
import json
import logging
from typing import Optional

import grpc

# Import proto after setting up path
import its_hub.integration.ext_proc.proto
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_proc_grpc
from envoy.config.core.v3 import base_pb2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_header(key: str, value: str) -> base_pb2.HeaderValue:
    """Create a header value proto."""
    return base_pb2.HeaderValue(
        key=key,
        raw_value=value.encode('utf-8')
    )


def create_request_headers(
    path: str = "/v1/chat/completions",
    method: str = "POST",
    its_budget: Optional[int] = None,
    its_endpoint: Optional[str] = None,
    its_api_key: Optional[str] = None,
) -> ext_proc_pb2.ProcessingRequest:
    """
    Create a ProcessingRequest with headers.

    Args:
        path: Request path
        method: HTTP method
        its_budget: ITS budget header value
        its_endpoint: ITS endpoint header value
        its_api_key: ITS API key header value
    """
    headers = [
        create_header(":path", path),
        create_header(":method", method),
        create_header(":authority", "localhost:8080"),
        create_header("content-type", "application/json"),
    ]

    # Add ITS headers if provided
    if its_budget is not None:
        headers.append(create_header("x-its-budget", str(its_budget)))
    if its_endpoint is not None:
        headers.append(create_header("x-its-endpoint", its_endpoint))
    if its_api_key is not None:
        headers.append(create_header("x-its-api-key", its_api_key))

    return ext_proc_pb2.ProcessingRequest(
        request_headers=ext_proc_pb2.HttpHeaders(
            headers=base_pb2.HeaderMap(headers=headers)
        )
    )


def create_request_body(
    messages: list[dict],
    model: str,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str | dict] = None,
) -> ext_proc_pb2.ProcessingRequest:
    """
    Create a ProcessingRequest with body.

    Args:
        messages: Chat messages in OpenAI format
        model: Model name (standard OpenAI format)
        tools: Optional tool definitions
        tool_choice: Optional tool choice strategy
    """
    body_data = {
        "model": model,
        "messages": messages,
    }

    if tools is not None:
        body_data["tools"] = tools
    if tool_choice is not None:
        body_data["tool_choice"] = tool_choice

    body_json = json.dumps(body_data).encode('utf-8')

    return ext_proc_pb2.ProcessingRequest(
        request_body=ext_proc_pb2.HttpBody(
            body=body_json,
            end_of_stream=True
        )
    )


async def test_its_request(
    budget: int = 3,
    endpoint: str = "http://localhost:8100/v1",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
):
    """
    Test a request WITH ITS headers.

    This simulates a client that wants to use ITS for improved responses.
    """
    logger.info("=" * 80)
    logger.info("TEST: Request WITH ITS headers")
    logger.info("=" * 80)

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = ext_proc_grpc.ExternalProcessorStub(channel)

        # Create request messages
        messages = [
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ]

        # Build request stream
        requests = [
            create_request_headers(
                path="/v1/chat/completions",
                method="POST",
                its_budget=budget,
                its_endpoint=endpoint,
                its_api_key=api_key,
            ),
            create_request_body(
                messages=messages,
                model=model,
            ),
        ]

        logger.info(f"Sending request with ITS: budget={budget}, endpoint={endpoint}, model={model}")

        # Send request and collect responses
        try:
            response_count = 0
            async for response in stub.Process(iter(requests)):
                response_count += 1
                logger.info(f"\n--- Response {response_count} ---")

                if response.HasField("request_headers"):
                    logger.info("Response type: request_headers")
                    logger.info(f"Status: {response.request_headers.response.status}")

                elif response.HasField("request_body"):
                    logger.info("Response type: request_body")
                    logger.info(f"Status: {response.request_body.response.status}")

                    # Check if body was replaced (ITS applied)
                    if response.request_body.response.status == ext_proc_pb2.CommonResponse.CONTINUE_AND_REPLACE:
                        body = response.request_body.response.body_mutation.body.decode('utf-8')
                        logger.info("✓ ITS was applied! Response body:")
                        try:
                            response_json = json.loads(body)
                            logger.info(json.dumps(response_json, indent=2))
                        except json.JSONDecodeError:
                            logger.info(body)
                    else:
                        logger.info("Body passed through (no ITS)")

        except Exception as e:
            logger.error(f"Error during request: {e}", exc_info=True)


async def test_passthrough_request():
    """
    Test a request WITHOUT ITS headers.

    This should pass through without modification.
    """
    logger.info("=" * 80)
    logger.info("TEST: Request WITHOUT ITS headers (passthrough)")
    logger.info("=" * 80)

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = ext_proc_grpc.ExternalProcessorStub(channel)

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        requests = [
            create_request_headers(
                path="/v1/chat/completions",
                method="POST",
                # No ITS headers
            ),
            create_request_body(
                messages=messages,
                model="gpt-4",
            ),
        ]

        logger.info("Sending request WITHOUT ITS headers")

        try:
            response_count = 0
            async for response in stub.Process(iter(requests)):
                response_count += 1
                logger.info(f"\n--- Response {response_count} ---")

                if response.HasField("request_headers"):
                    logger.info("Response type: request_headers")
                    logger.info(f"Status: {response.request_headers.response.status}")

                elif response.HasField("request_body"):
                    logger.info("Response type: request_body")
                    status = response.request_body.response.status
                    logger.info(f"Status: {status}")

                    if status == ext_proc_pb2.CommonResponse.CONTINUE:
                        logger.info("✓ Request passed through (as expected)")
                    else:
                        logger.warning("⚠ Unexpected: Request was modified!")

        except Exception as e:
            logger.error(f"Error during request: {e}", exc_info=True)


async def test_missing_model_in_body():
    """
    Test a request WITH ITS headers but WITHOUT model in body.

    This should pass through because model is required.
    """
    logger.info("=" * 80)
    logger.info("TEST: Request WITH ITS headers but NO model in body")
    logger.info("=" * 80)

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = ext_proc_grpc.ExternalProcessorStub(channel)

        requests = [
            create_request_headers(
                path="/v1/chat/completions",
                method="POST",
                its_budget=3,
                its_endpoint="http://localhost:8100/v1",
            ),
            # Create body WITHOUT model field
            ext_proc_pb2.ProcessingRequest(
                request_body=ext_proc_pb2.HttpBody(
                    body=json.dumps({"messages": [{"role": "user", "content": "test"}]}).encode('utf-8'),
                    end_of_stream=True
                )
            ),
        ]

        logger.info("Sending request with ITS headers but no model in body")

        try:
            response_count = 0
            async for response in stub.Process(iter(requests)):
                response_count += 1
                logger.info(f"\n--- Response {response_count} ---")

                if response.HasField("request_body"):
                    status = response.request_body.response.status
                    logger.info(f"Status: {status}")

                    if status == ext_proc_pb2.CommonResponse.CONTINUE:
                        logger.info("✓ Request passed through (missing model, as expected)")
                    else:
                        logger.warning("⚠ Unexpected: Request was modified despite missing model!")

        except Exception as e:
            logger.error(f"Error during request: {e}", exc_info=True)


async def test_non_chat_endpoint():
    """
    Test a request to a non-chat endpoint.

    This should pass through immediately.
    """
    logger.info("=" * 80)
    logger.info("TEST: Request to non-chat endpoint")
    logger.info("=" * 80)

    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = ext_proc_grpc.ExternalProcessorStub(channel)

        requests = [
            create_request_headers(
                path="/v1/embeddings",  # Different endpoint
                method="POST",
                its_budget=3,
                its_endpoint="http://localhost:8100/v1",
            ),
        ]

        logger.info("Sending request to /v1/embeddings endpoint")

        try:
            response_count = 0
            async for response in stub.Process(iter(requests)):
                response_count += 1
                logger.info(f"\n--- Response {response_count} ---")

                if response.HasField("request_headers"):
                    status = response.request_headers.response.status
                    logger.info(f"Status: {status}")

                    if status == ext_proc_pb2.CommonResponse.CONTINUE:
                        logger.info("✓ Request passed through (non-chat endpoint, as expected)")
                    else:
                        logger.warning("⚠ Unexpected response status!")

        except Exception as e:
            logger.error(f"Error during request: {e}", exc_info=True)


async def main():
    """Run all test scenarios."""
    logger.info("\n" + "=" * 80)
    logger.info("ENVOY EXTERNAL PROCESSOR TEST CLIENT")
    logger.info("=" * 80)
    logger.info("\nMake sure the ext_proc service is running:")
    logger.info("  uv run envoy-grpc")
    logger.info("\n" + "=" * 80 + "\n")

    # Wait a bit for user to read
    await asyncio.sleep(2)

    # Run test scenarios
    try:
        # Test 1: ITS request (will fail without real LLM, but tests the flow)
        await test_its_request(
            budget=3,
            endpoint="http://localhost:8100/v1",
            model="test-model",
        )

        await asyncio.sleep(1)

        # Test 2: Passthrough request
        await test_passthrough_request()

        await asyncio.sleep(1)

        # Test 3: Missing model in body
        await test_missing_model_in_body()

        await asyncio.sleep(1)

        # Test 4: Non-chat endpoint
        await test_non_chat_endpoint()

        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 80)

    except grpc.aio.AioRpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            logger.error("\n❌ Cannot connect to ext_proc service!")
            logger.error("   Make sure it's running: uv run envoy-grpc")
        else:
            logger.error(f"gRPC error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
