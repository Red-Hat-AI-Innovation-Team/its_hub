"""Inference-as-a-Service (IaaS) integration

Provides an OpenAI-compatible API server for inference-time scaling algorithms.
"""

import json
import logging
import time
import uuid
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from its_hub import OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api.types import ChatMessage, ChatMessages
from its_hub.core.algorithms.self_consistency import (
    SelfConsistencyResult,
    create_regex_projection_function,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="its_hub Inference-as-a-Service",
    description="OpenAI-compatible API for inference-time scaling algorithms",
    version="0.1.0-alpha",
)

# Global state - TODO: Replace with proper dependency injection in production
LM_DICT: dict[str, OpenAICompatibleLanguageModel] = {}
SCALING_ALG: Any | None = None  # TODO: Add proper type annotation
CONFIGURED_BUDGET: int = 4  # Default budget, overridden by /configure
CONFIGURED_TEMPERATURE: float | None = None  # Default temperature, overridden by /configure


class ConfigRequest(BaseModel):
    """Configuration request for setting up the IaaS service."""

    provider: str = Field("openai", description="LM provider: 'openai'")
    endpoint: str = Field(..., description="Language model endpoint URL")
    api_key: str | None = Field(None, description="API key for the language model")
    model: str = Field(..., description="Model name identifier")
    alg: str = Field(..., description="Scaling algorithm to use")
    extra_args: dict[str, Any] | None = Field(
        None, description="Additional provider-specific arguments"
    )
    step_token: str | None = Field(None, description="Token to mark generation steps")
    stop_token: str | None = Field(None, description="Token to stop generation")
    regex_patterns: list[str] | None = Field(
        None, description="Regex patterns for self-consistency projection function"
    )
    budget: int | None = Field(
        None,
        description="Default budget for requests that don't specify one",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature (overrides per-request temperature from upstream)",
    )
    tool_vote: str | None = Field(
        None,
        description="Tool voting strategy: 'tool_name', 'tool_args', 'tool_hierarchical'",
    )
    exclude_tool_args: list[str] | None = Field(
        None,
        description="Tool argument names to exclude from voting (e.g., ['timestamp', 'id'])",
    )

    @field_validator("alg")
    @classmethod
    def validate_algorithm(cls, v):
        """Validate that the algorithm is supported."""
        supported_algs = {"self-consistency"}
        if v not in supported_algs:
            raise ValueError(
                f"Algorithm '{v}' not supported. Choose from: {supported_algs}"
            )
        return v

    @field_validator("regex_patterns")
    @classmethod
    def validate_regex_patterns(cls, v, info):
        """Validate regex patterns are provided when using self-consistency."""
        if info.data.get("alg") == "self-consistency" and not v:
            raise ValueError(
                "regex_patterns are required when using self-consistency algorithm"
            )
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v, info):
        """Validate api_key is provided when using OpenAI provider."""
        provider = info.data.get("provider", "openai")
        if provider == "openai" and not v:
            raise ValueError("api_key is required when using openai provider")
        return v


@app.post("/configure", status_code=status.HTTP_200_OK)
async def config_service(request: ConfigRequest) -> dict[str, str]:
    """Configure the IaaS service with language model and scaling algorithm."""

    global LM_DICT, SCALING_ALG, CONFIGURED_BUDGET, CONFIGURED_TEMPERATURE

    if request.budget is not None:
        CONFIGURED_BUDGET = request.budget
    if request.temperature is not None:
        CONFIGURED_TEMPERATURE = request.temperature

    logger.info(f"Configuring service with model={request.model}, alg={request.alg}, budget={CONFIGURED_BUDGET}")

    try:
        # Configure language model based on provider
        # Default to OpenAI compatible
        lm = OpenAICompatibleLanguageModel(
            endpoint=request.endpoint,
            api_key=request.api_key,
            model_name=request.model,
            is_async=True,  # Enable async mode for better performance
            # SSL verification enabled by default (same as synchronous requests)
        )
        LM_DICT[request.model] = lm

        # Configure scaling algorithm
        if request.alg == "self-consistency":
            # Create projection function from regex patterns
            if request.regex_patterns:
                projection_func = create_regex_projection_function(
                    request.regex_patterns
                )
            else:
                projection_func = None
            SCALING_ALG = SelfConsistency(
                projection_func,
                tool_vote=request.tool_vote,
                exclude_args=request.exclude_tool_args,
            )

        logger.info(f"Successfully configured {request.alg} algorithm")
        return {
            "status": "success",
            "message": f"Initialized {request.model} with {request.alg} algorithm",
        }

    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration failed: {e!s}",
        ) from e


@app.get("/v1/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    """List available models (OpenAI-compatible endpoint)."""
    return {
        "data": [
            {"id": model, "object": "model", "owned_by": "its_hub"} for model in LM_DICT
        ]
    }


# Use the ChatMessage type from types.py directly


class ChatCompletionRequest(BaseModel):
    """Chat completion request with inference-time scaling support."""

    model: str = Field(..., description="Model identifier")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    budget: int | None = Field(
        None, ge=1, le=1000, description="Computational budget for scaling"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    stream: bool | None = Field(False, description="Stream response (not implemented)")
    tools: list[dict[str, Any]] | None = Field(
        None, description="Available tools for the model to call"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool choice strategy ('auto', 'none', or specific tool)"
    )
    return_response_only: bool = Field(
        True, description="Return only final response or include algorithm metadata"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate message format - flexible validation for various conversation formats."""
        if not v:
            raise ValueError("At least one message is required")
        return v


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: dict = Field(..., description="Generated message in OpenAI format")
    finish_reason: str = Field(..., description="Reason for completion")


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Generated tokens")
    total_tokens: int = Field(..., description="Total tokens used")


def _extract_algorithm_metadata(algorithm_result: Any) -> dict[str, Any] | None:
    """Extract metadata from algorithm results for API response."""
    if isinstance(algorithm_result, SelfConsistencyResult):
        return {
            "algorithm": "self-consistency",
            "all_responses": algorithm_result.responses,  # Now contains full message dicts with tool calls
            "response_counts": dict(algorithm_result.response_counts),
            "selected_index": algorithm_result.selected_index,
        }

    return None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Unique response identifier")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChoice] = Field(..., description="Generated choices")
    usage: ChatCompletionUsage = Field(..., description="Token usage statistics")
    metadata: dict[str, Any] | None = Field(
        None, description="Algorithm-specific metadata"
    )


async def _stream_chat_completions(request: ChatCompletionRequest) -> StreamingResponse:
    """Handle streaming requests by buffering ITS result then sending as SSE chunks."""

    async def _generate():
        response_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        try:
            lm = LM_DICT[request.model]
        except KeyError:
            yield f"data: {json.dumps({'error': f'Model {request.model} not found'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        if SCALING_ALG is None:
            yield f"data: {json.dumps({'error': 'Service not configured'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        effective_temp = CONFIGURED_TEMPERATURE if CONFIGURED_TEMPERATURE is not None else request.temperature
        if effective_temp is not None:
            lm.temperature = effective_temp

        chat_messages = ChatMessages(list(request.messages))

        effective_budget = request.budget or CONFIGURED_BUDGET
        logger.info(
            f"Streaming request: model={request.model}, budget={effective_budget}, temperature={effective_temp} (configured={CONFIGURED_TEMPERATURE}, request={request.temperature}), tool_vote={getattr(SCALING_ALG, 'tool_vote', None)}"
        )
        algorithm_result = await SCALING_ALG.ainfer(
            lm,
            chat_messages,
            effective_budget,
            return_response_only=True,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

        response_message = algorithm_result
        content = response_message.get("content")
        tool_calls = response_message.get("tool_calls")

        if tool_calls:
            for i, tc in enumerate(tool_calls):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": i,
                                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                                },
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            done_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"

        elif content:
            content_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

            done_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completion with inference-time scaling."""
    if request.stream:
        return await _stream_chat_completions(request)

    try:
        lm = LM_DICT[request.model]
    except KeyError:
        available_models = list(LM_DICT.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        ) from None

    if SCALING_ALG is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured. Please call /configure first.",
        )

    try:
        # Configure language model for this request
        effective_temp = CONFIGURED_TEMPERATURE if CONFIGURED_TEMPERATURE is not None else request.temperature
        if effective_temp is not None:
            lm.temperature = effective_temp

        # Create ChatMessages from the full conversation history
        # Convert Pydantic ChatMessage objects to list if needed
        chat_messages = ChatMessages(list(request.messages))

        effective_budget = request.budget or CONFIGURED_BUDGET
        logger.info(
            f"Processing request: model={request.model}, budget={effective_budget}, temperature={effective_temp} (configured={CONFIGURED_TEMPERATURE}, request={request.temperature}), tool_vote={getattr(SCALING_ALG, 'tool_vote', None)}"
        )

        # Generate response using scaling algorithm with full conversation context
        algorithm_result = await SCALING_ALG.ainfer(
            lm,
            chat_messages,
            effective_budget,
            return_response_only=request.return_response_only,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

        # Extract response content and metadata
        if not request.return_response_only and hasattr(algorithm_result, "the_one"):
            # Got a full result object
            response_message = algorithm_result.the_one
            metadata = _extract_algorithm_metadata(algorithm_result)
        else:
            # Got just a message dict response
            response_message = algorithm_result
            metadata = None

        # Use the selected response directly without any modification
        response_chat_message = response_message

        # TODO: Implement proper token counting
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_chat_message,
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=0,  # TODO: Implement token counting
                completion_tokens=0,  # TODO: Implement token counting
                total_tokens=0,  # TODO: Implement token counting
            ),
            metadata=metadata,
        )

        # Log response with content info
        content = response_message.get("content")
        if isinstance(content, list):
            has_image = any(
                item.get("type") == "image_url"
                for item in content
                if isinstance(item, dict)
            )
            text_content = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            img_note = " (with images)" if has_image else ""
            logger.info(
                f"Successfully generated response (length: {len(text_content)}{img_note})"
            )
        else:
            logger.info(
                f"Successfully generated response (content length: {len(content or '')})"
            )
        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e!s}",
        ) from e


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server to")
@click.option("--port", default=8000, help="Port to bind the server to")
@click.option("--dev", is_flag=True, help="Run in development mode with auto-reload")
def main(host: str, port: int, dev: bool) -> None:
    """Start the its_hub Inference-as-a-Service API server."""
    print("\n" + "=" * 60)
    print("🚀 its_hub Inference-as-a-Service (IaaS) API Server")
    print("⚠️  ALPHA VERSION - Not for production use")
    print(f"📍 Starting server on {host}:{port}")
    print(f"📖 API docs available at: http://{host}:{port}/docs")
    print("=" * 60 + "\n")

    uvicorn_config = {
        "host": host,
        "port": port,
        "log_level": "info" if not dev else "debug",
    }

    if dev:
        logger.info("Running in development mode with auto-reload")
        uvicorn.run("its_hub.integration.iaas:app", reload=True, **uvicorn_config)
    else:
        uvicorn.run(app, **uvicorn_config)


if __name__ == "__main__":
    main()
