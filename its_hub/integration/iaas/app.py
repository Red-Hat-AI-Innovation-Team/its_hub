"""Inference-as-a-Service (IaaS) FastAPI application.

Provides an OpenAI-compatible API server for inference-time scaling algorithms.
Delegates to ITSGateway for LM lifecycle management and algorithm dispatch,
following the same adapter pattern as the Envoy ext_proc integration.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from its_hub.api.types import ITSRequestConfig
from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.gateway import ITSGateway
from its_hub.integration.iaas.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ConfigRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class _ServiceConfig:
    """Mutable service-level defaults set via /configure."""

    endpoint: str = ""
    model: str = ""
    api_key: str | None = None
    budget: int = 4
    temperature: float | None = None


class _ServiceState:
    """Encapsulates all mutable service state (replaces module-level globals)."""

    def __init__(self):
        self.gateway: ITSGateway = ITSGateway(algorithm=SelfConsistency())
        self.config = _ServiceConfig()

    def reset(self):
        self.gateway = ITSGateway(algorithm=SelfConsistency())
        self.config = _ServiceConfig()


_state = _ServiceState()


@asynccontextmanager
async def _lifespan(application: FastAPI):
    yield
    await _state.gateway.ashutdown()


app = FastAPI(
    title="its_hub Inference-as-a-Service",
    description="OpenAI-compatible API for inference-time scaling algorithms",
    version="0.1.0-alpha",
    lifespan=_lifespan,
)


def _build_its_config(
    request: ChatCompletionRequest,
    its_budget: int | None = None,
    its_endpoint: str | None = None,
    its_api_key: str | None = None,
) -> ITSRequestConfig:
    """Build ITSRequestConfig merging headers, body, and service defaults.

    Priority: header > body > service default.
    """
    if its_budget is not None:
        budget = its_budget
    elif request.budget is not None:
        budget = request.budget
    else:
        budget = _state.config.budget

    api_endpoint = its_endpoint or _state.config.endpoint
    api_key = its_api_key if its_api_key is not None else _state.config.api_key

    return ITSRequestConfig(
        budget=budget,
        api_endpoint=api_endpoint,
        model=request.model,
        api_key=api_key,
    )


@app.post("/configure", status_code=status.HTTP_200_OK)
async def config_service(request: ConfigRequest) -> dict[str, str]:
    """Configure the IaaS service with language model and scaling algorithm."""
    try:
        _state.gateway.configure(
            alg=request.alg,
            regex_patterns=request.regex_patterns,
            tool_vote=request.tool_vote,
            exclude_tool_args=request.exclude_tool_args,
        )
        _state.config.endpoint = request.endpoint
        _state.config.model = request.model
        _state.config.api_key = request.api_key
        if request.budget is not None:
            _state.config.budget = request.budget
        if request.temperature is not None:
            _state.config.temperature = request.temperature

        logger.info(
            "Configured IaaS: model=%s, alg=%s, budget=%s",
            request.model,
            request.alg,
            _state.config.budget,
        )
        return {
            "status": "success",
            "message": f"Initialized {request.model} with {request.alg} algorithm",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Configuration failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration failed. Check server logs for details.",
        ) from e


@app.get("/v1/models")
async def list_models() -> dict[str, list[dict[str, str]]]:
    """List available models (OpenAI-compatible endpoint)."""
    if _state.config.model:
        return {
            "data": [
                {
                    "id": _state.config.model,
                    "object": "model",
                    "owned_by": "its_hub",
                }
            ]
        }
    return {"data": []}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_its_budget: int | None = Header(None),
    x_its_endpoint: str | None = Header(None),
    x_its_api_key: str | None = Header(None),
) -> ChatCompletionResponse | StreamingResponse:
    """Generate chat completion with inference-time scaling."""
    if request.stream:
        return await _stream_chat_completions(
            request, x_its_budget, x_its_endpoint, x_its_api_key
        )

    try:
        its_config = _build_its_config(
            request, x_its_budget, x_its_endpoint, x_its_api_key
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    try:
        messages_dicts = [msg.to_dict() for msg in request.messages]

        result = await _state.gateway.arun_chat_completion(
            config=its_config,
            messages=messages_dicts,
            tools=request.tools,
            tool_choice=request.tool_choice,
            return_response_only=request.return_response_only,
        )

        if request.return_response_only:
            response_message = result["message"]
        else:
            response_message = result["the_one"]

        usage = result.get("usage", {})

        metadata = None
        if not request.return_response_only:
            metadata = {
                "algorithm": "self-consistency",
                "all_responses": result.get("responses"),
                "response_counts": result.get("response_counts"),
                "selected_index": result.get("selected_index"),
            }

        return ChatCompletionResponse(
            id=f"chatcmpl-its-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            metadata=metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Chat completion failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation failed. Check server logs for details.",
        ) from e


async def _stream_chat_completions(
    request: ChatCompletionRequest,
    its_budget: int | None = None,
    its_endpoint: str | None = None,
    its_api_key: str | None = None,
) -> StreamingResponse:
    """Handle streaming requests by buffering ITS result then sending as SSE chunks."""

    async def _generate():
        response_id = f"chatcmpl-its-{uuid.uuid4()}"
        created = int(time.time())

        its_config = _build_its_config(request, its_budget, its_endpoint, its_api_key)

        if not its_config.api_endpoint:
            yield f"data: {json.dumps({'error': 'Service not configured'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        messages_dicts = [msg.to_dict() for msg in request.messages]

        try:
            result = await _state.gateway.arun_chat_completion(
                config=its_config,
                messages=messages_dicts,
                tools=request.tools,
                tool_choice=request.tool_choice,
                return_response_only=True,
            )
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        response_message = result["message"]
        content = response_message.get("content")
        tool_calls = response_message.get("tool_calls")

        if tool_calls:
            for i, tc in enumerate(tool_calls):
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc.get(
                                            "id", f"call_{uuid.uuid4().hex[:24]}"
                                        ),
                                        "type": "function",
                                        "function": {
                                            "name": tc.get("function", {}).get(
                                                "name", ""
                                            ),
                                            "arguments": tc.get("function", {}).get(
                                                "arguments", "{}"
                                            ),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": content},
                        "finish_reason": None,
                    }
                ],
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
