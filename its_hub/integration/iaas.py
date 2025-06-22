"""Inference-as-a-Service (IaaS) integration

Provides an OpenAI-compatible API server for inference-time scaling algorithms.
"""

import logging
import time
import uuid
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from its_hub.algorithms import BestOfN, ParticleFiltering
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.types import ChatMessage

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


class ConfigRequest(BaseModel):
    """Configuration request for setting up the IaaS service."""

    endpoint: str = Field(..., description="Language model endpoint URL")
    api_key: str = Field(..., description="API key for the language model")
    model: str = Field(..., description="Model name identifier")
    alg: str = Field(..., description="Scaling algorithm to use")
    step_token: str | None = Field(None, description="Token to mark generation steps")
    stop_token: str | None = Field(None, description="Token to stop generation")
    rm_name: str = Field(..., description="Reward model name")
    rm_device: str = Field(..., description="Device for reward model (e.g., 'cuda:0')")
    rm_agg_method: str | None = Field(
        None, description="Reward model aggregation method"
    )

    @field_validator("alg")
    @classmethod
    def validate_algorithm(cls, v):
        """Validate that the algorithm is supported."""
        supported_algs = {"particle-filtering", "best-of-n"}
        if v not in supported_algs:
            raise ValueError(
                f"Algorithm '{v}' not supported. Choose from: {supported_algs}"
            )
        return v


@app.post("/configure", status_code=status.HTTP_200_OK)
async def config_service(request: ConfigRequest) -> dict[str, str]:
    """Configure the IaaS service with language model and scaling algorithm."""
    try:
        from its_hub.integration.reward_hub import (
            AggregationMethod,
            LocalVllmProcessRewardModel,
        )
    except ImportError as e:
        logger.error(f"Failed to import reward_hub: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reward hub integration not available",
        ) from e

    global LM_DICT, SCALING_ALG

    logger.info(f"Configuring service with model={request.model}, alg={request.alg}")

    try:
        # Configure language model
        lm = OpenAICompatibleLanguageModel(
            endpoint=request.endpoint,
            api_key=request.api_key,
            model_name=request.model,
            # TODO: Consider enabling async mode for better performance
        )
        LM_DICT[request.model] = lm

        # Configure scaling algorithm
        if request.alg == "particle-filtering":
            # TODO: Make these parameters configurable
            sg = StepGeneration(
                request.step_token,
                max_steps=50,  # TODO: Make configurable
                stop_token=request.stop_token,
                temperature=0.001,  # Low temp for deterministic step generation
                include_stop_str_in_output=True,
                # TODO: Make thinking token markers configurable
                temperature_switch=(0.8, "<boi>", "<eoi>"),  # Higher temp for thinking
            )
            prm = LocalVllmProcessRewardModel(
                model_name=request.rm_name,
                device=request.rm_device,
                aggregation_method=AggregationMethod(request.rm_agg_method or "model"),
            )
            SCALING_ALG = ParticleFiltering(sg, prm)

        elif request.alg == "best-of-n":
            prm = LocalVllmProcessRewardModel(
                model_name=request.rm_name,
                device=request.rm_device,
                aggregation_method=AggregationMethod("model"),
            )
            # TODO: Consider separating outcome and process reward model interfaces
            orm = prm  # Using process reward model as outcome reward model
            SCALING_ALG = BestOfN(orm)

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
    budget: int = Field(
        8, ge=1, le=1000, description="Computational budget for scaling"
    )
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    stream: bool | None = Field(False, description="Stream response (not implemented)")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate message format and constraints."""
        if not v:
            raise ValueError("At least one message is required")
        if len(v) > 2:
            raise ValueError("Maximum 2 messages supported (optional system + user)")
        if v[-1].role != "user":
            raise ValueError("Last message must be from user")
        if len(v) == 2 and v[0].role != "system":
            raise ValueError("First message must be system when using 2 messages")
        return v


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(..., description="Reason for completion")


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Generated tokens")
    total_tokens: int = Field(..., description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Unique response identifier")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[ChatCompletionChoice] = Field(..., description="Generated choices")
    usage: ChatCompletionUsage = Field(..., description="Token usage statistics")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completion with inference-time scaling."""
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming responses not yet implemented",
        )

    if SCALING_ALG is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured. Please call /configure first.",
        )

    try:
        lm = LM_DICT[request.model]
    except KeyError:
        available_models = list(LM_DICT.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        ) from None

    try:
        # Configure language model for this request
        # FIXME: Mutating the shared lm instance is not thread-safe and can cause race conditions
        if request.temperature is not None:
            lm.temperature = request.temperature

        # Set system prompt if provided
        if len(request.messages) == 2:
            lm.system_prompt = request.messages[0].content
        else:
            lm.system_prompt = None

        # Extract user prompt
        prompt = request.messages[-1].content

        logger.info(
            f"Processing request for model={request.model}, budget={request.budget}"
        )

        # Generate response using scaling algorithm
        response_content = SCALING_ALG.infer(lm, prompt, request.budget)

        # TODO: Implement proper token counting
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=0,  # TODO: Implement token counting
                completion_tokens=0,  # TODO: Implement token counting
                total_tokens=0,  # TODO: Implement token counting
            ),
        )

        logger.info(
            f"Successfully generated response (length: {len(response_content)})"
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
