"""Inference-as-a-Service (IaaS) integration

Provides an OpenAI-compatible API server for inference-time scaling algorithms.
"""

import logging
import time
import uuid
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ValidationError

from its_hub.algorithms import BestOfN, ParticleFiltering
from its_hub.algorithms.self_consistency import (
    SelfConsistency,
    create_regex_projection_function,
)
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.types import ChatMessage, ChatCompletionMessage, ToolCall

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ Validation error: {exc.errors()}")
    logger.error(f"❌ Request body: {await request.body()}")
    return await request_validation_exception_handler(request, exc)


def _extract_algorithm_metadata(algorithm_result: Any) -> dict[str, Any] | None:
    """Extract metadata from algorithm results for API response."""
    from its_hub.algorithms.self_consistency import SelfConsistencyResult

    if isinstance(algorithm_result, SelfConsistencyResult):
        return {
            "algorithm": "self-consistency",
            "all_responses": algorithm_result.responses,
            "response_counts": dict(algorithm_result.response_counts),
            "selected_index": algorithm_result.selected_index,
        }

    # Add other algorithm result types here as needed
    # elif isinstance(algorithm_result, OtherAlgorithmResult):
    #     return {...}

    return None


class ConfigRequest(BaseModel):
    """Configuration request for setting up the IaaS service."""

    endpoint: str = Field(..., description="Language model endpoint URL")
    api_key: str = Field(..., description="API key for the language model")
    model: str = Field(..., description="Model name identifier")
    alg: str = Field(..., description="Scaling algorithm to use")
    step_token: str | None = Field(None, description="Token to mark generation steps")
    stop_token: str | None = Field(None, description="Token to stop generation")
    rm_name: str | None = Field(None, description="Reward model name (not required for self-consistency)")
    rm_device: str | None = Field(None, description="Device for reward model (e.g., 'cuda:0')")
    rm_agg_method: str | None = Field(
        None, description="Reward model aggregation method"
    )
    regex_patterns: list[str] | None = Field(
        None, description="Regex patterns for self-consistency projection function"
    )

    @field_validator("alg")
    @classmethod
    def validate_algorithm(cls, v):
        """Validate that the algorithm is supported."""
        supported_algs = {"particle-filtering", "best-of-n", "self-consistency"}
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
            raise ValueError("regex_patterns are required when using self-consistency algorithm")
        return v

    @field_validator("rm_name")
    @classmethod
    def validate_rm_name(cls, v, info):
        """Validate reward model name is provided for algorithms that need it."""
        alg = info.data.get("alg")
        if alg in {"particle-filtering", "best-of-n"} and not v:
            raise ValueError(f"rm_name is required when using {alg} algorithm")
        return v


@app.post("/configure", status_code=status.HTTP_200_OK)
async def config_service(request: ConfigRequest) -> dict[str, str]:
    """Configure the IaaS service with language model and scaling algorithm."""
    # Only import reward_hub if needed (not required for self-consistency)
    if request.alg in {"particle-filtering", "best-of-n"}:
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
                max_steps=50,  # TODO: Make configurable
                step_token=request.step_token,
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

        elif request.alg == "self-consistency":
            # Create projection function from regex patterns
            projection_func = create_regex_projection_function(request.regex_patterns)
            SCALING_ALG = SelfConsistency(projection_func)

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
    return_response_only: bool = Field(True, description="Return only the selected response (false to include algorithm metadata)")
    tools: list[dict[str, Any]] | None = Field(None, description="List of tools available for function calling")
    tool_choice: str | dict[str, Any] | None = Field(None, description="Controls which (if any) tool is called")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        """Validate message format and constraints."""
        if not v:
            raise ValueError("At least one message is required")
        # Note: Removed "last message must be user" constraint to support full conversation history
        return v


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatCompletionMessage = Field(..., description="Generated message")
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
    metadata: dict[str, Any] | None = Field(None, description="Algorithm-specific metadata")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completion with inference-time scaling."""
    logger.info(f"🔍 Received chat completion request: model={request.model}, budget={request.budget}, messages={len(request.messages)}")
    logger.info(f"🔍 Message types: {[msg.role for msg in request.messages]}")
    logger.info(f"🔍 Tools provided: {request.tools is not None}")
    
    if request.stream:
        logger.error("❌ Streaming not supported")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming responses not yet implemented",
        )

    if SCALING_ALG is None:
        logger.error("❌ Service not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured. Please call /configure first.",
        )

    try:
        logger.info(f"🔍 Looking up model: {request.model}")
        lm = LM_DICT[request.model]
        logger.info(f"✅ Model found: {request.model}")
    except KeyError:
        available_models = list(LM_DICT.keys())
        logger.error(f"❌ Model not found: {request.model}, available: {available_models}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        ) from None

    try:
        logger.info(f"🔍 Configuring language model...")
        # Configure language model for this request
        # FIXME: Mutating the shared lm instance is not thread-safe and can cause race conditions
        if request.temperature is not None:
            lm.temperature = request.temperature
            logger.info(f"🔍 Set temperature: {request.temperature}")

        # Handle full conversation history
        # Extract system message if present (first message with role="system")
        system_message = next((msg for msg in request.messages if msg.role == "system"), None)
        lm.system_prompt = system_message.content if system_message else None
        logger.info(f"🔍 System prompt: {lm.system_prompt is not None}")

        # Convert full conversation history to ChatMessage format for the algorithm
        logger.info(f"🔍 Converting {len(request.messages)} messages...")
        conversation_messages = []
        for i, msg in enumerate(request.messages):
            logger.info(f"🔍 Message {i}: role={msg.role}, content_len={len(msg.content or '')}, tool_calls={hasattr(msg, 'tool_calls') and msg.tool_calls is not None}, tool_call_id={hasattr(msg, 'tool_call_id') and msg.tool_call_id is not None}")
            # Convert to the ChatMessage format expected by the algorithm, preserving tool-related fields
            chat_msg = ChatMessage(
                role=msg.role,
                content=msg.content or "",  # Handle None content
                tool_calls=getattr(msg, 'tool_calls', None),  # Preserve tool_calls for assistant messages
                tool_call_id=getattr(msg, 'tool_call_id', None)  # Preserve tool_call_id for tool messages
            )
            conversation_messages.append(chat_msg)
        logger.info(f"✅ Converted {len(conversation_messages)} messages")

        logger.info(
            f"Processing request for model={request.model}, budget={request.budget}, messages={len(conversation_messages)}"
        )

        # Generate response using scaling algorithm with full conversation history
        # Pass tools and tool_choice to the language model if provided
        if request.tools is not None or request.tool_choice is not None:
            logger.info(f"🔍 Setting up tools wrapper (tools: {len(request.tools) if request.tools else 0}, tool_choice: {request.tool_choice is not None})")
            # Store original generate method
            original_generate = lm.generate
            
            # Create wrapper that passes tools
            def generate_with_tools(*args, **kwargs):
                if request.tools is not None:
                    kwargs['tools'] = request.tools
                    logger.info(f"🔍 Adding {len(request.tools)} tools to generation")
                if request.tool_choice is not None:
                    kwargs['tool_choice'] = request.tool_choice
                    logger.info(f"🔍 Adding tool_choice: {request.tool_choice}")
                return original_generate(*args, **kwargs)
            
            # Temporarily replace the generate method
            lm.generate = generate_with_tools
        
        logger.info(f"🔍 Calling scaling algorithm with budget={request.budget}")
        # Pass full conversation history instead of just the last message
        algorithm_result = SCALING_ALG.infer(lm, conversation_messages, request.budget, return_response_only=request.return_response_only)
        logger.info(f"✅ Algorithm completed successfully")
        
        # Restore original generate method if we modified it
        if request.tools is not None or request.tool_choice is not None:
            lm.generate = original_generate
            logger.info(f"🔍 Restored original generate method")

        # Extract response content and metadata
        if not request.return_response_only and hasattr(algorithm_result, 'the_one'):
            # Got a full result object
            response_message_obj = algorithm_result.the_one
            metadata = _extract_algorithm_metadata(algorithm_result)
        else:
            # Got just a message object
            response_message_obj = algorithm_result
            metadata = None

        # Convert message object to ChatCompletionMessage
        message = ChatCompletionMessage(
            role="assistant",
            content=response_message_obj.get("content"),
            tool_calls=[
                ToolCall(
                    id=tc["id"],
                    type=tc["type"],
                    function=tc["function"]
                ) for tc in response_message_obj.get("tool_calls", [])
            ] if response_message_obj.get("tool_calls") else None
        )

        # TODO: Implement proper token counting
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
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

        logger.info(
            f"Successfully generated response with {'tool calls' if message.tool_calls else 'content'}"
        )
        return response

    except Exception as e:
        logger.error(f"❌ Chat completion failed: {type(e).__name__}: {e}")
        logger.error(f"❌ Full error details: {repr(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
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
