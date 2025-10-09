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
from its_hub.algorithms.self_consistency import (
    SelfConsistency,
    create_regex_projection_function,
)
from its_hub.lms import OpenAICompatibleLanguageModel, LiteLLMLanguageModel, StepGeneration
from its_hub.types import ChatMessage, ChatMessages

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
LM_DICT: dict[str, OpenAICompatibleLanguageModel | LiteLLMLanguageModel] = {}
SCALING_ALG: Any | None = None  # TODO: Add proper type annotation


class ConfigRequest(BaseModel):
    """Configuration request for setting up the IaaS service."""

    provider: str = Field("openai", description="LM provider: 'openai' or 'litellm'")
    endpoint: str = Field(..., description="Language model endpoint URL")
    api_key: str | None = Field(None, description="API key for the language model")
    model: str = Field(..., description="Model name identifier")
    alg: str = Field(..., description="Scaling algorithm to use")
    extra_args: dict[str, Any] | None = Field(None, description="Additional provider-specific arguments")
    step_token: str | None = Field(None, description="Token to mark generation steps")
    stop_token: str | None = Field(None, description="Token to stop generation")
    rm_name: str | None = Field(
        None, description="Reward model name (not required for self-consistency)"
    )
    rm_device: str | None = Field(
        None, description="Device for reward model (e.g., 'cuda:0')"
    )
    rm_agg_method: str | None = Field(
        None, description="Reward model aggregation method"
    )
    regex_patterns: list[str] | None = Field(
        None, description="Regex patterns for self-consistency projection function"
    )
    tool_vote: str | None = Field(
        None,
        description="Tool voting strategy: 'tool_name', 'tool_args', 'tool_hierarchical'",
    )
    exclude_args: list[str] | None = Field(
        None,
        description="Argument names to exclude from tool voting (e.g., timestamp, id)",
    )
    use_llm_judge: bool = Field(
        False,
        description="Use LLM judge instead of process reward model for best-of-n"
    )
    judge_type: str | None = Field(
        "pointwise",
        description="Type of judge: 'pointwise' (scores individually) or 'groupwise' (ranks and selects top-N)"
    )
    judge_model: str | None = Field(
        None,
        description="LiteLLM model name for judge (e.g., 'gpt-4o-mini', 'claude-3-sonnet-20240229')"
    )
    judge_criterion: str | None = Field(
        "overall_quality",
        description="Evaluation criterion (overall_quality, technical_quality, writing_quality, relevance_quality, tool-judge) or custom criterion name"
    )
    judge_custom_prompt: str | None = Field(
        None,
        description="Custom evaluation prompt text. If provided, will create a custom criterion with judge_criterion as the name."
    )
    judge_api_key: str | None = Field(
        None,
        description="API key for judge model provider"
    )
    judge_base_url: str | None = Field(
        None,
        description="Base URL for custom judge endpoint"
    )
    judge_temperature: float | None = Field(
        0.0,
        description="Temperature for judge model (0.0 for deterministic)"
    )
    judge_max_tokens: int | None = Field(
        512,
        description="Maximum tokens for judge response"
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
            raise ValueError(
                "regex_patterns are required when using self-consistency algorithm"
            )
        return v

    @field_validator("rm_name")
    @classmethod
    def validate_rm_name(cls, v, info):
        """Validate reward model name is provided for algorithms that need it."""
        alg = info.data.get("alg")
        use_llm_judge = info.data.get("use_llm_judge", False)

        # For best-of-n, either rm_name or judge_model is required
        if alg == "best-of-n":
            if not use_llm_judge and not v:
                raise ValueError("rm_name is required when using best-of-n without LLM judge")
        # For particle-filtering, always need rm_name
        elif alg == "particle-filtering" and not v:
            raise ValueError("rm_name is required when using particle-filtering algorithm")
        return v

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, v, info):
        """Validate judge model is provided when using LLM judge."""
        use_llm_judge = info.data.get("use_llm_judge", False)
        if use_llm_judge and not v:
            raise ValueError("judge_model is required when use_llm_judge is True")
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
        # Configure language model based on provider
        if request.provider == "litellm":
            extra_kwargs = request.extra_args or {}
            lm = LiteLLMLanguageModel(
                model_name=request.model,
                api_key=request.api_key,
                api_base=request.endpoint if request.endpoint != "auto" else None,
                is_async=True,  # Enable async mode for better performance
                **extra_kwargs
            )
        else:
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
            if request.use_llm_judge:
                # Use LLM Judge adapter from its_hub integration
                try:
                    from its_hub.integration.reward_hub import LLMJudgeRewardModel
                    from reward_hub import AutoJudge
                except ImportError as e:
                    logger.error(f"Failed to import LLM Judge: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="LLM Judge integration not available",
                    ) from e

                # Register custom criterion if custom prompt provided
                if request.judge_custom_prompt:
                    logger.info(
                        f"Registering custom criterion: {request.judge_criterion}"
                    )
                    AutoJudge.register_criterion(
                        name=request.judge_criterion,
                        prompt_text=request.judge_custom_prompt,
                        description=f"Custom criterion: {request.judge_criterion}",
                    )

                logger.info(
                    f"Configuring LLM Judge: model={request.judge_model}, "
                    f"criterion={request.judge_criterion}"
                )

                # Create LLM Judge using the adapter (handles ChatMessages conversion)
                orm = LLMJudgeRewardModel(
                    model=request.judge_model,
                    criterion=request.judge_criterion,
                    api_key=request.judge_api_key,
                    base_url=request.judge_base_url,
                    temperature=request.judge_temperature,
                    max_tokens=request.judge_max_tokens,
                )
            else:
                # Use traditional process reward model
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
            if request.regex_patterns:
                projection_func = create_regex_projection_function(
                    request.regex_patterns
                )
            else:
                projection_func = None
            SCALING_ALG = SelfConsistency(
                projection_func,
                tool_vote=request.tool_vote,
                exclude_args=request.exclude_args,
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
    budget: int = Field(
        8, ge=1, le=1000, description="Computational budget for scaling"
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
    from its_hub.algorithms.self_consistency import SelfConsistencyResult

    if isinstance(algorithm_result, SelfConsistencyResult):
        return {
            "algorithm": "self-consistency",
            "all_responses": algorithm_result.responses,  # Now contains full message dicts with tool calls
            "response_counts": dict(algorithm_result.response_counts),
            "selected_index": algorithm_result.selected_index,
        }

    # TODO: Add metadata extraction for other algorithm result types
    # elif isinstance(algorithm_result, BestOfNResult):
    #     return {
    #         "algorithm": "best-of-n",
    #         "scores": algorithm_result.scores,
    #         "selected_index": algorithm_result.selected_index,
    #         ...
    #     }
    # elif isinstance(algorithm_result, BeamSearchResult):
    #     return {...}
    # elif isinstance(algorithm_result, ParticleGibbsResult):
    #     return {...}

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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate chat completion with inference-time scaling."""
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming responses not yet implemented",
        )

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
        if request.temperature is not None:
            lm.temperature = request.temperature

        # Create ChatMessages from the full conversation history
        # Convert Pydantic ChatMessage objects to list if needed
        chat_messages = ChatMessages(list(request.messages))

        logger.info(
            f"Processing request for model={request.model}, budget={request.budget}"
        )

        # Generate response using scaling algorithm with full conversation context
        algorithm_result = await SCALING_ALG.ainfer(
            lm,
            chat_messages,
            request.budget,
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

        logger.info(
            f"Successfully generated response (content length: {len(response_message.get('content') or '')})"
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
