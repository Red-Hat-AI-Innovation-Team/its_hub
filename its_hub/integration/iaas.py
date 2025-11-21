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
import sys
from pathlib import Path

# Set up logging to both console and file
log_file = Path("iaas_tau.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler(log_file, mode='a')  # File output
    ]
)

# Also configure reward_hub loggers to use the same handlers
reward_hub_logger = logging.getLogger('its_hub.integration.reward_hub')
reward_hub_logger.setLevel(logging.INFO)

# Configure litellm loggers (they're set to WARNING in lms.py, but ensure they also log to file)
litellm_logger = logging.getLogger('litellm')
litellm_logger.addHandler(logging.FileHandler(log_file, mode='a'))

logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="its_hub Inference-as-a-Service",
    description="OpenAI-compatible API for inference-time scaling algorithms",
    version="0.1.0-alpha",
)

# Global state - TODO: Replace with proper dependency injection in production
LM_DICT: dict[str, OpenAICompatibleLanguageModel | LiteLLMLanguageModel] = {}
# Store all configured algorithms: {"self-consistency": instance, "best-of-n": instance, ...}
ALGORITHM_REGISTRY: dict[str, Any] = {}
# Store algorithm configurations: {"self-consistency": {"models": [...], ...}, ...}
ALGORITHM_CONFIGS: dict[str, dict] = {}
ROUTER: Any | None = None  # LLM Router for dynamic algorithm selection
VERIFIER: Any | None = None  # LLM Verifier for verification

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
        None,
        description="Reward model name or 'llm-judge' to use LLM-as-a-judge (not required for self-consistency)",
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
    exclude_tool_args: list[str] | None = Field(
        None,
        description="Tool argument names to exclude from voting (e.g., ['timestamp', 'id'])",
    )

    # LLM Judge settings (only used when rm_name='llm-judge')
    judge_model: str | None = Field(
        None,
        description="LiteLLM model name for judge (required when rm_name='llm-judge')",
    )
    judge_base_url: str | None = Field(
        None,
        description="Base URL for judge endpoint (required when rm_name='llm-judge')",
    )
    judge_criterion: str | None = Field(
        "overall_quality",
        description="Built-in criterion ('overall_quality', 'multi_step_tool_judge') OR custom evaluation description/prompt",
    )
    judge_mode: str | None = Field(
        "groupwise",
        description="'pointwise' (score each individually) or 'groupwise' (rank and select top-N)",
    )
    judge_top_n: int | None = Field(
        1, description="For groupwise: number of top responses to select"
    )
    judge_api_key: str | None = Field(None, description="API key for judge model")
    judge_temperature: float | None = Field(
        0.0, description="Judge temperature (0.0 for deterministic)"
    )
    judge_max_tokens: int | None = Field(
        4096, description="Maximum tokens for judge response"
    )
    enable_judge_logging: bool | None = Field(
        True, description="Log judge scores and reasoning"
    )

    # Router settings (optional - for dynamic algorithm selection)
    enable_router: bool | None = Field(
        False, description="Enable LLM router for dynamic algorithm selection"
    )
    router_model: str | None = Field(
        "gpt-4o-mini",
        description="LiteLLM model name for router (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')",
    )
    router_api_key: str | None = Field(
        None, description="API key for router model (uses judge_api_key if not provided)"
    )
    router_base_url: str | None = Field(
        None,
        description="Base URL for router endpoint (uses judge_base_url if not provided, 'auto' for default)",
    )
    router_max_budget: int | None = Field(
        32, description="Maximum budget the router can allocate"
    )
    router_system_prompt: str | None = Field(
        None, description="Custom system prompt for router (uses default if not provided)"
    )

    # Verifier settings (optional - for verification)
    enable_verifier: bool | None = Field(
        False, description="Enable LLM Verifier for verification"
    )
    verifier_model: str | None = Field(
        "gpt-4o-mini", description="LiteLLM model name for verifier (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')"
    )
    verifier_api_key: str | None = Field(
        None, description="API key for verifier model (uses judge_api_key if not provided)"
    )
    verifier_base_url: str | None = Field(
        None, description="Base URL for verifier endpoint (uses judge_base_url if not provided, 'auto' for default)"
    )
    regenerator_model: str | None = Field(
        None, description="LiteLLM model name for regenerator (defaults to verifier_model)"
    )
    regenerator_max_tokens: int | None = Field(
        2048, description="Maximum tokens for regenerator response"
    )
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

        if alg == "best-of-n" and not v:
            raise ValueError(
                "rm_name is required for best-of-n (use model name or 'llm-judge')"
            )
        elif alg == "particle-filtering" and not v:
            raise ValueError("rm_name is required for particle-filtering")
        return v

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, v, info):
        """Validate judge model is provided when using LLM judge."""
        if info.data.get("rm_name") == "llm-judge" and not v:
            raise ValueError("judge_model is required when rm_name='llm-judge'")
        return v

    @field_validator("judge_base_url")
    @classmethod
    def validate_judge_base_url(cls, v, info):
        """Validate judge base URL - requires 'auto' or a valid URL when using LLM judge."""
        if info.data.get("rm_name") == "llm-judge":
            if not v:
                raise ValueError("judge_base_url is required when rm_name='llm-judge' (use 'auto' for default endpoint)")
            # Accept "auto" or any other string (assumed to be a valid URL)
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
    # Import router if enabled
    if request.enable_router:
        try:
            from its_hub.integration.llm_router import LLMRouter
        except ImportError as e:
            logger.error(f"Failed to import LLM Router: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM Router integration not available",
            ) from e
    
    if request.enable_verifier:
        try:
            from its_hub.integration.verifier import LLMVerifier
        except ImportError as e:
            logger.error(f"Failed to import LLM Verifier: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM Verifier integration not available",
            ) from e
    # Only import reward_hub if needed (not required for self-consistency)
    if request.alg in {"particle-filtering", "best-of-n"}:
        
        try:
            from its_hub.integration.reward_hub import (
                AggregationMethod,
            )
        except ImportError as e:
            logger.error(f"Failed to import reward_hub: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Reward hub integration not available",
            ) from e

    if request.alg == "best-of-n" and request.rm_name != "llm-judge" or request.alg == "particle-filtering":
        try:
            from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
        except ImportError as e:
            logger.error(f"vLLM is required; install with `pip install its-hub[vllm]`: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="vLLM is required; install with `pip install its-hub[vllm]`") from e

    global LM_DICT, ALGORITHM_REGISTRY, ALGORITHM_CONFIGS, ROUTER, VERIFIER

    logger.info(f"Configuring service with model={request.model}, alg={request.alg}, router_enabled={request.enable_router}, verifier_enabled={request.enable_verifier}")

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

        # Configure scaling algorithm and register it
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
            algorithm = ParticleFiltering(sg, prm)
            ALGORITHM_REGISTRY["particle-filtering"] = algorithm
            ALGORITHM_CONFIGS["particle-filtering"] = {
                "models": [request.model],
                "judge_model": request.rm_name,
            }

        elif request.alg == "best-of-n":
            if request.rm_name == "llm-judge":
                # Use LLM Judge adapter from its_hub integration
                try:
                    from its_hub.integration.reward_hub import LLMJudgeRewardModel
                    from reward_hub.llm_judge.prompts import (
                        Criterion,
                        CriterionRegistry,
                    )
                except ImportError as e:
                    logger.error(f"Failed to import LLM Judge: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="LLM Judge integration not available",
                    ) from e

                # Check if it's a built-in criterion or custom
                built_in_criteria = {"overall_quality", "multi_step_tool_judge"}

                if request.judge_criterion in built_in_criteria:
                    criterion_to_use = request.judge_criterion
                    logger.info(f"Using built-in criterion: {request.judge_criterion}")
                else:
                    # Custom criterion - register it with auto-generated name
                    criterion_name = (
                        f"custom_{hash(request.judge_criterion) & 0xFFFFFFFF:08x}"
                    )
                    logger.info(f"Registering custom criterion as: {criterion_name}")
                    custom_criterion = Criterion(
                        name=criterion_name,
                        content=request.judge_criterion,
                        description="Custom evaluation criterion",
                    )
                    CriterionRegistry.register(custom_criterion)
                    criterion_to_use = criterion_name

                logger.info(
                    f"Configuring LLM Judge: model={request.judge_model}, "
                    f"criterion={criterion_to_use}, mode={request.judge_mode}"
                )

                # Create LLM Judge using the adapter (handles ChatMessages conversion)
                # Convert "auto" to None for LiteLLM auto-detection of default endpoint
                judge_base_url = None if request.judge_base_url == "auto" else request.judge_base_url

                reward_model = LLMJudgeRewardModel(
                    model=request.judge_model,
                    criterion=criterion_to_use,
                    judge_type=request.judge_mode or "groupwise",
                    api_key=request.judge_api_key,
                    base_url=judge_base_url,
                    temperature=request.judge_temperature,
                    max_tokens=request.judge_max_tokens,
                    enable_judge_logging=request.enable_judge_logging
                    if request.enable_judge_logging is not None
                    else True,
                    top_n=request.judge_top_n or 1,
                )
            else:
                # Use traditional process reward model
                reward_model = LocalVllmProcessRewardModel(
                    model_name=request.rm_name,
                    device=request.rm_device,
                    aggregation_method=AggregationMethod("model"),
                )

            algorithm = BestOfN(reward_model)
            ALGORITHM_REGISTRY["best-of-n"] = algorithm
            ALGORITHM_CONFIGS["best-of-n"] = {
                "models": [request.model],
                "judge_model": request.rm_name or "llm-judge",
            }

        elif request.alg == "self-consistency":
            # Create projection function from regex patterns
            if request.regex_patterns:
                projection_func = create_regex_projection_function(
                    request.regex_patterns
                )
            else:
                projection_func = None
            algorithm = SelfConsistency(
                projection_func,
                tool_vote=request.tool_vote,
                exclude_args=request.exclude_tool_args,
            )
            ALGORITHM_REGISTRY["self-consistency"] = algorithm
            ALGORITHM_CONFIGS["self-consistency"] = {
                "models": [request.model],
            }

        # Configure router if enabled
        if request.enable_router:
            # Use judge credentials as fallback for router
            router_api_key = request.router_api_key or request.judge_api_key
            router_base_url = request.router_base_url or request.judge_base_url

            # Convert "auto" to None for LiteLLM
            if router_base_url == "auto":
                router_base_url = None

            logger.info(f"Initializing LLM Router with model={request.router_model}")

            # Pass custom system prompt if provided
            router_kwargs = {
                "router_model": request.router_model,
                "router_api_key": router_api_key,
                "router_base_url": router_base_url,
                "enable_logging": True,
            }

            if request.router_system_prompt:
                router_kwargs["system_prompt"] = request.router_system_prompt
                logger.info("Using custom router system prompt")

            ROUTER = LLMRouter(**router_kwargs)
            logger.info("LLM Router initialized successfully")
        else:
            ROUTER = None

        if request.enable_verifier:
            # Use judge credentials as fallback for verifier
            verifier_api_key = request.verifier_api_key or request.judge_api_key
            verifier_base_url = request.verifier_base_url or request.judge_base_url

            # Convert "auto" to None for LiteLLM
            if verifier_base_url == "auto":
                verifier_base_url = None

            verifier_kwargs = {
                "verifier_model": request.verifier_model,
                "verifier_api_key": verifier_api_key,
                "verifier_base_url": verifier_base_url,
                "regenerator_model": request.regenerator_model,
                "regenerator_max_tokens": request.regenerator_max_tokens or 2048,
                "enable_logging": True,
            }
            VERIFIER = LLMVerifier(**verifier_kwargs)
            logger.info(f"LLM Verifier initialized successfully (verifier={request.verifier_model}, regenerator={request.regenerator_model or request.verifier_model})")
        else:
            VERIFIER = None
        
        logger.info(f"Successfully configured {request.alg} algorithm")
        return {
            "status": "success",
            "message": f"Initialized {request.model} with {request.alg} algorithm"
            + (" (router enabled)" if request.enable_router else ""),
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
    verifier_budget: int = Field(
        2, ge=1, le=100, description="Computational budget for verifier"
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
    use_router: bool = Field(
        False, description="Use LLM router to dynamically select algorithm and budget (overrides budget parameter)"
    )
    router_max_budget: int | None = Field(
        None, description="Maximum budget the router can allocate (uses configured value if not specified)"
    )
    seed: int | None = Field(
        None, description="Random seed for reproducible outputs"
    )
    use_verifier: bool = Field(
        False, description="Use LLM verifier for iterative verification and regeneration"
    )
    verifier_policy: str | None = Field(
        None, description="Custom policy for verification (uses default if not provided)"
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
    """Token usage information - kept for API compatibility but not populated."""

    prompt_tokens: int = Field(default=0, description="Tokens in prompt")
    completion_tokens: int = Field(default=0, description="Generated tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")



def _extract_algorithm_metadata(algorithm_result: Any) -> dict[str, Any] | None:
    """Extract metadata from algorithm results for API response."""
    from its_hub.algorithms.self_consistency import SelfConsistencyResult
    from its_hub.algorithms.bon import BestOfNResult

    if isinstance(algorithm_result, SelfConsistencyResult):
        return {
            "algorithm": "self-consistency",
            "all_responses": algorithm_result.responses,  # Now contains full message dicts with tool calls
            "response_counts": dict(algorithm_result.response_counts),
            "selected_index": algorithm_result.selected_index,
        }

    elif isinstance(algorithm_result, BestOfNResult):
        return {
            "algorithm": "best-of-n",
            "responses": algorithm_result.responses,
            "scores": algorithm_result.scores,
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


def _sanitize_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Sanitize incoming messages to handle common client-side formatting issues.

    Fixes:
    1. tool_calls: None -> removes the field entirely
    2. Extra 'name' field in tool_calls -> removes it
    3. Ensures tool_calls follow OpenAI format
    """
    sanitized = []

    for msg in messages:
        # Convert to dict for easier manipulation
        if hasattr(msg, 'to_dict'):
            msg_dict = msg.to_dict()
        elif hasattr(msg, '__dict__'):
            msg_dict = dict(msg.__dict__)
        else:
            # Already a dict
            msg_dict = dict(msg)

        # Fix tool_calls field
        if "tool_calls" in msg_dict:
            if msg_dict["tool_calls"] is None:
                # Remove None tool_calls
                del msg_dict["tool_calls"]
            elif isinstance(msg_dict["tool_calls"], list) and msg_dict["tool_calls"]:
                # Clean up tool call structure
                cleaned_tool_calls = []
                for tc in msg_dict["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue

                    # Create properly formatted tool call
                    cleaned_tc = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": tc.get("function", {})
                    }

                    # Remove any extra 'name' field at the top level
                    # (it should only be in the 'function' object)

                    cleaned_tool_calls.append(cleaned_tc)

                msg_dict["tool_calls"] = cleaned_tool_calls if cleaned_tool_calls else None

                # Remove tool_calls if it's now None
                if msg_dict["tool_calls"] is None:
                    del msg_dict["tool_calls"]

        # Recreate ChatMessage from sanitized dict
        # Return as-is if already a proper ChatMessage, otherwise create new one
        if isinstance(msg, ChatMessage) and "tool_calls" not in msg_dict:
            sanitized.append(msg)
        else:
            sanitized.append(ChatMessage(**msg_dict))

    return sanitized


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

    if not ALGORITHM_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not configured. Please call /configure first.",
        )

    try:
        # Configure language model for this request
        if request.temperature is not None:
            lm.temperature = request.temperature

        # Sanitize incoming messages to handle client-side formatting issues
        sanitized_messages = _sanitize_messages(list(request.messages))

        # Create ChatMessages from the full conversation history
        chat_messages = ChatMessages(sanitized_messages)

        # Initialize routing_info
        routing_info = None

        # Determine which algorithm to use
        if request.use_router and ROUTER is not None:
            # Use router to select algorithm dynamically
            logger.info("Using LLM Router for algorithm selection")
            routing_decision = await ROUTER.route(
                chat_messages,
                available_algorithms=ALGORITHM_CONFIGS,
                max_budget=request.router_max_budget or 32,
            )

            selected_algorithm = routing_decision.algorithm
            actual_budget = routing_decision.budget
            selected_model_name = routing_decision.model

            routing_info = {
                "algorithm": routing_decision.algorithm,
                "budget": routing_decision.budget,
                "model": routing_decision.model,
                "reasoning": routing_decision.reasoning,
            }

            logger.info(
                f"Router selected: algorithm={selected_algorithm}, "
                f"budget={actual_budget}, model={selected_model_name}, "
            )

            # Get the algorithm instance
            if selected_algorithm not in ALGORITHM_REGISTRY:
                logger.error(f"Router selected unavailable algorithm: {selected_algorithm}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Selected algorithm '{selected_algorithm}' not configured",
                )

            selected_alg = ALGORITHM_REGISTRY[selected_algorithm]

            # Get the language model for the selected model
            if selected_model_name in LM_DICT:
                selected_lm = LM_DICT[selected_model_name]
            else:
                logger.warning(
                    f"Router selected model '{selected_model_name}' not found. "
                    f"Using '{request.model}'"
                )
                selected_lm = lm

        else:
            # Use the first configured algorithm (backward compatibility)
            selected_alg = list(ALGORITHM_REGISTRY.values())[0]
            actual_budget = request.budget
            selected_lm = lm

        # Generate response using selected algorithm
        algorithm_result = await selected_alg.ainfer(
            selected_lm,
            chat_messages,
            actual_budget,
            return_response_only=request.return_response_only,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

        # Apply verification and regeneration if enabled
        verification_history = None
        if request.use_verifier and VERIFIER is not None:
            # Extract the response message to verify
            if not request.return_response_only and hasattr(algorithm_result, "the_one"):
                response_to_verify = algorithm_result.the_one
            else:
                response_to_verify = algorithm_result

            # Defensive check: ensure response_to_verify is not None
            if response_to_verify is None:
                logger.error("Algorithm result is None - cannot verify")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Algorithm returned None result",
                )

            print(f"\n[IAAS] Response BEFORE verification/refinement:")
            # Handle case where content might be None
            content_preview = response_to_verify.get('content') or ''
            if isinstance(content_preview, str):
                print(f"[IAAS] {content_preview[:300]}...\n")
            else:
                print(f"[IAAS] Content type: {type(content_preview)}\n")

            # Create messages with the algorithm's response appended
            messages_to_verify = chat_messages.to_chat_messages() + [ChatMessage(**response_to_verify)]

            # Get verified response with iterative verification and regeneration
            verified_response, verification_history = await VERIFIER.get_verified_response(
                messages=messages_to_verify,
                policy=request.verifier_policy,
                verification_budget=request.verifier_budget,
                tools=request.tools,
                num_turns_to_keep=8
            )

            print(f"\n[IAAS] Response AFTER verification/refinement:")
            print(f"[IAAS] {verified_response.get('content', '')[:300]}...\n")

            # Replace the algorithm result with the verified response
            if not request.return_response_only and hasattr(algorithm_result, "responses"):
                # For result objects like BestOfNResult, replace the response at selected_index
                algorithm_result.responses[algorithm_result.selected_index] = verified_response
            else:
                # For direct response (return_response_only=True), replace entirely
                algorithm_result = verified_response
        

        # Extract response content and metadata
        if not request.return_response_only and hasattr(algorithm_result, "the_one"):
            # Got a full result object
            response_message = algorithm_result.the_one
            metadata = _extract_algorithm_metadata(algorithm_result)
        else:
            # Got just a message dict response
            response_message = algorithm_result
            metadata = None

        # Defensive check: ensure response_message is not None
        if response_message is None:
            logger.error("Response message is None after algorithm execution")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Algorithm returned None response message",
            )

        # Add routing info to metadata if available
        if routing_info:
            if metadata is None:
                metadata = {}
            metadata["routing"] = routing_info

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(),
            metadata=metadata,
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
