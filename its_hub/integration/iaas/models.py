"""Pydantic request/response models for the IaaS API."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from its_hub.api.types import SUPPORTED_ALGORITHMS, ChatMessage


class ConfigRequest(BaseModel):
    """Configuration request for setting up the IaaS service."""

    endpoint: str = Field(..., description="Language model endpoint URL")
    api_key: str | None = Field(None, description="API key for the language model")
    model: str = Field(..., description="Model name identifier")
    alg: str = Field(..., description="Scaling algorithm to use")
    regex_patterns: list[str] | None = Field(
        None, description="Regex patterns for self-consistency projection function"
    )
    budget: int | None = Field(
        None,
        ge=1,
        le=1000,
        description="Default budget for requests that don't specify one",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature",
    )
    tool_vote: str | None = Field(
        None,
        description="Tool voting strategy: 'tool_name', 'tool_args', 'tool_hierarchical'",
    )
    exclude_tool_args: list[str] | None = Field(
        None,
        description="Tool argument names to exclude from voting",
    )
    threshold: float | None = Field(
        None,
        gt=0.5,
        le=1.0,
        description="Supermajority vote-share threshold for adaptive-self-consistency early stopping (default 0.75)",
    )
    confidence_threshold: float | None = Field(
        None,
        gt=0.5,
        le=1.0,
        description="Beta posterior confidence threshold for beta-self-consistency early stopping (default 0.95)",
    )

    @field_validator("alg")
    @classmethod
    def validate_algorithm(cls, v):
        if v not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm '{v}' not supported. Choose from: {SUPPORTED_ALGORITHMS}"
            )
        return v


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
    stream: bool | None = Field(False, description="Stream response")
    tools: list[dict[str, Any]] | None = Field(
        None, description="Available tools for the model to call"
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool choice strategy"
    )
    return_response_only: bool = Field(
        True, description="Return only final response or include algorithm metadata"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message is required")
        return v


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    metadata: dict[str, Any] | None = None
