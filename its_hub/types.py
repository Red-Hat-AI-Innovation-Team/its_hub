"""Type definitions for its_hub."""

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class ChatMessage:
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """OpenAI tool call structure."""
    
    id: str = Field(..., description="Unique tool call ID")
    type: str = Field(..., description="Type of tool call (e.g., 'function')")
    function: dict[str, Any] = Field(..., description="Function call details")


class ChatCompletionMessage(BaseModel):
    """OpenAI chat completion message with optional tool calls."""
    
    role: Literal["assistant"] = Field(..., description="Message role")
    content: str | None = Field(None, description="Message content")
    tool_calls: list[ToolCall] | None = Field(None, description="Tool calls made by assistant")
