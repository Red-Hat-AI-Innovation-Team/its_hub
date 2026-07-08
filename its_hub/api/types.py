"""Type definitions for its_hub."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Literal


@dataclass
class ChatMessage:
    """A chat message with role and content.

    Content can be:
    - str: Simple text content
    - list[dict]: Multi-modal content (text, images, etc.)
    - None: No content (e.g., when using tool_calls)
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None
    tool_calls: list[dict] | None = None  # Store as plain dicts
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ChatMessage:
        """Create ChatMessage from dictionary, ignoring unknown fields."""
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert ChatMessage to dictionary, excluding None values."""
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result

    def extract_text_content(self) -> str:
        """Extract text content from message, handling both string and list formats.
        For list content (multi-modal), extracts all text parts and warns about non-text content.
        Returns empty string if no text content is found.
        """
        if self.content is None:
            return ""

        if isinstance(self.content, str):
            return self.content

        # Must be list[dict] at this point
        text_parts = []
        has_image = False

        for item in self.content:
            content_type = item.get("type", "")

            if content_type == "text":
                text_parts.append(item.get("text", ""))
            elif content_type == "image_url":
                has_image = True
            elif content_type:
                raise ValueError(
                    f"Unsupported content type '{content_type}' in messages content dict."
                )

        if has_image:
            logging.warning(
                "Image content detected in message but is not supported. "
                "Image content will be ignored. Only text content is processed."
            )

        return " ".join(text_parts)


class ChatMessages:
    """Unified wrapper for handling both string prompts and conversation history."""

    def __init__(self, str_or_messages: str | list[ChatMessage]):
        self._str_or_messages = str_or_messages
        self._is_string = isinstance(str_or_messages, str)

    @classmethod
    def from_prompt_or_messages(
        cls, prompt_or_messages: str | list[ChatMessage] | ChatMessages
    ) -> ChatMessages:
        """Create ChatMessages from various input formats."""
        if isinstance(prompt_or_messages, ChatMessages):
            return prompt_or_messages
        return cls(prompt_or_messages)

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert to list of ChatMessage objects."""
        if self._is_string:
            return [ChatMessage(role="user", content=self._str_or_messages)]
        return self._str_or_messages

    def to_batch(self, size: int) -> list[list[ChatMessage]]:
        """Create a batch of identical chat message lists for parallel generation."""
        chat_messages = self.to_chat_messages()
        return [list(chat_messages) for _ in range(size)]

    def to_prompt(self) -> str:
        """Convert to prompt string representation.

        This method is used by experimental algorithms (BeamSearch, ParticleGibbs, PlanningWrapper)
        for backward compatibility. It converts chat messages to a simple string format.
        """
        if self._is_string:
            return self._str_or_messages

        # Convert list of ChatMessage to string
        parts = []
        for msg in self._str_or_messages:
            role = msg.role
            content = msg.content
            if content is None:
                content = ""
            elif isinstance(content, list):
                # Extract text from multi-modal content
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @property
    def is_string(self) -> bool:
        """Check if the original input was a string."""
        return self._is_string


@dataclass
class GenerationUsage:
    """Accumulated token usage from LLM API calls.

    Pass an instance to agenerate(usage_accumulator=...) to collect
    prompt/completion token counts across all parallel API calls.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    num_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.num_calls += 1

    def merge(self, other: GenerationUsage) -> None:
        """Merge another GenerationUsage into this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.num_calls += other.num_calls

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ITSRequestConfig:
    """Per-request configuration for ITS execution.

    Can be constructed incrementally — headers provide budget/endpoint/api_key,
    then model is set from the request body.
    """

    budget: int
    api_endpoint: str
    model: str | None = None
    api_key: str | None = None

    def __post_init__(self):
        if self.budget < 1 or self.budget > 1000:
            raise ValueError("budget must be between 1 and 1000")

    def __repr__(self) -> str:
        return (
            f"ITSRequestConfig(budget={self.budget}, "
            f"api_endpoint='{self.api_endpoint}', "
            f"model={self.model!r}, api_key={'***' if self.api_key else None})"
        )
