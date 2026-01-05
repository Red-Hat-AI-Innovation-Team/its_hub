"""Type definitions for its_hub."""

from __future__ import annotations

from dataclasses import dataclass
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
        return [chat_messages for _ in range(size)]

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
