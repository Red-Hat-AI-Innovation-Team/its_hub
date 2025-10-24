"""Type definitions for its_hub."""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class Function:
    """Function definition for tool calls."""

    name: str
    description: str | None = None
    parameters: dict | None = None


@dataclass
class ToolCall:
    """A tool call made by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: Function | None = None


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
    tool_calls: list[dict] | None = None  # Store as plain dicts, not Pydantic objects
    tool_call_id: str | None = None

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
                raise ValueError(f"Unsupported content type '{content_type}' in messages content dict.")

        if has_image:
            logging.warning(
                "Image content detected in message but is not supported. "
                "Image content will be ignored. Only text content is processed."
            )

        return " ".join(text_parts)

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

    def to_prompt(self) -> str:
        # TODO: chatMessage to string conversion will be deprecated in the future.
        """Convert to prompt string representation.
        """
        if self._is_string:
            return self._str_or_messages

        lines = []
        for msg in self._str_or_messages:
            text_content = msg.extract_text_content()

            if msg.role == "tool":
                # Tool messages: include tool_call_id context
                lines.append(f"tool[{msg.tool_call_id}]: {text_content}")
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant with tool calls: show tool calls + content if any
                tool_call_strs = []
                for tc in msg.tool_calls:
                    if tc.function:
                        tool_call_strs.append(f"{tc.function.name}()")
                tool_calls_text = ", ".join(tool_call_strs)
                if text_content:
                    lines.append(f"assistant: {text_content} [calls: {tool_calls_text}]")
                else:
                    lines.append(f"assistant: [calls: {tool_calls_text}]")
            else:
                # Regular messages
                lines.append(f"{msg.role}: {text_content}")

        return "\n".join(lines)

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert to list of ChatMessage objects."""
        if self._is_string:
            return [ChatMessage(role="user", content=self._str_or_messages)]
        return self._str_or_messages

    def to_batch(self, size: int) -> list[list[ChatMessage]]:
        """Create a batch of identical chat message lists for parallel generation."""
        chat_messages = self.to_chat_messages()
        return [chat_messages for _ in range(size)]

    @property
    def is_string(self) -> bool:
        """Check if the original input was a string."""
        return self._is_string
