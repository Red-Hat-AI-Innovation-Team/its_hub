"""Type definitions for its_hub."""

from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class ChatMessage:
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant"]
    content: str
