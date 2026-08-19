"""Type definitions for its_hub."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields, replace
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
    content: str | list[dict] | None = None
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


SELF_CONSISTENCY_ALGORITHMS = frozenset(
    {"self-consistency", "adaptive-self-consistency", "beta-self-consistency"}
)

# All currently supported algorithms are self-consistency variants.
SUPPORTED_ALGORITHMS = SELF_CONSISTENCY_ALGORITHMS

# Valid values for ITSRequestConfig.tool_vote / SelfConsistency.tool_vote.
VALID_TOOL_VOTE_OPTIONS = frozenset(
    {"tool_name", "tool_args", "tool_hierarchical", "tool_flat_all"}
)


def _validate_optional_fields(
    budget: int | None,
    alg: str | None,
    regex_patterns: list[str] | None,
    tool_vote: str | None,
    threshold: float | None,
    confidence_threshold: float | None,
) -> None:
    """Validate the optional scaling fields. Shared by ITSRequestConfig and
    ITSRequestConfigUpdate so the two cannot drift. Each check is guarded so
    that ``None`` ("not supplied") is always accepted."""
    if budget is not None and not (1 <= budget <= 1000):
        raise ValueError("budget must be between 1 and 1000")
    if alg is not None and alg not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Algorithm {alg!r} not supported. "
            f"Choose from: {SUPPORTED_ALGORITHMS}"
        )
    if regex_patterns is not None:
        for p in regex_patterns:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern {p!r}: {e}") from e
    if threshold is not None and not (0.5 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0.5, 1.0], got: {threshold}")
    if confidence_threshold is not None and not (0.5 < confidence_threshold <= 1.0):
        raise ValueError(
            f"confidence_threshold must be in (0.5, 1.0], "
            f"got: {confidence_threshold}"
        )
    if tool_vote is not None and tool_vote not in VALID_TOOL_VOTE_OPTIONS:
        raise ValueError(
            f"tool_vote must be one of {VALID_TOOL_VOTE_OPTIONS}, got: {tool_vote}"
        )


def _config_repr(obj: ITSRequestConfig | ITSRequestConfigUpdate) -> str:
    return (
        type(obj).__name__
        + "("
        + ", ".join(
            f"{f.name}={'***' if f.name == 'api_key' else getattr(obj, f.name)!r}"
            for f in fields(obj)
        )
        + ")"
    )


@dataclass(frozen=True)
class ITSRequestConfig:
    """A fully-resolved configuration snapshot, ready to drive one request.

    ``api_endpoint`` and ``model`` are mandatory — constructing without them
    raises. The remaining fields carry system defaults (``budget``, ``alg``,
    ``api_key``) or are genuinely optional (``None`` is a meaningful "not set",
    e.g. ``temperature=None`` means "don't send it upstream").
    """

    # LM target
    api_endpoint: str
    model: str
    api_key: str | None = None
    temperature: float | None = None
    # Scaling parameters
    budget: int = 4
    alg: str = "self-consistency"
    regex_patterns: list[str] | None = None
    tool_vote: str | None = None
    exclude_tool_args: list[str] | None = None
    threshold: float | None = None
    confidence_threshold: float | None = None

    def __post_init__(self):
        if not self.api_endpoint:
            raise ValueError("api_endpoint must be specified")
        if not self.model:
            raise ValueError("model must be specified")
        _validate_optional_fields(
            self.budget,
            self.alg,
            self.regex_patterns,
            self.tool_vote,
            self.threshold,
            self.confidence_threshold,
        )

    def __repr__(self) -> str:
        return _config_repr(self)


@dataclass
class ITSRequestConfigUpdate:
    """A partial configuration overlay.

    Every field is ``Optional``: ``None`` means "not supplied — keep the value
    below in the merge hierarchy". Because of this, ``tool_vote`` cannot be
    cleared back to its default once set (pass the new value, or
    ``"tool_hierarchical"`` explicitly); ``regex_patterns`` and
    ``exclude_tool_args`` can be cleared by passing ``[]``.

    Use :meth:`merge` to layer one update over another, and :meth:`resolve` to
    materialize a complete :class:`ITSRequestConfig` (the only point at which
    mandatory-field presence is checked).
    """

    # LM target
    api_endpoint: str | None = None
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    # Scaling parameters
    budget: int | None = None
    alg: str | None = None
    regex_patterns: list[str] | None = None
    tool_vote: str | None = None
    exclude_tool_args: list[str] | None = None
    threshold: float | None = None
    confidence_threshold: float | None = None

    def __post_init__(self):
        _validate_optional_fields(
            self.budget,
            self.alg,
            self.regex_patterns,
            self.tool_vote,
            self.threshold,
            self.confidence_threshold,
        )

    def __repr__(self) -> str:
        return _config_repr(self)

    def merge(self, overlay: ITSRequestConfigUpdate) -> ITSRequestConfigUpdate:
        """Return a copy of ``self`` with non-``None`` fields of ``overlay``
        applied. ``replace()`` re-fires ``__post_init__``, so the result is
        format-validated."""
        return replace(
            self,
            **{
                f.name: getattr(overlay, f.name)
                for f in fields(overlay)
                if getattr(overlay, f.name) is not None
            },
        )

    def resolve(self) -> ITSRequestConfig:
        """Materialize a complete :class:`ITSRequestConfig`.

        Raises ``ValueError`` if ``api_endpoint`` or ``model`` are absent;
        this is the single point at which mandatory-field presence is
        checked. Optional fields are forwarded only when non-``None``;
        otherwise the resolved config's defaults (``budget=4``,
        ``alg="self-consistency"``, ``api_key=None``) apply.
        """
        if not self.api_endpoint:
            raise ValueError("api_endpoint must be specified")
        if not self.model:
            raise ValueError("model must be specified")
        optionals = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"api_endpoint", "model"}
            and getattr(self, f.name) is not None
        }
        return ITSRequestConfig(
            api_endpoint=self.api_endpoint,
            model=self.model,
            **optionals,
        )
