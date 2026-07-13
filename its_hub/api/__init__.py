"""Abstract base classes for its_hub components."""

from .algorithm import (
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from .errors import (
    RETRYABLE_ERRORS,
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextLengthError,
    InternalServerError,
    MaxCompletionTokensError,
    RateLimitError,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)
from .lm import AbstractLanguageModel
from .orchestrator import AbstractOrchestrator
from .reward_models.orm import AbstractOutcomeRewardModel
from .reward_models.prm import AbstractProcessRewardModel
from .types import ChatMessage, ChatMessages, GenerationUsage

__all__ = [  # noqa: RUF022
    # Algorithm abstractions
    "AbstractScalingAlgorithm",
    "AbstractScalingResult",
    # Language model abstractions
    "AbstractLanguageModel",
    # Reward model abstractions
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
    # Orchestrator abstraction
    "AbstractOrchestrator",
    # Common types
    "ChatMessage",
    "ChatMessages",
    "GenerationUsage",
    # Error types
    "APIError",
    "RateLimitError",
    "ContextLengthError",
    "MaxCompletionTokensError",
    "AuthenticationError",
    "APIConnectionError",
    "BadRequestError",
    "InternalServerError",
]
