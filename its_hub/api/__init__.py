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
    RateLimitError,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)
from .lm import AbstractLanguageModel
from .reward_models.orm import AbstractOutcomeRewardModel
from .reward_models.prm import AbstractProcessRewardModel
from .types import ChatMessage, ChatMessages

__all__ = [  # noqa: RUF022
    # Algorithm abstractions
    "AbstractScalingAlgorithm",
    "AbstractScalingResult",
    # Language model abstractions
    "AbstractLanguageModel",
    # Reward model abstractions
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
    # Common types
    "ChatMessage",
    "ChatMessages",
    # Error types
    "APIError",
    "RateLimitError",
    "ContextLengthError",
    "AuthenticationError",
    "APIConnectionError",
    "BadRequestError",
    "InternalServerError",
]
