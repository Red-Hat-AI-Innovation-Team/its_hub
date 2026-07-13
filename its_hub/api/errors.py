"""
Error handling utilities for OpenAI-compatible API calls.

This module provides specific exception classes and utilities to handle
different types of API failures with appropriate retry logic and informative
error messages.
"""

import json
from typing import Any

try:
    from aiohttp import ClientError as AiohttpClientError
except ImportError:
    AiohttpClientError = None


class APIError(Exception):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_details = error_details or {}
        super().__init__(message)


class RateLimitError(APIError):
    """Rate limit exceeded - retryable."""

    pass


class ContextLengthError(APIError):
    """Context length exceeded - not retryable."""

    pass


class MaxCompletionTokensError(APIError):
    """Generation truncated by max_completion_tokens limit - not retryable."""

    pass


class AuthenticationError(APIError):
    """Authentication failed - not retryable."""

    pass


class APIConnectionError(APIError):
    """Network/connection issues - retryable."""

    pass


class BadRequestError(APIError):
    """Bad request (invalid parameters) - not retryable."""

    pass


class InternalServerError(APIError):
    """Server error - retryable."""

    pass


# Retryable error types — includes both API-level and transport-level exceptions.
# ConnectionError covers socket resets and refused connections.
# TimeoutError covers asyncio.TimeoutError and socket timeouts.
# OSError covers low-level network failures (DNS, broken pipe, etc.).
_RETRYABLE_BASE = (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
    ConnectionError,
    TimeoutError,
    OSError,
)
RETRYABLE_ERRORS = (
    (*_RETRYABLE_BASE, AiohttpClientError)
    if AiohttpClientError is not None
    else _RETRYABLE_BASE
)


def parse_api_error(status_code: int, error_text: str) -> APIError:
    """
    Parse API error response and return appropriate exception.

    Args:
        status_code: HTTP status code
        error_text: Raw error response text

    Returns:
        Appropriate APIError subclass instance
    """
    error_details = {}
    error_message = error_text

    # Try to parse JSON error response
    try:
        error_json = json.loads(error_text)
        if "error" in error_json:
            error_info = error_json["error"]
            if isinstance(error_info, dict):
                error_message = error_info.get("message", error_text)
                error_details = error_info
            else:
                error_message = str(error_info)
    except (json.JSONDecodeError, KeyError):
        # Use raw text if JSON parsing fails
        pass

    # Classify error based on status code and message content
    error_message_lower = error_message.lower()

    if status_code == 429 or "rate limit" in error_message_lower:
        return RateLimitError(
            f"Rate limit exceeded: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    if status_code == 400 and "output limit" in error_message_lower:
        return MaxCompletionTokensError(
            f"Max completion tokens reached: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    if status_code == 400 and any(
        phrase in error_message_lower
        for phrase in [
            "context length",
            "maximum context",
            "too long",
            "token limit",
            "context_length_exceeded",
        ]
    ):
        return ContextLengthError(
            f"Context length exceeded: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    if (
        status_code in [401, 403]
        or "authentication" in error_message_lower
        or "unauthorized" in error_message_lower
    ):
        return AuthenticationError(
            f"Authentication failed: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    if status_code == 400:
        return BadRequestError(
            f"Bad request: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    if status_code >= 500:
        return InternalServerError(
            f"Server error: {error_message}",
            status_code=status_code,
            error_details=error_details,
        )

    # Default to APIConnectionError for other cases (network issues, etc.)
    return APIConnectionError(
        f"API request failed: {error_message}",
        status_code=status_code,
        error_details=error_details,
    )


def enhanced_on_backoff(details):
    """
    Enhanced backoff callback that shows specific error information.

    Args:
        details: Backoff details dictionary containing exception info
    """
    exception = details.get("exception")
    if isinstance(exception, APIError):
        error_type = type(exception).__name__
        print(
            f"Retrying after {details['wait']:.1f}s (attempt {details['tries']}) - "
            f"{error_type}: {exception.message}"
        )
    else:
        # Fallback for non-APIError exceptions
        print(
            f"Retrying after {details['wait']:.1f}s (attempt {details['tries']}) - "
            f"Error: {exception!s}"
        )


def should_retry(exception: Exception) -> bool:
    """
    Determine if an exception should be retried.

    Args:
        exception: The exception to check

    Returns:
        True if the exception should be retried, False otherwise
    """
    return isinstance(exception, RETRYABLE_ERRORS)


def format_non_retryable_error(exception: APIError) -> str:
    """
    Format a helpful message for non-retryable errors.

    Args:
        exception: The non-retryable APIError

    Returns:
        Formatted error message with suggestions
    """
    if isinstance(exception, ContextLengthError):
        return (
            f"❌ {exception.message}\n"
            f"💡 Suggestion: Reduce input length or use a model with a larger context window"
        )

    if isinstance(exception, MaxCompletionTokensError):
        return (
            f"❌ {exception.message}\n"
            f"💡 Suggestion: Increase max_completion_tokens to allow longer responses"
        )

    if isinstance(exception, AuthenticationError):
        return (
            f"❌ {exception.message}\n"
            f"💡 Suggestion: Check your API key and endpoint configuration"
        )

    if isinstance(exception, BadRequestError):
        return (
            f"❌ {exception.message}\n"
            f"💡 Suggestion: Check your request parameters (temperature, max_completion_tokens, etc.)"
        )

    return f"❌ {exception.message}"
