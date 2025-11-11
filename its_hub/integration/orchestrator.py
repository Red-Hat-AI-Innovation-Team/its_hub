"""Stateless ITS Orchestrator for Envoy ext_proc integration.

This module provides a long-lived orchestrator with per-request configuration,
similar to how HTTP services reuse client instances across requests.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from its_hub.algorithms.self_consistency import SelfConsistency
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.types import ChatMessages, ChatMessage

logger = logging.getLogger(__name__)


class ITSRequestConfig(BaseModel):
    """Per-request configuration for ITS execution.

    Can be constructed incrementally - headers provide budget/endpoint/api_key,
    then model is set from the request body.
    """

    budget: int = Field(..., ge=1, le=1000, description="Computational budget for scaling")
    api_endpoint: str = Field(
        ..., description="LLM API endpoint (or Envoy cluster URL for loop-back)"
    )
    model: Optional[str] = Field(None, description="Model name identifier from request body")
    api_key: Optional[str] = Field(None, description="API key for the LLM endpoint")


class ITSOrchestrator:
    """Long-lived orchestrator for running ITS algorithms.

    This orchestrator is initialized once at service startup and reused across
    all requests. Per-request configuration is passed as arguments to run_chat_completion().

    Similar to how HTTP services use a single static HTTP client instance,
    this orchestrator maintains a cache of LM clients keyed by (endpoint, model)
    to reuse connections.

    Example:
        # At service startup
        orchestrator = ITSOrchestrator()

        # Per request
        config = ITSRequestConfig(
            budget=10,
            api_endpoint="http://envoy-cluster/v1/chat/completions",
            model="gpt-4",
            api_key="sk-...",
        )
        result = await orchestrator.run_chat_completion(config, messages)
    """

    def __init__(self):
        """Initialize the orchestrator with empty client cache."""
        self._lm_cache: dict[tuple[str, str], OpenAICompatibleLanguageModel] = {}
        self._algorithm = SelfConsistency(
            consistency_space_projection_func=None,
            tool_vote=None,
            exclude_args=None,
        )
        logger.info("ITSOrchestrator initialized (self-consistency, simple mode)")

    def _get_or_create_lm(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> OpenAICompatibleLanguageModel:
        """Get cached LM client or create new one.

        Clients are cached by (endpoint, model) to reuse HTTP connections.

        Args:
            endpoint: API endpoint URL
            model: Model name
            api_key: Optional API key

        Returns:
            Cached or new LM client instance
        """
        cache_key = (endpoint, model)
        log_prefix = f"[{request_id}] " if request_id else ""

        if cache_key not in self._lm_cache:
            logger.info(
                "%sCreating new LM client: endpoint=%s, model=%s",
                log_prefix,
                endpoint,
                model,
            )
            self._lm_cache[cache_key] = OpenAICompatibleLanguageModel(
                endpoint=endpoint,
                api_key=api_key,
                model_name=model,
                is_async=True,
            )
        else:
            logger.debug(
                "%sReusing cached LM client: endpoint=%s, model=%s",
                log_prefix,
                endpoint,
                model,
            )

        return self._lm_cache[cache_key]

    async def run_chat_completion(
        self,
        config: ITSRequestConfig,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str | dict[str, Any]] = None,
        return_response_only: bool = True,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run chat completion with ITS algorithm.

        Args:
            config: Per-request ITS configuration
            messages: OpenAI-format conversation messages
            tools: Optional tool definitions for function calling
            tool_choice: Optional tool choice strategy
            return_response_only: If True, return dict with 'message' and 'usage';
                                if False, return full algorithm result with 'usage'

        Returns:
            Dictionary with structure:
            - When return_response_only=True:
              {
                "message": {...},  # Selected assistant message
                "usage": {         # Aggregated usage from all LLM calls
                  "prompt_tokens": int,
                  "completion_tokens": int,
                  "total_tokens": int
                }
              }
            - When return_response_only=False:
              {
                "responses": [...],     # All responses with individual usage
                "response_counts": {...},
                "selected_index": int,
                "the_one": {...},       # Selected response
                "usage": {...}          # Aggregated usage
              }

        Raises:
            Exception: If LLM calls fail or model not specified
        """
        log_prefix = f"[{request_id}] " if request_id else ""

        # Validate that model is set
        if not config.model:
            raise ValueError("Model must be specified in ITSRequestConfig before running")

        # Get or create LM client (reuses cached instance if available)
        lm = self._get_or_create_lm(
            endpoint=config.api_endpoint,
            model=config.model,
            api_key=config.api_key,
            request_id=request_id,
        )

        # Convert to ChatMessages
        chat_messages = ChatMessages([ChatMessage(**msg) for msg in messages])

        logger.info(
            "%sRunning self-consistency: budget=%s, endpoint=%s, model=%s, messages=%s, tools=%s",
            log_prefix,
            config.budget,
            config.api_endpoint,
            config.model,
            len(messages),
            "yes" if tools else "no",
        )

        try:
            # Run the ITS algorithm
            # The algorithm now always returns usage information
            result = await self._algorithm.ainfer(
                lm=lm,
                prompt_or_messages=chat_messages,
                budget=config.budget,
                return_response_only=return_response_only,
                tools=tools,
                tool_choice=tool_choice,
            )

            # Result structure depends on return_response_only:
            # - True: {"message": {...}, "usage": {...}}
            # - False: SelfConsistencyResult with .usage field

            if return_response_only:
                # Result is already a dict with message and usage
                logger.info(
                    "%sITS algorithm completed successfully. Usage: %s",
                    log_prefix,
                    result["usage"],
                )
                return result
            else:
                # Result is SelfConsistencyResult object
                # Convert to dict format
                result_dict = {
                    "responses": result.responses,
                    "response_counts": result.response_counts,
                    "selected_index": result.selected_index,
                    "the_one": result.the_one if isinstance(result.the_one, dict) else result.the_one.model_dump(),
                    "usage": result.usage
                }
                logger.info(
                    "%sITS algorithm completed successfully. Usage: %s (selected_index=%s)",
                    log_prefix,
                    result.usage,
                    result.selected_index,
                )
                return result_dict

        except Exception as e:
            logger.error(
                "%sITS algorithm failed: %s",
                log_prefix,
                e,
                exc_info=True,
            )
            raise

    def clear_cache(self):
        """Clear the LM client cache.

        Useful for testing or if endpoint configurations change.
        """
        logger.info(f"Clearing LM client cache ({len(self._lm_cache)} entries)")
        self._lm_cache.clear()

    def shutdown(self):
        """Cleanup resources on service shutdown."""
        logger.info("ITSOrchestrator shutting down")
        self.clear_cache()
