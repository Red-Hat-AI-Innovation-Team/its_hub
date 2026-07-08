"""Concrete ITS gateway implementation.

This module provides a long-lived gateway with per-request configuration,
similar to how HTTP services reuse client instances across requests.
"""

import hashlib
import logging
from typing import Any

from its_hub.api import (
    AbstractGateway,
    AbstractOrchestrator,
    AbstractScalingAlgorithm,
    ChatMessage,
    ChatMessages,
    GenerationUsage,
    ITSRequestConfig,
)
from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.orchestrator import LMOrchestrator

logger = logging.getLogger(__name__)


class ITSGateway(AbstractGateway):
    """Long-lived gateway for running ITS algorithms.

    Initialized once at service startup and reused across all requests.
    Per-request configuration is passed as arguments to arun_chat_completion().

    Maintains a cache of LM clients keyed by (endpoint, model, hashed_api_key)
    to reuse HTTP connections across requests while preventing credential
    cross-contamination.

    Example:
        # At service startup
        gateway = ITSGateway()

        # Per request
        config = ITSRequestConfig(
            budget=10,
            api_endpoint="http://envoy-cluster/v1",
            model="gpt-4",
            api_key="sk-...",
        )
        result = await gateway.arun_chat_completion(config, messages)
    """

    def __init__(
        self,
        algorithm: AbstractScalingAlgorithm | None = None,
        orchestrator: AbstractOrchestrator | None = None,
    ):
        if orchestrator is None:
            orchestrator = LMOrchestrator()
        self._orchestrator = orchestrator

        if algorithm is None:
            algorithm = SelfConsistency(orchestrator=orchestrator)
        self._algorithm = algorithm

        self._lm_cache: dict[tuple[str, str, str], OpenAICompatibleLanguageModel] = {}
        logger.info("ITSGateway initialized")

    @staticmethod
    def _hash_api_key(api_key: str | None) -> str:
        return hashlib.sha256((api_key or "").encode()).hexdigest()[:16]

    def _get_or_create_lm(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        request_id: str | None = None,
    ) -> OpenAICompatibleLanguageModel:
        """Get cached LM client or create new one.

        Clients are cached by (endpoint, model, hashed_api_key) to reuse
        HTTP connections while isolating different credentials.
        """
        cache_key = (endpoint, model, self._hash_api_key(api_key))
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
                api_key=api_key or "",
                model_name=model,
            )
        else:
            logger.debug(
                "%sReusing cached LM client: endpoint=%s, model=%s",
                log_prefix,
                endpoint,
                model,
            )

        return self._lm_cache[cache_key]

    async def arun_chat_completion(
        self,
        config: ITSRequestConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        return_response_only: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        request_id = kwargs.get("request_id")
        log_prefix = f"[{request_id}] " if request_id else ""

        if not config.model:
            raise ValueError(
                "Model must be specified in ITSRequestConfig before running"
            )

        lm = self._get_or_create_lm(
            endpoint=config.api_endpoint,
            model=config.model,
            api_key=config.api_key,
            request_id=request_id,
        )

        chat_messages = ChatMessages(
            [ChatMessage.from_dict(msg) for msg in messages]
        )

        logger.info(
            "%sRunning ITS: budget=%s, endpoint=%s, model=%s, messages=%s, tools=%s",
            log_prefix,
            config.budget,
            config.api_endpoint,
            config.model,
            len(messages),
            "yes" if tools else "no",
        )

        try:
            result = await self._algorithm.ainfer(
                lm=lm,
                prompt_or_messages=chat_messages,
                budget=config.budget,
                return_response_only=False,
                tools=tools,
                tool_choice=tool_choice,
            )

            usage_dict = {}
            if isinstance(result.usage, GenerationUsage):
                usage_dict = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }

            if return_response_only:
                logger.info(
                    "%sITS completed. Usage: %s",
                    log_prefix,
                    usage_dict,
                )
                return {
                    "message": result.the_one,
                    "usage": usage_dict,
                }
            else:
                result_dict = {
                    "responses": result.responses,
                    "response_counts": result.response_counts,
                    "selected_index": result.selected_index,
                    "the_one": result.the_one,
                    "usage": usage_dict,
                }
                logger.info(
                    "%sITS completed. Usage: %s (selected_index=%s)",
                    log_prefix,
                    usage_dict,
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
        """Clear the LM client cache."""
        logger.info("Clearing LM client cache (%d entries)", len(self._lm_cache))
        self._lm_cache.clear()

    def shutdown(self) -> None:
        """Cleanup resources on service shutdown."""
        logger.info("ITSGateway shutting down")
        self.clear_cache()
