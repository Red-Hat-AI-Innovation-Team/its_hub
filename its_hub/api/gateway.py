"""Abstract gateway interface for inference-time scaling services."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from its_hub.api.types import ITSRequestConfigUpdate


class AbstractGateway(ABC):
    """Abstract base class for ITS gateway adapters.

    Gateway implementations manage LM client lifecycle, algorithm dispatch,
    and request/response conversion. Platform-specific adapters (Envoy ext_proc,
    FastAPI IaaS, etc.) use a concrete gateway to run ITS algorithms
    on incoming chat completion requests.
    """

    @abstractmethod
    async def arun_chat_completion(
        self,
        config: ITSRequestConfigUpdate,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        return_response_only: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run chat completion with ITS algorithm asynchronously.

        Args:
            config: Per-request ITS configuration overlay (merged over the
                gateway's service default at run time)
            messages: OpenAI-format conversation messages
            tools: Optional tool definitions for function calling
            tool_choice: Optional tool choice strategy
            return_response_only: If True, return dict with 'message' and 'usage';
                                if False, return full algorithm result with 'usage'
            **kwargs: Additional parameters (e.g., request_id for logging)

        Returns:
            Dictionary with 'message'/'the_one' and aggregated 'usage'.
        """
        pass

    def run_chat_completion(
        self,
        config: ITSRequestConfigUpdate,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Synchronous wrapper for arun_chat_completion."""
        return asyncio.run(self.arun_chat_completion(config, messages, **kwargs))
