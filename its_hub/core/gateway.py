"""Concrete ITS gateway implementation.

This module provides a long-lived gateway with per-request configuration,
similar to how HTTP services reuse client instances across requests.
"""

import logging
import time
from dataclasses import fields, replace
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
from its_hub.core.algorithms.adaptive_self_consistency import AdaptiveSelfConsistency
from its_hub.core.algorithms.beta_self_consistency import BetaSelfConsistency
from its_hub.core.algorithms.self_consistency import (
    SelfConsistency,
    create_regex_projection_function,
)
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.orchestrator import LMOrchestrator

logger = logging.getLogger(__name__)

# The self-consistency family shares the same voting surface (regex/tool-vote
# projection) and only differs in how it decides when to stop sampling.
SELF_CONSISTENCY_ALGORITHMS = frozenset(
    {
        "self-consistency",
        "adaptive-self-consistency",
        "beta-self-consistency",
    }
)

# All currently supported algorithms are self-consistency variants.
SUPPORTED_ALGORITHMS = SELF_CONSISTENCY_ALGORITHMS

# System defaults seeded into every gateway's default config.
_DEFAULT_BUDGET = 4
_DEFAULT_ALG = "self-consistency"


class ITSGateway(AbstractGateway):
    """Long-lived gateway for running ITS algorithms.

    Initialized once at service startup and reused across all requests.
    Per-request configuration is passed as arguments to arun_chat_completion().

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
        orchestrator: AbstractOrchestrator | None = None,
        default_config: ITSRequestConfig | None = None,
    ):
        if orchestrator is None:
            orchestrator = LMOrchestrator()
        self._orchestrator = orchestrator
        self._default_config = default_config or ITSRequestConfig(
            budget=_DEFAULT_BUDGET, alg=_DEFAULT_ALG
        )
        logger.info(
            f"ITSGateway initialized with default alg={self._default_config.alg} and budget={self._default_config.budget}"
        )

    @property
    def default_config(self) -> ITSRequestConfig:
        """The service-default config set via :meth:`configure`."""
        return self._default_config

    def configure(self, config: ITSRequestConfig) -> None:
        """Merge ``config`` into the service default.

        Validation (alg, regex, thresholds) runs automatically in
        ``ITSRequestConfig.__post_init__`` via ``_merge``.
        """
        self._default_config = self._merge(self._default_config, config)
        logger.info(
            "ITSGateway reconfigured with default alg=%s", self._default_config.alg
        )

    @staticmethod
    def _merge(base: ITSRequestConfig, overlay: ITSRequestConfig) -> ITSRequestConfig:
        """Return a copy of ``base`` with non-``None`` fields from ``overlay``."""
        return replace(
            base,
            **{
                f.name: getattr(overlay, f.name)
                for f in fields(overlay)
                if getattr(overlay, f.name) is not None
            },
        )

    def _build_algorithm(self, config: ITSRequestConfig) -> AbstractScalingAlgorithm:
        """Construct a fresh scaling algorithm from a config snapshot."""
        alg = config.alg

        projection_func = None
        if config.regex_patterns:
            projection_func = create_regex_projection_function(config.regex_patterns)

        common_kwargs = {
            "consistency_space_projection_func": projection_func,
            "exclude_args": config.exclude_tool_args,
            "orchestrator": self._orchestrator,
        }
        # Only override the algorithm's default tool_vote when one is explicitly
        # provided; otherwise let it fall back to DEFAULT_TOOL_VOTE so that
        # tool-calling responses still vote sensibly with no extra config.
        if config.tool_vote is not None:
            common_kwargs["tool_vote"] = config.tool_vote

        if alg == "adaptive-self-consistency":
            extra = {} if config.threshold is None else {"threshold": config.threshold}
            return AdaptiveSelfConsistency(**common_kwargs, **extra)
        elif alg == "beta-self-consistency":
            extra = (
                {}
                if config.confidence_threshold is None
                else {"confidence_threshold": config.confidence_threshold}
            )
            return BetaSelfConsistency(**common_kwargs, **extra)
        else:  # self-consistency
            return SelfConsistency(**common_kwargs)

    def _build_lm(
        self,
        config: ITSRequestConfig,
        request_id: str | None = None,
    ) -> OpenAICompatibleLanguageModel:
        """Construct a one-shot LM client from a config snapshot."""
        log_prefix = f"[{request_id}] " if request_id else ""
        logger.info(
            "%sCreating LM client: endpoint=%s, model=%s",
            log_prefix,
            config.api_endpoint,
            config.model,
        )
        return OpenAICompatibleLanguageModel(
            endpoint=config.api_endpoint,
            api_key=config.api_key,
            model_name=config.model,
            temperature=config.temperature,
        )

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

        merged = self._merge(self._default_config, config)

        if not merged.api_endpoint:
            raise ValueError("api_endpoint must be specified")
        if not merged.model:
            raise ValueError("Model must be specified")

        # Snapshot the algorithm + LM for this request only. Nothing on `self`
        # is mutated, so a concurrent /configure cannot swap the algorithm or
        # close this request's HTTP session mid-flight.
        algorithm = self._build_algorithm(merged)
        lm = self._build_lm(merged, request_id)

        chat_messages = ChatMessages([ChatMessage.from_dict(msg) for msg in messages])

        logger.info(
            "%sRunning ITS: algorithm=%s, budget=%s, endpoint=%s, model=%s, messages=%s, tools=%s",
            log_prefix,
            type(algorithm).__name__,
            merged.budget,
            merged.api_endpoint,
            merged.model,
            len(messages),
            "yes" if tools else "no",
        )

        t0 = time.monotonic()
        try:
            result = await algorithm.ainfer(
                lm=lm,
                prompt_or_messages=chat_messages,
                budget=merged.budget,
                return_response_only=False,
                tools=tools,
                tool_choice=tool_choice,
            )
        finally:
            await lm.close()

        usage_dict = {}
        if isinstance(result.usage, GenerationUsage):
            usage_dict = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "num_calls": result.usage.num_calls,
            }

        duration_s = time.monotonic() - t0

        if return_response_only:
            logger.info(
                "%sITS completed in %.2fs. Usage: %s",
                log_prefix,
                duration_s,
                usage_dict,
            )
            return {"message": result.the_one, "usage": usage_dict, "alg": merged.alg}

        logger.info(
            "%sITS completed in %.2fs. Usage: %s (selected_index=%s)",
            log_prefix,
            duration_s,
            usage_dict,
            result.selected_index,
        )
        return {
            "responses": result.responses,
            "response_counts": result.response_counts,
            "selected_index": result.selected_index,
            "the_one": result.the_one,
            "usage": usage_dict,
            "alg": merged.alg,
        }
