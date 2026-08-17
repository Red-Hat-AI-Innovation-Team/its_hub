"""Concrete ITS gateway implementation.

This module provides a long-lived gateway with per-request configuration,
similar to how HTTP services reuse client instances across requests.
"""

import hashlib
import logging
import time
from collections import OrderedDict
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
    validate_regex_patterns,
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

    _DEFAULT_MAX_LM_CACHE_SIZE = 64

    def __init__(
        self,
        algorithm: AbstractScalingAlgorithm | None = None,
        orchestrator: AbstractOrchestrator | None = None,
        max_lm_cache_size: int = _DEFAULT_MAX_LM_CACHE_SIZE,
    ):
        if orchestrator is None:
            orchestrator = LMOrchestrator()
        self._orchestrator = orchestrator

        if algorithm is None:
            algorithm = SelfConsistency(orchestrator=orchestrator)
        self._algorithm = algorithm

        self._algorithm_name = type(algorithm).__name__
        self._max_lm_cache_size = max_lm_cache_size
        self._lm_cache: OrderedDict[
            tuple[str, str, str], OpenAICompatibleLanguageModel
        ] = OrderedDict()
        logger.info("ITSGateway initialized with algorithm=%s", self._algorithm_name)

    def configure(
        self,
        alg: str,
        regex_patterns: list[str] | None = None,
        tool_vote: str | None = None,
        exclude_tool_args: list[str] | None = None,
        threshold: float | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        """Configure the gateway's scaling algorithm at runtime.

        ``threshold`` applies only to ``adaptive-self-consistency`` and
        ``confidence_threshold`` only to ``beta-self-consistency``; both fall
        back to the algorithm's own default when left as ``None``.

        Raises ValueError for unsupported algorithms or invalid options.
        """
        if alg not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Algorithm {alg!r} not supported. Choose from: {SUPPORTED_ALGORITHMS}"
            )

        # The self-consistency family shares projection/tool-vote plumbing.
        projection_func = None
        if regex_patterns:
            validate_regex_patterns(regex_patterns)
            projection_func = create_regex_projection_function(regex_patterns)

        common_kwargs = {
            "consistency_space_projection_func": projection_func,
            "exclude_args": exclude_tool_args,
            "orchestrator": self._orchestrator,
        }
        # Only override the algorithm's default tool_vote when one is explicitly
        # provided; otherwise let it fall back to DEFAULT_TOOL_VOTE so that
        # tool-calling responses still vote sensibly with no extra config.
        if tool_vote is not None:
            common_kwargs["tool_vote"] = tool_vote

        if alg == "adaptive-self-consistency":
            extra = {} if threshold is None else {"threshold": threshold}
            algorithm = AdaptiveSelfConsistency(**common_kwargs, **extra)
        elif alg == "beta-self-consistency":
            extra = (
                {}
                if confidence_threshold is None
                else {"confidence_threshold": confidence_threshold}
            )
            algorithm = BetaSelfConsistency(**common_kwargs, **extra)
        else:  # self-consistency
            algorithm = SelfConsistency(**common_kwargs)

        self._algorithm = algorithm
        self._algorithm_name = type(algorithm).__name__
        logger.info("ITSGateway reconfigured with algorithm=%s", self._algorithm_name)

    @staticmethod
    def _hash_api_key(api_key: str | None) -> str:
        return hashlib.sha256((api_key or "").encode()).hexdigest()[:16]

    async def _get_or_create_lm(
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

        if cache_key in self._lm_cache:
            self._lm_cache.move_to_end(cache_key)
            logger.debug(
                "%sReusing cached LM client: endpoint=%s, model=%s",
                log_prefix,
                endpoint,
                model,
            )
            return self._lm_cache[cache_key]

        evicted_lm = None
        if len(self._lm_cache) >= self._max_lm_cache_size:
            evicted_key, evicted_lm = self._lm_cache.popitem(last=False)
            logger.info(
                "%sEvicting LM client from cache: endpoint=%s, model=%s",
                log_prefix,
                evicted_key[0],
                evicted_key[1],
            )

        logger.info(
            "%sCreating new LM client: endpoint=%s, model=%s",
            log_prefix,
            endpoint,
            model,
        )
        lm = OpenAICompatibleLanguageModel(
            endpoint=endpoint,
            api_key=api_key or "",
            model_name=model,
        )
        self._lm_cache[cache_key] = lm

        if evicted_lm is not None:
            await evicted_lm.close()

        return lm

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

        if not config.api_endpoint:
            raise ValueError("api_endpoint must be specified in ITSRequestConfig")
        if not config.model:
            raise ValueError(
                "Model must be specified in ITSRequestConfig before running"
            )

        lm = await self._get_or_create_lm(
            endpoint=config.api_endpoint,
            model=config.model,
            api_key=config.api_key,
            request_id=request_id,
        )

        chat_messages = ChatMessages([ChatMessage.from_dict(msg) for msg in messages])

        logger.info(
            "%sRunning ITS: algorithm=%s, budget=%s, endpoint=%s, model=%s, messages=%s, tools=%s",
            log_prefix,
            self._algorithm_name,
            config.budget,
            config.api_endpoint,
            config.model,
            len(messages),
            "yes" if tools else "no",
        )

        t0 = time.monotonic()
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
            return {"message": result.the_one, "usage": usage_dict}

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
        }

    async def ashutdown(self) -> None:
        """Cleanup resources on service shutdown.

        Closes all cached LM clients (releasing aiohttp sessions) then
        clears the cache.
        """
        logger.info(
            "ITSGateway shutting down, closing %d LM clients",
            len(self._lm_cache),
        )
        for lm in self._lm_cache.values():
            await lm.close()
        self._lm_cache.clear()
