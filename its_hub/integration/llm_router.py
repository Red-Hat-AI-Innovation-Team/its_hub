"""LLM Router for dynamic algorithm selection based on query complexity.

The router analyzes incoming queries and selects the most appropriate scaling algorithm
along with optimal parameters (budget, models) based on predicted complexity.
"""

import json
import logging
import re

from pydantic.dataclasses import dataclass

from its_hub.lms import LiteLLMLanguageModel
from its_hub.types import ChatMessage, ChatMessages

logger = logging.getLogger(__name__)

# ============================================================================
# Router Prompts
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are an expert AI system router that selects the optimal inference-time scaling algorithm based on the task complexity.

Your task is to analyze the current trajectory and predict:
1. The COMPLEXITY of the next response to be generated (low, medium, high)
2. The best ALGORITHM to use from the available algorithms
3. The optimal BUDGET (number of samples to scale)
4. The best model to solve the task based on the complexity from the available models. 

Consider these factors when evaluating complexity:
- Mathematical reasoning: equations, proofs, multi-step calculations
- Code generation: algorithm design, debugging, refactoring
- Multi-step reasoning: planning, analysis, decomposition
- Tool usage: multiple tool calls, sequential dependencies
- Ambiguity: unclear requirements, multiple valid approaches

Algorithm selection guidelines:
- Use **self-consistency** when the task has a single verifiable or deterministic answer (e.g., math, logic, factual QA, code with tests). It reduces variance and improves reliability through aggregation.
- Use **best-of-n with LLM judge** when outputs are subjective, multi-criteria, or process-driven (e.g., reasoning steps, summaries, tool selection, creative tasks). It ensures higher quality through rubric-based selection.
- Use **particle-filtering** when the task involves sequential decision-making, multiple dependent steps, or evolving uncertainty across turns (e.g., agents, tool chains, planning problems).

For best-of-n and particle-filtering:
For best-of-n and particle-filtering:
- Select appropriate judge/reward models based on task type
- For general tasks: use llm-judge with fast models
- For math: consider specialized reward models if available

Respond ONLY with valid JSON in this exact format:
{
  "reasoning": "<brief explanation>"
  "complexity": "low|medium|high",
  "algorithm": <algorithm_name>,
  "budget": "<integer>",
  "generation_model": "<model_name>",
  "judge_model": "<model_name_or_llm-judge>" | null,
}"""

ROUTER_USER_PROMPT_TEMPLATE = """Analyze this conversation context and select the optimal routing configuration:

Conversation context:
{conversation_context}

Available algorithms:
- {available_algorithms}

Available models:
- Generation models: {available_generation_models}
- Judge/reward models: {available_judge_models}

Max budget: {max_budget}

Respond with the routing decision in JSON format."""


# ============================================================================
# Router Output Parsing Utilities
# ============================================================================


@dataclass
class RoutingDecision:
    """Represents a routing decision from the LLM router."""

    complexity: str  # "low" | "medium" | "high"
    algorithm: str  # "self-consistency" | "best-of-n" | "particle-filtering"
    budget: int
    generation_model: str
    judge_model: str | None
    reasoning: str

    def __post_init__(self):
        """Validate the routing decision."""
        valid_complexities = {"low", "medium", "high"}
        if self.complexity not in valid_complexities:
            raise ValueError(
                f"Invalid complexity: {self.complexity}. "
                f"Must be one of {valid_complexities}"
            )

        valid_algorithms = {"self-consistency", "best-of-n", "particle-filtering"}
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid algorithm: {self.algorithm}. "
                f"Must be one of {valid_algorithms}"
            )

        if self.budget < 1:
            raise ValueError(f"Budget must be >= 1, got {self.budget}")

        # Judge model is required for best-of-n and particle-filtering
        if self.algorithm in {"best-of-n", "particle-filtering"} and not self.judge_model:
            logger.warning(
                f"Algorithm {self.algorithm} typically requires a judge model, "
                "but none was specified. Using default."
            )


def parse_router_response(response_content: str) -> RoutingDecision:
    """Parse the LLM router's response into a RoutingDecision object.

    Args:
        response_content: The text response from the router LLM

    Returns:
        RoutingDecision object with parsed fields

    Raises:
        ValueError: If the response cannot be parsed or is invalid
    """
    # Try to extract JSON from the response
    # The LLM might include markdown code blocks or extra text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_content, re.DOTALL)

    if not json_match:
        raise ValueError(
            f"Could not find valid JSON in router response: {response_content[:200]}"
        )

    json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in router response: {e}") from e

    # Extract and validate required fields
    try:
        decision = RoutingDecision(
            complexity=data["complexity"].lower(),
            algorithm=data["algorithm"].lower(),
            budget=int(data["budget"]),
            generation_model=data["generation_model"],
            judge_model=data.get("judge_model"),
            reasoning=data.get("reasoning", "No reasoning provided"),
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in router response: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid field value in router response: {e}") from e

    return decision


def apply_routing_constraints(
    decision: RoutingDecision,
    max_budget: int = 32,
    available_generation_models: list[str] | None = None,
    available_judge_models: list[str] | None = None,
) -> RoutingDecision:
    """Apply constraints and defaults to a routing decision.

    Args:
        decision: The original routing decision
        max_budget: Maximum allowed budget
        available_generation_models: List of available generation models
        available_judge_models: List of available judge models

    Returns:
        Modified RoutingDecision with constraints applied
    """
    # Constrain budget
    constrained_budget = min(decision.budget, max_budget)
    if constrained_budget != decision.budget:
        logger.info(
            f"Budget constrained from {decision.budget} to {constrained_budget}"
        )

    # Validate generation model
    if available_generation_models and decision.generation_model not in available_generation_models:
        logger.warning(
            f"Requested generation model '{decision.generation_model}' not available. "
            f"Using '{available_generation_models[0]}'"
        )
        generation_model = available_generation_models[0]
    else:
        generation_model = decision.generation_model

    # Validate judge model
    judge_model = decision.judge_model
    if decision.algorithm in {"best-of-n", "particle-filtering"}:
        if available_judge_models and judge_model not in available_judge_models:
            logger.warning(
                f"Requested judge model '{judge_model}' not available. "
                f"Using '{available_judge_models[0] if available_judge_models else 'llm-judge'}'"
            )
            judge_model = available_judge_models[0] if available_judge_models else "llm-judge"

    return RoutingDecision(
        complexity=decision.complexity,
        algorithm=decision.algorithm,
        budget=constrained_budget,
        generation_model=generation_model,
        judge_model=judge_model,
        reasoning=decision.reasoning,
    )


class LLMRouter:
    """Routes queries to optimal scaling algorithms based on predicted complexity.

    The router uses an LLM to analyze the query and conversation context to select:
    - The most appropriate scaling algorithm
    - Optimal budget allocation
    - Best models for generation and judging
    """

    def __init__(
        self,
        router_model: str = "gpt-4.1-mini",
        router_api_key: str | None = None,
        router_base_url: str | None = None,
        max_budget: int = 16,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        available_generation_models: list[str] | None = None,
        available_judge_models: list[str] | None = None,
        enable_routing_logging: bool = True,
    ):
        """Initialize the LLM Router.

        Args:
            router_model: LiteLLM model name for the router (e.g., "gpt-4.1-mini")
            router_api_key: API key for the router model
            router_base_url: Base URL for the router model endpoint
            max_budget: Maximum allowed budget for any algorithm
            temperature: Temperature for router LLM (0.0 for deterministic)
            max_tokens: Maximum tokens for router response
            available_generation_models: List of models available for generation
            available_judge_models: List of models available for judging
            enable_routing_logging: Whether to log routing decisions
        """
        self.max_budget = max_budget
        self.available_generation_models = available_generation_models or []
        self.available_judge_models = available_judge_models or []
        self.enable_logging = enable_routing_logging

        # Initialize router LLM using LiteLLM
        self.router_lm = LiteLLMLanguageModel(
            model_name=router_model,
            api_key=router_api_key,
            api_base=router_base_url,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            is_async=True,
        )

        logger.info(
            f"Initialized LLM Router with model={router_model}, "
            f"max_budget={max_budget}"
        )

    async def route(
        self,
        messages: list[ChatMessage] | ChatMessages,
    ) -> RoutingDecision:
        """Route a query to the optimal algorithm with parameters.

        Args:
            messages: The conversation messages to route

        Returns:
            RoutingDecision with algorithm, budget, and model selections
        """
        # Convert to ChatMessages if needed
        chat_messages = ChatMessages.from_prompt_or_messages(messages)

        # Extract the query (last user message)
        message_list = chat_messages.to_chat_messages()
        query = self._extract_query(message_list)

        # Prepare routing prompt
        routing_prompt = ROUTER_USER_PROMPT_TEMPLATE.format(
            query=query,
            generation_models=", ".join(self.available_generation_models) or "default",
            judge_models=", ".join(self.available_judge_models) or "llm-judge",
            context_length=len(message_list),
        )

        # Query the router LLM
        router_messages = [ChatMessage(role="user", content=routing_prompt)]

        try:
            response_tuple = await self.router_lm.agenerate(router_messages)

            # Unpack response - LiteLLM returns (message_dict, usage_dict)
            if isinstance(response_tuple, tuple):
                response_message, _ = response_tuple
            else:
                response_message = response_tuple

            response_content = response_message.get("content", "")

            # Parse the response
            decision = parse_router_response(response_content)

            # Apply constraints
            decision = apply_routing_constraints(
                decision,
                max_budget=self.max_budget,
                available_generation_models=self.available_generation_models,
                available_judge_models=self.available_judge_models,
            )

            if self.enable_logging:
                logger.info(
                    f"Routing decision: complexity={decision.complexity}, "
                    f"algorithm={decision.algorithm}, budget={decision.budget}, "
                    f"reasoning={decision.reasoning}"
                )

            return decision

        except Exception as e:
            logger.error(f"Router LLM call failed: {e}. Using fallback routing.")
            return self._fallback_routing()

    def _extract_query(self, messages: list[ChatMessage]) -> str:
        """Extract the main query from conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            The extracted query string
        """
        # Find the last user message
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.extract_text_content()

        # Fallback: use all messages
        return ChatMessages(messages).to_prompt()

    def _fallback_routing(self) -> RoutingDecision:
        """Provide a fallback routing decision when the router LLM fails.

        Returns:
            Conservative RoutingDecision using self-consistency
        """
        logger.warning("Using fallback routing: self-consistency with budget=8")

        return RoutingDecision(
            complexity="medium",
            algorithm="self-consistency",
            budget=8,
            generation_model=self.available_generation_models[0] if self.available_generation_models else "default",
            judge_model=None,
            reasoning="Fallback routing due to router LLM failure",
        )
