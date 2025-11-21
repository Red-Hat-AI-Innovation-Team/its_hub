
"""LLM Router for dynamic algorithm selection based on query complexity.
The router analyzes incoming queries and selects the most appropriate scaling algorithm
along with optimal parameters (budget, models) based on predicted complexity.
"""

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from its_hub.lms import LiteLLMLanguageModel
from its_hub.types import ChatMessage, ChatMessages

# Set up dedicated router logger
logger = logging.getLogger(__name__)

# Create a separate file handler for router logs
router_logger = logging.getLogger("its_hub.router")
router_logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure file handler for router logs
router_file_handler = logging.FileHandler(log_dir / "router.log")
router_file_handler.setLevel(logging.DEBUG)

# Configure console handler for router logs
router_console_handler = logging.StreamHandler()
router_console_handler.setLevel(logging.INFO)

# Configure formatter
router_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
router_file_handler.setFormatter(router_formatter)
router_console_handler.setFormatter(router_formatter)

# Add handlers to router logger
if not router_logger.handlers:
    router_logger.addHandler(router_file_handler)
    router_logger.addHandler(router_console_handler)

# Prevent propagation to root logger to avoid duplicate logs
router_logger.propagate = False

# ============================================================================
# Router Prompts
# ============================================================================

ROUTER_SYSTEM_PROMPT = """You are an AI router that selects the optimal inference strategy for generating the next response in a conversation based on the previous conversation history.

The task that you are solving will be a multi-step multi-turn agentic task. 


The task is a multi-turn task, so at every step you are trying to select the best strategy to generate only the next(1 response) towards solving the user's task. So decision should be made with this in mind.
You will be receiving conversation history which basically would contain user query, agent response, tool response, your task is to select the best inference time scaling strategy with the most optimal set of parameters required to get the most immediate next step only.

You must output:

1. reasoning: reason briefly on what would be the best strategy and parameters for the next step generation
2. algorithm: self-consistency | best-of-n
3. budget: integer sample count
4. model: model tier or name

Return ONLY valid JSON.

---------------------------------------------------------
General Algorithm Guidelines
---------------------------------------------------------

Self-Consistency
- Use when the next step is deterministic and governed by rules.
- Best for:
  * policy checks
  * filling tool arguments from known context
  * confirmations
  * summarizing tool results

Best-of-N
- Use when there are multiple plausible next actions or phrasing variations.
- Best for:
  * selecting which tool to call in case of multiple different possible directions
  * constructing tool arguments when several interpretations exist
  * ambiguous state transitions
  * uncertain policy interpretation


Make sure to choose ONLY from the available algorithms and models provided to you.


You can use budget as low as 1 for tasks which are very simple, trivial and obvious, and you can go upto the max budget allowed based on the predicted complexity of the next step.


To provide you more context regarding the task, you will be provided with the environment policies and rules for you to have better sense of direction where the agent should be heading towards.
---------------------------------------------------------
Required Output Format
```json
{
  "reasoning": "<brief reasoning>",
  "algorithm": "self-consistency|best-of-n",
  "budget": <integer>,
  "model": "<model_name_or_size>"
}
```"""

ROUTER_USER_PROMPT_TEMPLATE = """Analyze this conversation and select the optimal routing configuration for the next response:

Conversation:
{conversation}

Available algorithms and their configurations:
You can only choose from the following algorithms and their configurations:
{algorithms_config}

Max allowed budget: {max_budget}

Respond with the routing decision in JSON format."""



def prepare_router_content(messages, algorithms_config, max_budget, num_turns_to_keep=3):

    # Convert to ChatMessages first if needed
    chat_messages = ChatMessages.from_prompt_or_messages(messages)

    # Get the underlying messages list and slice it
    messages_list = chat_messages.to_chat_messages()
    if len(messages_list) > num_turns_to_keep:
        messages_list = messages_list[-num_turns_to_keep:]
        chat_messages = ChatMessages(messages_list)

    conversation_str = chat_messages.to_prompt()

    return ROUTER_USER_PROMPT_TEMPLATE.format(conversation=conversation_str,
                                       algorithms_config=algorithms_config,
                                       max_budget=max_budget)



def parse_router_response(response_content: str) -> dict:
    # Try to extract JSON from the response
    json_match = re.match(r'```json(.*)```', response_content, re.DOTALL)

    if not json_match:
        raise ValueError(
            f"Could not find valid JSON in router response: {response_content[:200]}"
        )

    json_str = json_match.group(1)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in router response: {e}") from e

    return data



class RouterDecision(BaseModel):
    reasoning: str
    algorithm: str
    budget: int
    model: str

    @classmethod
    def from_dict(cls, data: dict) -> "RouterDecision":
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "reasoning": self.reasoning,
            "algorithm": self.algorithm,
            "budget": self.budget,
            "model": self.model,
        }
class LLMRouter:
    def __init__(
        self,
        router_model: str = "gpt-4o-mini",
        router_api_key: str | None = None,
        router_base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_logging: bool = True,
        system_prompt: str | None = None,
        router_max_budget: int = 8,
    ):
        """Initialize the LLM Router.

        Args:
            router_model: LiteLLM model name for router (e.g., "gpt-4o-mini")
            router_api_key: API key for the router model
            router_base_url: Base URL for the router model endpoint
            temperature: Temperature for router LLM (0.0 for deterministic)
            max_tokens: Maximum tokens for router response
            enable_logging: Whether to log routing decisions
            system_prompt: Custom system prompt for router (uses default if None)
        """
        self.enable_logging = enable_logging

        # Use custom system prompt or default
        router_system_prompt = system_prompt if system_prompt is not None else ROUTER_SYSTEM_PROMPT
        router_logger.info(f"Using custom system prompt: {router_system_prompt}")

        # Initialize router LLM using LiteLLM
        self.router_lm = LiteLLMLanguageModel(
            model_name=router_model,
            api_key=router_api_key,
            api_base=router_base_url,
            system_prompt=router_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            is_async=True,
        )
        self.router_max_budget = router_max_budget

        router_logger.info(f"Initialized LLM Router with model={router_model}")

    async def route(
        self,
        messages: list[ChatMessage] | ChatMessages,
        available_algorithms: dict[str, dict],
        num_turns_to_keep=3,
        max_budget: int | None = None
    ) -> dict:


        # Format algorithms config for prompt
        alg_lines = []
        for alg_name, config in available_algorithms.items():
            models_str = ", ".join(config.get("models", []))
            judge_info = f" (judge: {config.get('judge_model')})" if config.get("judge_model") else ""
            alg_lines.append(f"  - {alg_name}: models=[{models_str}]{judge_info}")
        algorithms_config_str = "\n".join(alg_lines)
        
        router_logger.info(f"Available Algorithms:\n{algorithms_config_str}")

        # Prepare routing prompt
        routing_prompt = prepare_router_content(
            messages=messages,
            algorithms_config=algorithms_config_str,
            max_budget=max_budget if max_budget is not None else self.router_max_budget,
            num_turns_to_keep=num_turns_to_keep
        )                

        # Log router input
        if self.enable_logging:
            router_logger.info(f"Router Input:\n{routing_prompt}")

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

            # Log raw router response
            if self.enable_logging:
                router_logger.info(f"Router raw response:\n{response_content}")

            # Parse the response
            decision = parse_router_response(response_content)

            decision = RouterDecision.from_dict(decision)
            
            if decision.algorithm not in available_algorithms:
                router_logger.warning(f"Router selected unavailable algorithm: {decision.algorithm}. Using fallback routing.")
                return RouterDecision.from_dict(self._default_routing(available_algorithms))
            
            return decision

        except Exception as e:
            print(e)
            router_logger.error(f"Router LLM call failed: {e}. Using fallback routing.")
            return RouterDecision.from_dict(self._default_routing(available_algorithms))
        
    def _default_routing(self, available_algorithms: dict[str, dict]) -> dict:
        
        default_budget = 1
        first_alg = next(iter(available_algorithms.keys()))
        first_model = available_algorithms[first_alg].get("models", ["default"])[0]

        router_logger.info(f"Using fallback routing: {first_alg} with budget {default_budget}")

        return {
                "algorithm": first_alg,
                "budget": default_budget,
                "model": first_model,
                "reasoning": "Fallback routing due to router LLM failure",
                }
