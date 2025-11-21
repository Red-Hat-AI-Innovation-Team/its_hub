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

# Configure verifier logger
def setup_verifier_logger(log_file: str = "verifier.log") -> logging.Logger:
    """Set up a dedicated logger for the verifier that streams to both file and console."""
    logger = logging.getLogger("its_hub.verifier")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - append mode
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger

# Initialize the verifier logger
verifier_logger = setup_verifier_logger()

POLICY = open("policy.md", "r").read()

VERIFIER_SYSTEM_PROMPT = """
You are a verifier that is tasked with verifiying if the response/action taken by the agent is compliant with the domain policy or not.


You will check if the agent's response or action deviates from the policy, if it deviates you will be generating a reflection on the deviation and why it deviated from the policy.
You will output two things:
1. deviation: a boolean value indicating if the action taken deviates from the policy(true if it deviates, false if it doesn't)
2. reflection: a string value containing the reflection on the deviation and why it deviated from the policy.(Only to be generated if the deviation is true, else leave it as "")

Consider the following when making the decision:
1. Look at what the user requested and see what the agent did to fulfill that request. 
2. Check if the agent's response is actually compliant with the policy or did it diverge due to user's request. The user may ask for something that is not compliant with the policy, in that case the agent should not fulfill it and you should reflect on the deviation.
3. The agent should never execute actions that is divergent from the policy, if it does, your reflection should be able to bring it back on track.
4. You need to look at all the information you can get from the conversation history provided to you, including the tool responses, details that were previously extracted by the agent, etc., then make the decision based on what you understand from the information. Some of these state based details may be relevant to the compliance / misalignment with the policy.

Some examples of how the agent might deviate from the policy:
1. The user forces the agent to do something that is not compliant with the policy.
2. The agent ignores or forgets one of the rules and does something that is not compliant with the policy.
3. The user's request is not satisfiable according to the policy, but the agent tries to fulfill it anyway.
4. Agent misunderstands or does not consider some of the information about the user or the environment and does something that is not compliant with the policy.
5. The sequence of actions taken by the agent is not in the way the policy suggests.


Steps you need to follow to detect deviation:
1. Understand the user's ask.
2. Focus on the relevant section of the policy.
3. Reason about how the action might be conflicting based on the data you have in <reasoning></reasoning> tags before you generate the final response.

<reasoning>
<string>
</reasoning>


Output Format:
```json
{
    "reflection": <string>,
    "deviation": <boolean>,
}
```"""

VERIFIER_USER_PROMPT_TEMPLATE = """Analyze this conversation and the current action taken by the agent and check if it is compliant with the policy or not according to the rules provided.

--------------------------------
Conversation:
{conversation}
--------------------------------
Current Action:
{current_action}
--------------------------------
Policy:
{policy}
--------------------------------
Respond with the verifier decision in JSON format specified.
"""

REGENERATION_SYSTEM_PROMPT = """You are tasked with regenerating a response based on the reflection provided by the verifier.

The verifier has identified that the previous response deviated from the policy. Your job is to generate a corrected response that addresses the issues raised in the reflection while staying compliant with the policy.

You should maintain the conversation context and provide a response that is natural and follows the policy guidelines."""



def prepare_verifier_content(messages, policy, num_turns_to_keep=3):

    # Convert to ChatMessages first if needed
    chat_messages = ChatMessages.from_prompt_or_messages(messages)

    # Get the underlying messages list and slice it
    messages_list = chat_messages.to_chat_messages()[:-1]
    if len(messages_list) > num_turns_to_keep:
        messages_list = messages_list[-num_turns_to_keep:]
        chat_messages = ChatMessages(messages_list)

    conversation_str = chat_messages.to_prompt()

    # Extract current action - if it has tool calls, only include the first tool call
    last_message = messages_list[-1]
    if last_message.tool_calls and len(last_message.tool_calls) > 0:
        # Create a copy of the message with only the first tool call for verification
        first_tool_call = last_message.tool_calls[0]
        temp_message = ChatMessage(
            role=last_message.role,
            content=last_message.content,
            tool_calls=[first_tool_call]
        )
        # Format the current action to show only the first tool call
        current_action = temp_message.extract_text_content()
        if first_tool_call.get("function"):
            function_info = first_tool_call.get("function", {})
            function_name = function_info.get("name", "unknown")
            function_args = function_info.get("arguments", "{}")
            current_action += f"\n[Tool Call: {function_name}({function_args})]"
    else:
        current_action = last_message.extract_text_content()
    # print(current_action)

    return VERIFIER_USER_PROMPT_TEMPLATE.format(conversation=conversation_str,
                                       current_action=current_action,
                                       policy=policy)



def parse_verifier_response(response_content: str) -> dict:
    # Try to extract JSON from the response
    json_match = re.search(r'```json\n?(.*?)```', response_content, re.DOTALL)

    if not json_match:
        raise ValueError(
            f"Could not find valid JSON in verifier response: \n{response_content}\n\n"
        )

    json_str = json_match.group(1)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in verifier response: {e}") from e

    return data



class VerifierDecision(BaseModel):
    deviation: bool
    reflection: str

    @classmethod
    def from_dict(cls, data: dict) -> "VerifierDecision":
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "deviation": self.deviation,
            "reflection": self.reflection,
        }
class LLMVerifier:
    def __init__(
        self,
        verifier_model: str = "gpt-4o-mini",
        verifier_api_key: str | None = None,
        verifier_base_url: str | None = None,
        regenerator_model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        regenerator_max_tokens: int = 2048,
        enable_logging: bool = True,
        system_prompt: str | None = None,
        log_file: str = "verifier.log",
    ):
        """Initialize the LLM Verifier.

        Args:
            verifier_model: LiteLLM model name for verifier (e.g., "gpt-4o-mini")
            verifier_api_key: API key for the verifier model
            verifier_base_url: Base URL for the verifier model endpoint
            regenerator_model: LiteLLM model name for regenerator (defaults to verifier_model)
            temperature: Temperature for verifier LLM (0.0 for deterministic)
            max_tokens: Maximum tokens for verifier response
            regenerator_max_tokens: Maximum tokens for regenerator response
            enable_logging: Whether to log verifier decisions
            system_prompt: Custom system prompt for verifier (uses default if None)
            log_file: Path to the log file for verifier logs (default: "verifier.log")
        """
        self.enable_logging = enable_logging
        self.verifier_model = verifier_model
        self.regenerator_model = regenerator_model or verifier_model
        self.verifier_api_key = verifier_api_key
        self.verifier_base_url = verifier_base_url
        self.regenerator_max_tokens = regenerator_max_tokens

        # Set up logger with custom log file if logging is enabled
        if enable_logging:
            global verifier_logger
            verifier_logger = setup_verifier_logger(log_file)

        # Use custom system prompt or default
        verifier_system_prompt = system_prompt if system_prompt is not None else VERIFIER_SYSTEM_PROMPT

        # Initialize verifier LLM using LiteLLM
        self.verifier_lm = LiteLLMLanguageModel(
            model_name=verifier_model,
            api_key=verifier_api_key,
            api_base=verifier_base_url,
            system_prompt=verifier_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            is_async=True,
        )


    async def verify(
        self,
        messages: list[ChatMessage] | ChatMessages,
        policy: str|None = None,
        num_turns_to_keep: int = 3
    ) -> VerifierDecision:
        """Verify if the current action complies with the policy.

        Args:
            messages: Conversation history including the current action
            policy: Policy to verify against (uses default if None)
            num_turns_to_keep: Number of recent turns to include in context

        Returns:
            VerifierDecision with deviation flag and reflection
        """
        if policy is None:
            policy = POLICY

        verifier_prompt = prepare_verifier_content(
            messages=messages,
            policy=policy,
            num_turns_to_keep=num_turns_to_keep
        )

        verifier_messages = [ChatMessage(role="user", content=verifier_prompt)]
        if self.enable_logging:
            verifier_logger.info(f"Verifier Input:\n{verifier_prompt}\n")

        try:
            response_tuple = await self.verifier_lm.agenerate(verifier_messages)

            if isinstance(response_tuple, tuple):
                response_message, _ = response_tuple
            else:
                response_message = response_tuple

            response_content = response_message.get("content", "")
            if self.enable_logging:
                verifier_logger.info(f"Verifier Response:\n{response_content}\n")

            verifier_decision_dict = parse_verifier_response(response_content)

            verifier_decision = VerifierDecision.from_dict(verifier_decision_dict)

            if self.enable_logging:
                verifier_logger.info(f"Verification result - Deviation: {verifier_decision.deviation}, Reflection: {verifier_decision.reflection}")

            return verifier_decision
        except Exception as e:
            verifier_logger.error(f"Verification failed: {e}")
            return VerifierDecision(deviation=False, reflection="")

    async def _regenerate_response(
        self,
        conversation_messages: list[ChatMessage],
        reflection: str,
        policy: str,
        tools: list[dict] | None = None,
    ) -> dict:
        """Generate a corrected response based on verifier reflection.

        Args:
            conversation_messages: Original conversation history
            reflection: Verifier's reflection on what went wrong
            policy: Policy to comply with

        Returns:
            Regenerated response message
        """
        # Create regeneration prompt
        regeneration_prompt = f"""Based on the following reflection about policy deviation, generate a corrected response.

Reflection:
{reflection}

Policy:
{policy}

Please provide a response that addresses the issues raised in the reflection while staying compliant with the policy."""

        # Create regenerator LLM with the regeneration system prompt
        regenerator_lm = LiteLLMLanguageModel(
            model_name=self.regenerator_model,
            api_key=self.verifier_api_key,
            api_base=self.verifier_base_url,
            system_prompt=REGENERATION_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=self.regenerator_max_tokens,
            is_async=True,
        )

        # Prepare messages for regeneration - include conversation context
        regeneration_messages = conversation_messages[:-1] + [
            ChatMessage(role="user", content=regeneration_prompt)
        ]
        if self.enable_logging:
            verifier_logger.info(f"Regeneration Messages:\n{regeneration_messages}\n")

        try:
            response_tuple = await regenerator_lm.agenerate(regeneration_messages, tools=tools)

            if isinstance(response_tuple, tuple):
                response_message, _ = response_tuple
            else:
                response_message = response_tuple

            if self.enable_logging:
                verifier_logger.info(f"Regenerated response:\n{response_message}\n")

            return response_message
        except Exception as e:
            verifier_logger.error(f"Regeneration failed: {e}")
            raise

    async def get_verified_response(
        self,
        messages: list[ChatMessage] | ChatMessages,
        policy: str | None = None,
        verification_budget: int = 3,
        num_turns_to_keep: int = 3,
        tools: list[dict] | None = None,
    ) -> tuple[dict, list[VerifierDecision]]:
        """Get a verified response with iterative verification and regeneration.

        This method performs iterative verification and regeneration:
        1. Verify the current response against the policy
        2. If deviation detected and budget remains, regenerate the response
        3. Repeat until no deviation or budget exhausted
        4. Return the final response and verification history

        Args:
            messages: Conversation history including the initial response
            policy: Policy to verify against (uses default if None)
            verification_budget: Maximum number of verification-regeneration iterations
            num_turns_to_keep: Number of recent turns to include in verification context

        Returns:
            Tuple of (final_response_message, verification_history)
        """
        if policy is None:
            policy = POLICY

        # Convert to ChatMessages if needed
        if not isinstance(messages, ChatMessages):
            messages = ChatMessages(messages)

        verification_history = []
        current_messages = messages.to_chat_messages()

        # Store the original message to preserve all tool calls if no deviation
        original_message = current_messages[-1]

        for iteration in range(verification_budget):
            if self.enable_logging:
                verifier_logger.info(f"Verification iteration {iteration + 1}/{verification_budget}")

            # Verify current response
            verifier_decision = await self.verify(
                current_messages,
                policy=policy,
                num_turns_to_keep=num_turns_to_keep
            )
            verification_history.append(verifier_decision)

            # If no deviation, return the ORIGINAL response with all tool calls intact
            if not verifier_decision.deviation:
                if self.enable_logging:
                    verifier_logger.info("No deviation detected - returning original verified response")
                # Return the original message if this is the first iteration, otherwise return current
                return_message = original_message if iteration == 0 else current_messages[-1]
                return return_message.to_dict(), verification_history

            # If deviation detected but budget exhausted, return last response
            if iteration == verification_budget - 1:
                if self.enable_logging:
                    verifier_logger.warning(
                        f"Verification budget exhausted ({verification_budget} iterations). "
                        "Returning last generated response despite deviation."
                    )
                return current_messages[-1].to_dict(), verification_history

            # Regenerate response based on reflection
            if self.enable_logging:
                verifier_logger.info(f"Deviation detected - regenerating response (iteration {iteration + 1})")

            try:
                regenerated_message = await self._regenerate_response(
                    current_messages,
                    verifier_decision.reflection,
                    policy,
                    tools
                )

                # Update current messages with regenerated response
                current_messages = current_messages[:-1] + [ChatMessage(**regenerated_message)]

            except Exception as e:
                verifier_logger.error(f"Regeneration failed: {e}. Returning last response.")
                return current_messages[-1].to_dict(), verification_history

        # Should not reach here, but return last response as fallback
        return current_messages[-1].to_dict(), verification_history


