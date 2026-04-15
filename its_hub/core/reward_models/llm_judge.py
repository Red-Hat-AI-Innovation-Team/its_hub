"""Reward model implementations for production use."""

import json
import logging
import re
from typing import ClassVar

from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractOutcomeRewardModel,
    ChatMessage,
    ChatMessages,
)
from its_hub.core.orchestrator import LMOrchestrator


class LLMJudge(AbstractOutcomeRewardModel):
    """
    LLM-based judge that scores conversations using generative reward.

    Reuses AbstractLanguageModel for API communication (retry, error handling, batching).
    Scores are generated via LLM prompting and parsed from structured JSON output.
    """

    DEFAULT_JUDGE_PROMPT = """Score the following conversation on a scale of 0-10.
Return only a JSON object with your score.

Conversation:
{conversation}

Format: {{"score": <number>}}"""

    SCORE_RESPONSE_FORMAT: ClassVar[dict] = {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["score", "reasoning"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        lm: AbstractLanguageModel,
        judge_prompt: str | None = None,
        fallback_score: float = 5.0,
        response_format: dict | None = SCORE_RESPONSE_FORMAT,
    ):
        """
        Initialize LLM judge.

        Args:
            lm: Language model to use for scoring (reuses existing LM abstraction)
            judge_prompt: Custom judge prompt template. Use {conversation} placeholder.
                         If None, uses DEFAULT_JUDGE_PROMPT.
            fallback_score: Score to return if JSON parsing fails (default: 5.0)
            response_format: Response format for structured outputs. Defaults to
                SCORE_RESPONSE_FORMAT which enforces a {"score": <number>, "reasoning": "..."}
                schema. Pass None to disable structured outputs (relies on prompt-based
                JSON extraction only). Pass a custom dict to use your own schema.
        """
        self.lm = lm
        self.judge_prompt = judge_prompt or self.DEFAULT_JUDGE_PROMPT
        self.fallback_score = fallback_score
        self.response_format = response_format

    def _format_conversation(self, messages: list[ChatMessage]) -> str:
        """Format conversation messages as readable text, including tool calls."""
        lines = []
        for msg in messages:
            role = msg.role
            content = msg.content or ""

            # Format tool calls if present
            tool_calls = msg.tool_calls
            if tool_calls:
                tool_strs = []
                for tc in tool_calls:
                    if isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "{}")
                        tool_strs.append(f"{func_name}({func_args})")
                if tool_strs:
                    lines.append(f"{role} [tool calls]: {', '.join(tool_strs)}")
                if content:  # Also include content if present
                    lines.append(f"{role}: {content}")
            else:
                # Regular message with content only
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_judge_prompt(self, conversation: list[ChatMessage]) -> list[ChatMessage]:
        """Build judge prompt from conversation."""
        conversation_text = self._format_conversation(conversation)
        prompt_text = self.judge_prompt.format(conversation=conversation_text)
        return [ChatMessage(role="user", content=prompt_text)]

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON object from LLM response text.

        Handles common LLM response patterns:
        - Raw JSON: {"score": 7}
        - Markdown code blocks: ```json\n{"score": 7}\n```
        - JSON embedded in surrounding text
        """
        text = text.strip()

        # 1. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try extracting from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Try finding a JSON object in the text
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _parse_score(self, response_content: str) -> float:
        """Parse score from LLM response with fallback.

        Tries full JSON parsing first, then falls back to regex extraction
        for truncated JSON (common with models that pad output with whitespace
        or repeating characters).
        """
        parsed = self._extract_json(response_content)
        if parsed is not None:
            try:
                return float(parsed.get("score", self.fallback_score))
            except (ValueError, TypeError, AttributeError) as e:
                logging.warning(
                    f"JSON parsed but 'score' field invalid: {parsed}. Error: {e}"
                )
                return self.fallback_score

        # Fallback: extract score directly from truncated/malformed JSON
        # Handles cases like: {"score": 7.5\n\n\n... (no closing brace)
        match = re.search(r'"score"\s*:\s*([\d.]+)', response_content)
        if match:
            try:
                score = float(match.group(1))
                logging.info(
                    f"Extracted score {score} from truncated JSON via regex fallback."
                )
                return score
            except ValueError:
                pass

        logging.warning(
            f"Failed to extract score from response: {response_content[:200]}. "
            f"Using fallback score {self.fallback_score}."
        )
        return self.fallback_score

    def score(
        self,
        messages: list[ChatMessage] | ChatMessages,
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations synchronously.

        Not implemented - LLMJudge requires async for API calls.
        Use ascore() instead.
        """
        raise NotImplementedError(
            "LLMJudge requires async API calls. Use ascore() instead of score()."
        )

    async def ascore(
        self,
        messages: list[ChatMessage] | ChatMessages,
        orchestrator: AbstractOrchestrator | None = None,
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations asynchronously using LLM.

        Args:
            messages: Single conversation or multiple conversations
            orchestrator: Orchestrator that manages parallel calls to LM
            **kwargs: Additional parameters passed to LM (temperature, max_tokens, etc.)

        Returns:
            Single score or list of scores
        """
        # Detect batch vs single
        is_batch = messages and isinstance(messages[0], list)

        # Normalize to batch
        conversations = messages if is_batch else [messages]

        # Build judge prompts for all conversations
        judge_prompts = [self._build_judge_prompt(conv) for conv in conversations]

        if orchestrator is None:
            # Fallback to default implementation
            orchestrator = LMOrchestrator()

        # Use structured outputs if configured, but allow kwargs to override
        if "response_format" not in kwargs and self.response_format is not None:
            kwargs["response_format"] = self.response_format

        responses = await orchestrator.agenerate(self.lm, judge_prompts, **kwargs)

        # Parse scores from responses
        scores = [self._parse_score(r.get("content") or "") for r in responses]

        # Return single or batch
        return scores if is_batch else scores[0]
