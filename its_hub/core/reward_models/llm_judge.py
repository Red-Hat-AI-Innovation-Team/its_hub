"""Reward model implementations for production use."""

import json
import logging

from its_hub.api import AbstractLanguageModel, AbstractOutcomeRewardModel


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

    def __init__(
        self,
        lm: AbstractLanguageModel,
        judge_prompt: str | None = None,
        fallback_score: float = 5.0,
    ):
        """
        Initialize LLM judge.

        Args:
            lm: Language model to use for scoring (reuses existing LM abstraction)
            judge_prompt: Custom judge prompt template. Use {conversation} placeholder.
                         If None, uses DEFAULT_JUDGE_PROMPT.
            fallback_score: Score to return if JSON parsing fails (default: 5.0)
        """
        self.lm = lm
        self.judge_prompt = judge_prompt or self.DEFAULT_JUDGE_PROMPT
        self.fallback_score = fallback_score

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format conversation messages as readable text, including tool calls."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format tool calls if present
            tool_calls = msg.get("tool_calls")
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

    def _build_judge_prompt(self, conversation: list[dict]) -> list[dict]:
        """Build judge prompt from conversation."""
        conversation_text = self._format_conversation(conversation)
        prompt_text = self.judge_prompt.format(conversation=conversation_text)
        return [{"role": "user", "content": prompt_text}]

    def _parse_score(self, response_content: str) -> float:
        """Parse score from LLM response with fallback."""
        try:
            # Try to parse JSON
            parsed = json.loads(response_content)
            score = float(parsed.get("score", self.fallback_score))
            return score
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logging.warning(
                f"Failed to parse score from response: {response_content[:100]}. "
                f"Using fallback score {self.fallback_score}. Error: {e}"
            )
            return self.fallback_score

    def score(
        self,
        messages: list[list[dict]] | list[dict],
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
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations asynchronously using LLM.

        Args:
            messages: Single conversation or multiple conversations
            **kwargs: Additional parameters passed to LM (temperature, max_tokens, etc.)

        Returns:
            Single score or list of scores
        """
        # Detect batch vs single
        is_batch = messages and isinstance(messages[0], list)

        # Normalize to batch
        conversations = messages if is_batch else [messages]

        # Build judge prompts for all conversations
        from .types import ChatMessage

        judge_prompts = [
            [
                ChatMessage(
                    role="user", content=self._build_judge_prompt(conv)[0]["content"]
                )
            ]
            for conv in conversations
        ]

        # Leverage LM's async batching!
        responses = await self.lm.agenerate(judge_prompts, **kwargs)

        # Parse scores from responses
        scores = [self._parse_score(r.get("content", "")) for r in responses]

        # Return single or batch
        return scores if is_batch else scores[0]
