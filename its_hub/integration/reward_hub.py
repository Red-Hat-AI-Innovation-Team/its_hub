from reward_hub.base import AggregationMethod

from its_hub.base import AbstractProcessRewardModel, AbstractOutcomeRewardModel
from its_hub.types import ChatMessage, ChatMessages

import litellm
from typing import Tuple


class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(
        self, model_name: str, device: str, aggregation_method: AggregationMethod
    ):
        from reward_hub.vllm.reward import VllmProcessRewardModel

        self.model = VllmProcessRewardModel(model_name=model_name, device=device)
        self.aggregation_method = aggregation_method

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """score response(s) asynchronously"""
        import asyncio
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        is_single_response = isinstance(response_or_responses, str)
        responses = (
            [response_or_responses] if is_single_response else response_or_responses
        )

        # Build conversation messages with responses
        base_msgs = [
            ChatMessage(role="user", content=f"System: {msg.content}")
            if msg.role == "system"
            else msg
            for msg in chat_messages.to_chat_messages()
        ]
        messages = [
            [
                *[{"role": msg.role, "content": msg.content} for msg in base_msgs],
                {"role": "assistant", "content": response},
            ]
            for response in responses
        ]

        # Run in thread to avoid blocking event loop
        res = await asyncio.to_thread(
            self.model.score,
            messages=messages,
            aggregation_method=self.aggregation_method,
            return_full_prm_result=False,
        )
        return res[0] if is_single_response else res

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """score response(s) synchronously"""
        import asyncio
        return asyncio.run(self.ascore(prompt_or_messages, response_or_responses))


class LLMJudgeRewardModel(AbstractOutcomeRewardModel):
    """
    Adapter for reward_hub's LLM Judge models to work with its_hub's AbstractOutcomeRewardModel interface.

    This class wraps reward_hub's PointwiseJudgeModel to make it compatible with its_hub's
    prompt/response format and can be used with algorithms like Best-of-N.
    """

    def __init__(
        self,
        model: str,
        criterion: str = "overall_quality",
        api_key: str | None = None,
        base_url: str | None = None,
        judge_type: str = "pointwise",
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_n: int | None = None,
        **litellm_kwargs,
    ):
        """
        Initialize LLM Judge reward model.

        Args:
            model: LiteLLM model name (e.g., "gpt-4o-mini", "claude-3-sonnet-20240229")
            criterion: Evaluation criterion from CriterionRegistry (default: "overall_quality")
                      Built-in options: overall_quality, writing_quality, technical_quality,
                      relevance_quality, tool-judge
            api_key: API key for the model provider
            base_url: Base URL for custom endpoints
            temperature: Temperature for judge generation (0.0 for deterministic)
            max_tokens: Maximum tokens for judge response
            **litellm_kwargs: Additional arguments passed to LiteLLM
        """
        from reward_hub import AutoJudge

        if top_n:
            self.top_n = top_n
        else:
            self.top_n = None

        self.judge = AutoJudge.from_litellm(
            model=model,
            judge_type=judge_type,
            criterion=criterion,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_n=top_n,
            **litellm_kwargs,
        )
        self.criterion = criterion
        self.model = model
        
    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response: str | list[str],
    ) -> float | list[float]:
        """
        Score response(s) using the LLM judge.

        Args:
            prompt_or_messages: The prompt or conversation context
            response: The response(s) to evaluate (single string or list of strings)

        Returns:
            Score from 0.0 to 1.0 (normalized from judge's 0-10 scale)
            Returns float for single response, list[float] for multiple responses
        """
        # Use async version
        import asyncio
        return asyncio.run(self.ascore(prompt_or_messages, response))

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response: str | list[str],
    ) -> float | list[float]:
        """
        Score response(s) asynchronously using the LLM judge.

        Args:
            prompt_or_messages: The prompt or conversation context
            response: The response(s) to evaluate (single string or list of strings)

        Returns:
            Score from 0.0 to 1.0 (normalized from judge's 0-10 scale)
            Returns float for single response, list[float] for multiple responses
        """
        # Convert to ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # Build base conversation in OpenAI format
        base_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_messages.to_chat_messages()
        ]

        # Handle both single response and batch of responses
        is_single_response = isinstance(response, str)
        responses = [response] if is_single_response else response

        # Build complete conversations (base + each response)
        conversations = [
            base_messages + [{"role": "assistant", "content": resp}]
            for resp in responses
        ]

        # Call judge with multiple conversations
        # Judge expects List[List[dict]] for multiple conversations
        raw_scores = await self.judge.ascore(conversations)

        # Normalize scores to 0-1 range
        if is_single_response:
            # Judge returns float for single conversation
            return raw_scores / 10.0
        else:
            # Judge returns List[float] for multiple conversations
            return [score / 10.0 for score in raw_scores]
        
    async def ascore_with_usage(self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response: str | list[str],
        custom_criterion: str = None,
    ) -> Tuple[float | list[float], litellm.utils.Usage]:
        """
        Async version of score with usage
        
        Score response(s) asynchronously using the LLM judge.

        Args:
            prompt_or_messages: The prompt or conversation context
            response: The response(s) to evaluate (single string or list of strings)

        Returns:
            Score from 0.0 to 1.0 (normalized from judge's 0-10 scale)
            Returns float for single response, list[float] for multiple responses
        """
        # Convert to ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # Build base conversation in OpenAI format
        base_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_messages.to_chat_messages()
        ]

        # Handle both single response and batch of responses
        is_single_response = isinstance(response, str)
        responses = [response] if is_single_response else response

        # Build complete conversations (base + each response)
        conversations = [
            base_messages + [{"role": "assistant", "content": resp}]
            for resp in responses
        ]

        # Call judge with multiple conversations
        # Judge expects List[List[dict]] for multiple conversations
        raw_scores, usage = await self.judge.ascore_with_usage(conversations, top_n=self.top_n, custom_criterion=custom_criterion)

        # Normalize scores to 0-1 range
        if is_single_response:
            # Judge returns float for single conversation
            return raw_scores / 10.0, usage
        else:
            # Judge returns List[float] for multiple conversations
            return [score / 10.0 for score in raw_scores], usage
    
    
    
    