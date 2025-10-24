from reward_hub.base import AggregationMethod
from reward_hub.llm_judge import create_pointwise_judge, create_groupwise_judge

from its_hub.base import AbstractProcessRewardModel, AbstractOutcomeRewardModel
from its_hub.types import ChatMessage, ChatMessages
from typing import Optional
import logging
import json

logger = logging.getLogger(__name__)


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
            ChatMessage(role="user", content=f"System: {msg.extract_text_content()}")
            if msg.role == "system"
            else msg
            for msg in chat_messages.to_chat_messages()
        ]
        messages = [
            [
                *[{"role": msg.role, "content": msg.extract_text_content()} for msg in base_msgs],
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
        criterion: str,
        judge_type: str = "groupwise",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_judge_logging: bool = True,
        top_n: int = 1,
        **litellm_kwargs,
    ):
        """
        Initialize LLM Judge reward model.

        Args:
            model: LiteLLM model name (e.g., "gpt-4o-mini", "claude-3-sonnet-20240229")
            criterion: Evaluation criterion from CriterionRegistry (default: "overall_quality")
                      Built-in options: overall_quality, writing_quality, technical_quality,
                      relevance_quality, tool-judge
            judge_type: Type of judge - "pointwise" or "groupwise" (default: "groupwise")
            api_key: API key for the model provider
            base_url: Base URL for custom endpoints
            temperature: Temperature for judge generation (0.0 for deterministic)
            max_tokens: Maximum tokens for judge response
            enable_judge_logging: If True, log judge scores and reasoning (default: True)
            top_n: For groupwise judges, number of top responses to select (default: 1)
            **litellm_kwargs: Additional arguments passed to LiteLLM
        """

        if judge_type == "pointwise":
            self.judge = create_pointwise_judge(
                model=model,
                criterion=criterion,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                **litellm_kwargs,
            )
        elif judge_type == "groupwise":
            self.judge = create_groupwise_judge(
                model=model,
                criterion=criterion,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                **litellm_kwargs,
            )
        else:
            raise ValueError(
                f"Invalid judge type: {judge_type}. Must be 'pointwise' or 'groupwise'."
            )

        self.judge_type = judge_type
        self.criterion = criterion
        self.model = model
        self.top_n = top_n
        self.enable_judge_logging = enable_judge_logging

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
            - For single response: float score
            - For multiple responses: list[float] scores

        Note:
            - If enable_judge_logging=True, the judge's reasoning is logged internally
              along with the scores from the JudgeResult object.
            - For pointwise judges: logs individual scores and reasoning per response
            - For groupwise judges: logs ranking reasoning and top-N selection with binary scores
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
            - For single response: float score
            - For multiple responses: list[float] scores

        Note:
            - If enable_judge_logging=True, the judge's reasoning is logged internally
              along with the scores from the JudgeResult object.
            - For pointwise judges: logs individual scores and reasoning per response
            - For groupwise judges: logs ranking reasoning and top-N selection with binary scores
        """
        # Convert to ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # Build base conversation in OpenAI format
        base_messages = [
            {"role": msg.role, "content": msg.extract_text_content()} # Reward Hub expects content to be a string
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

        if self.judge_type == "groupwise":
            judge_result = await self.judge.ascore(
                conversations,
                return_judge_reasoning=self.enable_judge_logging,
                top_n=self.top_n,
            )
        else:
            judge_result = await self.judge.ascore(
                conversations, return_judge_reasoning=self.enable_judge_logging
            )

        # Log judge results if enabled
        if self.enable_judge_logging and judge_result.reasonings:
            if self.judge_type == "pointwise":
                # Pointwise: log each response's individual score and reasoning
                for i, (score, reasoning, response) in enumerate(
                    zip(judge_result.scores, judge_result.reasonings, responses)
                ):
                    extra_data = {
                        "judge_type": "pointwise",
                        "response_index": i,
                        "score": score,
                        "reasoning": reasoning,
                        "response_preview": response[:300] + "..."
                        if len(response) > 300
                        else response,
                        "criterion": self.criterion,
                        "model": self.model,
                    }
                    logger.info(
                        f"Pointwise Judge Result for response {i}:\n{json.dumps(extra_data, indent=2)}"
                    )
            else:
                # Groupwise: log ranking reasoning and which responses were selected as top-N
                # Binary scores: 1.0 for top-N, 0.0 for others
                top_indices = [
                    i for i, score in enumerate(judge_result.scores) if score == 1.0
                ]
                response_previews = [
                    {
                        "index": i,
                        "score": score,
                        "preview": resp[:300] + "..." if len(resp) > 300 else resp,
                    }
                    for i, (score, resp) in enumerate(
                        zip(judge_result.scores, responses)
                    )
                ]
                extra_data = {
                    "judge_type": "groupwise",
                    "top_n": self.top_n,
                    "top_indices": top_indices,
                    "scores": judge_result.scores,
                    "response_previews": response_previews,
                    "ranking_reasoning": judge_result.reasonings[0]
                    if judge_result.reasonings
                    else None,
                    "criterion": self.criterion,
                    "model": self.model,
                }
                logger.info(
                    f"Groupwise Judge Result: selected {len(top_indices)} of {len(responses)} responses\n{json.dumps(extra_data, indent=2, default=str)}"
                )

        # Return only scores (single float if single response, list otherwise)
        if is_single_response:
            return judge_result.scores[0]
        return judge_result.scores
