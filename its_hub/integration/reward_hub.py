from reward_hub.base import AggregationMethod

from its_hub.base import AbstractProcessRewardModel
from its_hub.types import ChatMessage, ChatMessages


class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(
        self, model_name: str, device: str, aggregation_method: AggregationMethod
    ):
        from reward_hub.vllm.reward import VllmProcessRewardModel

        self.model = VllmProcessRewardModel(model_name=model_name, device=device)
        self.aggregation_method = aggregation_method

    def _build_score_messages(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> tuple[bool, list[list[dict]], list[str]]:
        """Build messages for scoring.

        Args:
            prompt_or_messages: Prompt string or list of chat messages
            response_or_responses: Single response string or list of responses

        Returns:
            Tuple of (is_single_response, messages, responses_list)
        """
        # Convert to uniform ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        is_single_response = isinstance(response_or_responses, str)
        responses = (
            [response_or_responses] if is_single_response else response_or_responses
        )

        # Build conversation messages with responses (convert system to user for reward model compatibility)
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

        return is_single_response, messages, responses

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float:
        is_single_response, messages, _ = self._build_score_messages(
            prompt_or_messages, response_or_responses
        )
        res = self.model.score(
            messages=messages,
            aggregation_method=self.aggregation_method,
            return_full_prm_result=False,
        )
        return res[0] if is_single_response else res

    async def score_async(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """score response(s) asynchronously"""
        import asyncio

        is_single_response, messages, _ = self._build_score_messages(
            prompt_or_messages, response_or_responses
        )

        # Run in thread to avoid blocking event loop
        res = await asyncio.to_thread(
            self.model.score,
            messages=messages,
            aggregation_method=self.aggregation_method,
            return_full_prm_result=False,
        )
        return res[0] if is_single_response else res
