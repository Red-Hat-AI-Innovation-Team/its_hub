"""
Process reward model implementation for experimental algorithms. This requires installing reward-hub:
    pip install its_hub[experimental]
"""

import asyncio

from reward_hub.base import AggregationMethod
from reward_hub.vllm.reward import VllmProcessRewardModel

from its_hub.api import AbstractProcessRewardModel, ChatMessage, ChatMessages


class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    """
    Process reward model using reward-hub's vLLM implementation.

    This provides step-by-step scoring for algorithms like BeamSearch and ParticleFiltering.
    Requires installing: pip install its_hub[experimental]
    """

    def __init__(
        self, model_name: str, device: str, aggregation_method: AggregationMethod
    ):
        """
        Initialize the process reward model.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-Math-PRM-7B")
            device: Device to run on (e.g., "cuda:0", "cpu")
            aggregation_method: Method to aggregate step scores (from reward_hub.base.AggregationMethod)
        """
        self.model = VllmProcessRewardModel(model_name=model_name, device=device)
        self.aggregation_method = aggregation_method

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """Score response(s) asynchronously."""
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
                *[
                    {"role": msg.role, "content": msg.extract_text_content()}
                    for msg in base_msgs
                ],
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
        """Score response(s) synchronously."""
        return asyncio.run(self.ascore(prompt_or_messages, response_or_responses))
