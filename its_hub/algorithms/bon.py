from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage, ChatMessages
from its_hub.utils import extract_content_from_lm_response


@dataclass
class BestOfNResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    scores: list[float]
    selected_index: int
    usage: dict | None = None
    reward_usage: dict | None = None
    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, orm: AbstractOutcomeRewardModel):
        self.orm = orm

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | BestOfNResult:
        """run inference asynchronously with best-of-n"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # generate responses
        responses = await lm.agenerate(
            chat_messages.to_batch(budget), tools=tools, tool_choice=tool_choice
        )

        # extract content from message dict responses
        response_contents = [extract_content_from_lm_response(r["choices"][0]["message"]) for r in responses]

        # score responses
        # TODO: make batched a configurable parameter or remove non-batched branch
        # Currently hardcoded to True, will be addressed in future PR
        reward_usage = None
        batched = True
        if batched:
            scores, judge_usage = await self.orm.ascore_with_usage(chat_messages, response_contents)
        else:
            scores = []
            for r in response_contents:
                scores, judge_usage = await self.orm.ascore_with_usage(chat_messages, r)
                scores.append(scores)

        # select the best response
        selected_index = scores.index(max(scores))
        
        reward_usage = judge_usage.model_dump()
        print("judge_usage", reward_usage)
        total_usage = {}
        for response in responses:
            if "usage" in response:
                usage = response["usage"]
                for key, value in usage.items():
                    if isinstance(value, dict):
                        if key not in total_usage:
                            total_usage[key] = {}
                        for k, v in value.items():
                            total_usage[key][k] = total_usage[key].get(k, 0) + v
                    else:
                        total_usage[key] = total_usage.get(key, 0) + value
        total_usage = total_usage
        

        # return the result - preserve original message format with tool calls
        result = BestOfNResult(
            responses=responses,  # Keep original dict format with tool calls
            scores=scores,
            selected_index=selected_index,
            usage=total_usage,
            reward_usage=reward_usage,
        )
        return result.the_one if return_response_only else result

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | BestOfNResult:
        """run inference synchronously with best-of-n"""
        import asyncio
        return asyncio.run(
            self.ainfer(lm, prompt_or_messages, budget, return_response_only, tools, tool_choice)
        )
