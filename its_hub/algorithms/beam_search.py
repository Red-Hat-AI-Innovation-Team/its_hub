import copy
from dataclasses import field

import numpy as np
from pydantic.dataclasses import dataclass

from its_hub.aggregators import HardcodedAggregator
from its_hub.base import (
    AbstractLanguageModel,
    AbstractProcessRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
    AbstractTrajectoryAggregator,
)
from its_hub.lms import StepGeneration
from its_hub.types import ChatMessage, ChatMessages


@dataclass
class BeamSearchResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    scores: list[float]
    selected_index: int
    steps_used: list[int]

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


@dataclass
class Path:
    steps: list[str]
    is_stopped: bool
    score: float
    step_scores: list[float] = field(default_factory=list)  # Raw PRM scores per step for trajectory aggregation

    def deepcopy(self):
        # create a deep copy of the path object
        return Path(
            steps=copy.deepcopy(self.steps),
            is_stopped=self.is_stopped,
            score=self.score,
            step_scores=copy.deepcopy(self.step_scores),
        )


class BeamSearch(AbstractScalingAlgorithm):
    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        beam_width: int,
        aggregator: AbstractTrajectoryAggregator | None = None,
    ):
        self.sg = sg
        self.prm = prm
        self.beam_width = beam_width
        self.aggregator: AbstractTrajectoryAggregator = aggregator or HardcodedAggregator("prod")

    async def _asearch_one_level(
        self,
        lm: AbstractLanguageModel,
        candidates: list[Path],
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[Path]:
        """search one level asynchronously"""
        is_stopped_in_the_beginning = [c.is_stopped for c in candidates]

        # collect batch inputs
        prompts, steps_so_far = [], []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            prompts.append(prompt)
            steps_so_far.append(c.steps)

        # collect batch outputs
        sg_forward_results = await self.sg.aforward(
            lm, prompts, steps_so_far, tools=tools, tool_choice=tool_choice
        )

        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            next_step, is_stopped = sg_forward_results[i]
            c.steps.append(next_step)
            c.is_stopped = is_stopped
            i += 1

        # collect batch inputs for scoring
        steps_so_far = []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            steps_so_far.append(c.steps)

        # collect batch outputs for scoring
        scores = await self.prm.ascore(
            prompt,
            [
                self.sg._post_process(steps_so_far_per_prompt, stopped=True)
                for steps_so_far_per_prompt in steps_so_far
            ],
        )

        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            c.score = scores[i]
            c.step_scores.append(scores[i])
            i += 1

        return candidates

    def _search_one_level(
        self,
        lm: AbstractLanguageModel,
        candidates: list[Path],
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[Path]:
        """search one level synchronously"""
        import asyncio

        return asyncio.run(
            self._asearch_one_level(lm, candidates, prompt, tools, tool_choice)
        )

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | BeamSearchResult:
        """run inference asynchronously with beam search"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        assert budget % self.beam_width == 0, "budget must be divisible by beam_width"
        assert budget >= self.beam_width, (
            "budget must be greater than or equal to beam_width"
        )

        num_beams = budget // self.beam_width

        candidates = [
            Path(steps=[], is_stopped=False, score=0) for _ in range(num_beams)
        ]

        while not all(c.is_stopped for c in candidates):
            # TODO: Update _asearch_one_level to support native ChatMessages format instead of string conversion
            candidates = await self._asearch_one_level(
                lm,
                candidates,
                chat_messages.to_prompt(),
                tools=tools,
                tool_choice=tool_choice,
            )

            # get the top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[: self.beam_width]

            # duplicate the candidates with the highest score
            new_candidates = []
            for _ in range(num_beams):
                for c in candidates:
                    new_candidates.append(c.deepcopy())
            candidates = new_candidates

        scores = [self.aggregator.aggregate(c.step_scores) for c in candidates]
        steps_used = [len(c.steps) for c in candidates]
        result = BeamSearchResult(
            responses=[
                {
                    "role": "assistant",
                    "content": self.sg._post_process(c.steps, stopped=True),
                }
                for c in candidates
            ],
            scores=scores,
            selected_index=int(np.argmax(scores)),
            steps_used=steps_used,
        )
        return result.the_one if return_response_only else result
