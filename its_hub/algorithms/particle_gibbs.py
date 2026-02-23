import copy
import logging
import random
from enum import Enum

import numpy as np
from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractProcessRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.lms import StepGeneration
from its_hub.types import ChatMessage, ChatMessages


@dataclass
class ParticleGibbsResult(AbstractScalingResult):
    responses_lst: list[list[dict]]  # Keep original message format with tool calls
    log_weights_lst: list[list[float]]
    ref_indices_lst: list[list[int]]
    selected_index: int
    steps_used_lst: list[list[int]]

    @property
    def the_one(self) -> dict:
        return self.responses_lst[-1][self.selected_index]


@dataclass
class ParticleFilteringResult(AbstractScalingResult):
    responses: list[dict]  # Keep original message format with tool calls
    log_weights_lst: list[float]
    selected_index: int
    steps_used_lst: list[int]

    @property
    def the_one(self) -> dict:
        return self.responses[self.selected_index]


@dataclass
class Particle:
    steps: list[str]
    is_stopped: bool
    partial_log_weights: list[float]  # Store aggregated log weights until each step

    @property
    def log_weight(self) -> float:
        """Return the most recent log weight."""
        if self.partial_log_weights:
            return self.partial_log_weights[-1]
        return 0.0

    def deepcopy(self):
        # create a deep copy of the particle object
        return Particle(
            steps=copy.deepcopy(self.steps),
            is_stopped=self.is_stopped,
            partial_log_weights=copy.deepcopy(self.partial_log_weights),
        )


def _inv_sigmoid(x):
    assert 0 <= x <= 1, "x must be between 0 and 1"
    # clip values to avoid numerical issues when x is close to 0 or 1
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def _softmax(x):
    # shift x by the maximum value for numerical stability
    x_shifted = x - np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))


class SelectionMethod(Enum):
    SAMPLE = "sample"
    ARGMAX = "argmax"


class ResamplingMethod(Enum):
    SYSTEMATIC = "systematic"
    MULTINOMIAL = "multinomial"


class TemperatureMethod(Enum):
    ESS = "ess"
    ENTROPY = "entropy"
    BASE = "base"


class ParticleGibbs(AbstractScalingAlgorithm):
    """
    Particle-based Monte Carlo methods for inference time scaling.
    It supports the following variants:
    - Particle Filtering (PF): num_iterations = 1
    - Entropic Particle Filtering (ePF): num_iterations = 1 and does_entropic_annealing = True
    - Particle Gibbs (PG): num_iterations > 1
    - PG with ancestor sampling (PGAS): num_iterations > 1 and does_ancestor_sampling = True
    """

    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        num_iterations: int = 1,
        final_response_selection: str | SelectionMethod = SelectionMethod.ARGMAX,
        num_ref_particles: int = 1,
        does_ancestor_sampling: bool = False,
        does_entropic_annealing: bool = False,
        does_lookahead_modulation: bool = False,
        ess_threshold: float = 0.5,
        early_phase: float = 0.5,
        resampling_method: str | ResamplingMethod = ResamplingMethod.MULTINOMIAL,
        temperature_method: str | TemperatureMethod = TemperatureMethod.ESS,
    ):
        if isinstance(final_response_selection, str):
            final_response_selection = SelectionMethod(final_response_selection)

        if isinstance(resampling_method, str):
            resampling_method = ResamplingMethod(resampling_method)

        if isinstance(temperature_method, str):
            temperature_method = TemperatureMethod(temperature_method)

        self.sg = sg
        self.prm = prm
        self.num_iterations = num_iterations
        self.final_response_selection = final_response_selection
        self.num_ref_particles = num_ref_particles
        self.does_ancestor_sampling = does_ancestor_sampling
        self.does_entropic_annealing = does_entropic_annealing
        self.does_lookahead_modulation = does_lookahead_modulation
        self.ess_threshold = ess_threshold
        self.max_steps = self.sg.max_steps
        self.early_phase = early_phase
        self.resampling_method = resampling_method
        self.temperature_method = temperature_method

    async def _apropagate(
        self,
        lm: AbstractLanguageModel,
        particles: list[Particle],
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[Particle]:
        """propagate particles asynchronously (batched)"""
        is_stopped_in_the_beginning = [p.is_stopped for p in particles]

        # collect batch inputs
        prompts, steps_so_far = [], []
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            prompts.append(prompt)
            steps_so_far.append(p.steps)

        # collect batch outputs
        sg_forward_results = await self.sg.aforward(
            lm, prompts, steps_so_far, tools=tools, tool_choice=tool_choice
        )

        # update particles
        i = 0
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            next_step, is_stopped = sg_forward_results[i]
            p.steps.append(next_step)
            p.is_stopped = is_stopped
            i += 1

        # collect batch inputs for scoring
        steps_so_far = []
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            steps_so_far.append(p.steps)

        # collect batch outputs for scoring
        scores = await self.prm.ascore(
            prompt,
            [
                self.sg._post_process(steps_so_far_per_prompt, stopped=True)
                for steps_so_far_per_prompt in steps_so_far
            ],
        )

        # update particles
        i = 0
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            p.partial_log_weights.append(_inv_sigmoid(scores[i]))
            i += 1

        return particles

    def _entropy_n(self, probabilities: list[float]) -> float:
        """Compute normalized entropy for particles at step t.

        Required for entropic annealing with entropy-based temperature.

        Args:
            probabilities: list of weight probabilities at step t

        Returns:
            normalized entropy of the particles
        """
        # p should be a probability distribution (e.g., output of softmax)
        p = np.asarray(probabilities)
        # Avoid log(0) by clipping probabilities
        p = np.clip(p, 1e-7, 1 - 1e-7)
        entropy = -p * np.log(p)
        entropy = np.sum(entropy)

        # Make normalization robust to edge cases (e.g., len(p) <= 1)
        if len(p) > 2 and np.log(len(p)) > 0:
            entropy_normalized = entropy / np.log(len(p))
        else:
            entropy_normalized = entropy
        return entropy_normalized

    def _effective_sample_size(self, probabilities: list[float]) -> float:
        """Compute effective sample size for particles at step t.

        Required for entropic annealing with ESS-based temperature and activating entropic annealing.
        ESS and ESS ratio are easier to interpret than entropy.

        Args:
            probabilities: list of weight probabilities at step t

        Returns:
            effective sample size of the particles at step t
        """
        p = np.asarray(probabilities)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        effective_sample_size = 1 / np.sum(p**2)
        return effective_sample_size

    def _temperature_base(
        self,
        value_max: float,
        progress: float,
    ) -> float:
        value = value_max - progress
        temperature = max(1.0, value)
        return temperature

    def _temperature_entropy(
        self,
        entropy_n: float,
        progress: float,
    ) -> float:
        beta = entropy_n + (1 - entropy_n) * (progress)
        value = 1 / beta
        temperature = max(1.0, value)
        return temperature

    def _temperature_ess(self, ess_ratio: float, progress: float) -> float:
        """Compute temperature for entropic annealing based on effective sample size ratio.

        Args:
            ess_ratio: effective sample size ratio (ESS / num_particles)
            progress: sampling progress (t/T_max)

        Returns:
            temperature for entropic annealing
        """
        if ess_ratio <= 0:
            return 1.0
        value = 1.0 / ess_ratio * (1 - progress)
        temperature = max(1.0, value)
        return temperature

    def _temperature_annealing(
        self,
        probabilities: list[float],
        current_step: int,
        num_particles: int,
        value_max: float = 2.0,
    ) -> float:
        if num_particles <= 1:
            return 1.0

        progress = current_step / self.max_steps

        entropy_n = self._entropy_n(probabilities)
        ess = self._effective_sample_size(probabilities)
        ess_ratio = ess / num_particles

        # use 1.0 as default temperature if ess_ratio is not less than ess_threshold
        temperature = 1.0
        if ess_ratio < self.ess_threshold and progress < self.early_phase:
            match self.temperature_method:
                case TemperatureMethod.ESS:
                    temperature = self._temperature_ess(ess_ratio, progress)
                case TemperatureMethod.ENTROPY:
                    temperature = self._temperature_entropy(entropy_n, progress)
                case TemperatureMethod.BASE:
                    temperature = self._temperature_base(value_max, progress)
        return temperature

    def _resampling_systematic(
        self, particles: list[Particle], probabilities: np.ndarray, num_particles: int
    ) -> list[Particle]:
        """Perform systematic resampling from the given normalized weights.

        This is a way to sample particles with replacement and preserve diversity.
        Any particle with a weights > 1/N should be resampled.

        Args:
            particles: list of particles
            probabilities: array of particle weights (must sum to 1)
            num_particles: number of particles to resample

        Returns:
            list of resampled particles
        """
        positions = (np.arange(num_particles) + np.random.uniform(0, 1)) / num_particles

        indices = np.zeros(num_particles, dtype=int)
        cumulative_sum = np.cumsum(probabilities)
        i, j = 0, 0

        while i < num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        resampled_particles = [particles[i] for i in indices]
        return resampled_particles

    def _resampling_multinomial(
        self, particles: list[Particle], probabilities: list[float], num_particles: int
    ) -> list[Particle]:
        return random.choices(particles, weights=probabilities, k=num_particles)

    def _resampling(
        self, particles: list[Particle], probabilities: list[float], num_particles: int
    ) -> list[Particle]:
        if self.resampling_method == ResamplingMethod.SYSTEMATIC:
            return self._resampling_systematic(particles, probabilities, num_particles)
        elif self.resampling_method == ResamplingMethod.MULTINOMIAL:
            return self._resampling_multinomial(particles, probabilities, num_particles)
        else:
            raise ValueError(f"Invalid resampling method: {self.resampling_method}")

    def _propagate(
        self,
        lm: AbstractLanguageModel,
        particles: list[Particle],
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[Particle]:
        """propagate particles synchronously"""
        import asyncio

        return asyncio.run(self._apropagate(lm, particles, prompt, tools, tool_choice))

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> dict | ParticleGibbsResult:
        """run inference asynchronously with particle gibbs"""
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        assert budget % self.num_iterations == 0, (
            "budget must be divisible by num_iterations"
        )

        if response_format is not None:
            logging.warning(
                "response_format is ignored for particle-based algorithms using step-wise generation"
            )

        num_particles = budget // self.num_iterations

        ref_particles = []
        responses_lst = []
        log_weights_lst = []
        ref_indices_lst = []
        steps_used_lst = []

        for _ in range(self.num_iterations):
            num_free_particles = num_particles - len(ref_particles)

            particles = [
                Particle(steps=[], is_stopped=False, partial_log_weights=[])
                for _ in range(num_free_particles)
            ] + ref_particles

            current_step = 0

            while not all(p.is_stopped for p in particles):
                # TODO: Update _apropagate to support native ChatMessages format instead of string conversion
                particles = await self._apropagate(
                    lm,
                    particles,
                    chat_messages.to_prompt(),
                    tools=tools,
                    tool_choice=tool_choice,
                )

                current_step += 1  # Increment after propagation

                # resampling (free) particles
                # Use partial log weights at the current step for fair comparison
                log_weights = []
                for p in particles:
                    if p.is_stopped:
                        # Stopped particles use their final weight
                        log_weights.append(p.log_weight)
                    else:
                        # Non-stopped particles use weight at current step
                        log_weights.append(p.partial_log_weights[current_step - 1])

                probabilities = _softmax(log_weights)

                # entropic annealing
                if self.does_entropic_annealing:
                    temperature = self._temperature_annealing(
                        probabilities, current_step, num_free_particles
                    )
                    # apply temperature annealing to the log weights
                    log_weights = np.array(log_weights)
                    probabilities = _softmax(log_weights * (1 / temperature))

                # resampling (by default multinomial)
                resampled_particles = self._resampling(
                    particles, probabilities, num_free_particles
                )

                if self.does_ancestor_sampling:
                    raise NotImplementedError("Ancestor sampling is not implemented")

                if self.does_lookahead_modulation:
                    raise NotImplementedError("Lookahead modulation is not implemented")

                # duplicate the resampled particles
                resampled_particles = [p.deepcopy() for p in resampled_particles]

                # Truncate reference particles if they were resampled as non-reference
                # This ensures all particles have the same number of steps
                for p in resampled_particles:
                    if len(p.steps) > current_step:
                        # This particle was a reference particle that got resampled
                        p.steps = p.steps[:current_step]
                        p.partial_log_weights = p.partial_log_weights[:current_step]
                        p.is_stopped = False

                # add reference particles
                particles = resampled_particles + ref_particles

            # select the reference particles
            log_weights = [p.log_weight for p in particles]
            probabilities = _softmax(log_weights)
            ref_indices = random.choices(
                range(len(particles)), weights=probabilities, k=self.num_ref_particles
            )
            ref_particles = [particles[i] for i in ref_indices]

            responses_lst.append(
                [
                    {
                        "role": "assistant",
                        "content": self.sg._post_process(p.steps, stopped=True),
                    }
                    for p in particles
                ]
            )
            log_weights_lst.append(log_weights)
            ref_indices_lst.append(ref_indices)
            steps_used_lst.append([len(p.steps) for p in particles])

        # select the chosen particle based on final response selection method
        # log_weights and probabilities are from the last iteration
        match self.final_response_selection:
            case SelectionMethod.SAMPLE:
                selected_index = random.choices(
                    range(len(particles)), weights=probabilities, k=1
                )[0]
            case SelectionMethod.ARGMAX:
                selected_index = np.argmax(log_weights).item()

        result = ParticleGibbsResult(
            responses_lst=responses_lst,
            log_weights_lst=log_weights_lst,
            ref_indices_lst=ref_indices_lst,
            selected_index=selected_index,
            steps_used_lst=steps_used_lst,
        )

        return result.the_one if return_response_only else result


class ParticleFiltering(ParticleGibbs):
    """
    Particle filtering being a special case of particle Gibbs with num_iterations=1
    """

    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        final_response_selection: str | SelectionMethod = SelectionMethod.ARGMAX,
        resampling_method: str | ResamplingMethod = ResamplingMethod.MULTINOMIAL,
    ):
        # initialize with num_iterations=1
        super().__init__(
            sg=sg,
            prm=prm,
            num_iterations=1,
            final_response_selection=final_response_selection,
            num_ref_particles=0,
            does_ancestor_sampling=False,
            does_entropic_annealing=False,
            does_lookahead_modulation=False,
        )

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> dict | ParticleFilteringResult:
        """run inference asynchronously with particle filtering"""
        result = await super().ainfer(
            lm,
            prompt_or_messages,
            budget,
            return_response_only=False,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        # Flatten the single-iteration result
        flattened_result = ParticleFilteringResult(
            responses=result.responses_lst[0],
            log_weights_lst=result.log_weights_lst[0],
            selected_index=result.selected_index,
            steps_used_lst=result.steps_used_lst[0],
        )

        if return_response_only:
            return flattened_result.the_one

        return flattened_result


class EntropicParticleFiltering(ParticleGibbs):
    """Entropic particle filtering with Entropic Annealing"""

    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        final_response_selection: str | SelectionMethod = SelectionMethod.ARGMAX,
        resampling_method: str | ResamplingMethod = ResamplingMethod.SYSTEMATIC,
        temperature_method: str | TemperatureMethod = TemperatureMethod.ESS,
        ess_threshold: float = 0.5,
        early_phase: float = 0.5,
    ):
        # initialize with num_iterations=1
        super().__init__(
            sg=sg,
            prm=prm,
            num_iterations=1,
            final_response_selection=final_response_selection,
            num_ref_particles=0,
            does_ancestor_sampling=False,
            does_entropic_annealing=True,
            does_lookahead_modulation=False,
            ess_threshold=ess_threshold,
            early_phase=early_phase,
            resampling_method=resampling_method,
            temperature_method=temperature_method,
        )

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | ParticleFilteringResult:
        result = super().infer(
            lm,
            prompt_or_messages,
            budget,
            return_response_only=False,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Flatten the single-iteration result
        flattened_result = ParticleFilteringResult(
            responses=result.responses_lst[0],
            log_weights_lst=result.log_weights_lst[0],
            selected_index=result.selected_index,
            steps_used_lst=result.steps_used_lst[0],
        )

        if return_response_only:
            return flattened_result.the_one

        return flattened_result
