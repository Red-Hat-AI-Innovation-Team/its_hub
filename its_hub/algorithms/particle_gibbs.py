import copy
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


@dataclass
class ParticleGibbsResult(AbstractScalingResult):
    responses_lst: list[list[str]]
    log_weights_lst: list[list[float]]
    ref_indices_lst: list[list[int]]
    selected_index: int
    steps_used_lst: list[list[int]]

    @property
    def the_one(self) -> str:
        return self.responses_lst[-1][self.selected_index]


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


class ParticleGibbs(AbstractScalingAlgorithm):
    """
    Particle-based Monte Carlo methods for inference time scaling.
    It supports the following variants:
    - Particle Filtering (PF): num_iterations = 1
    - Particle Gibbs (PG): num_iterations > 1
    - PG with ancestor sampling (PGAS): num_iterations > 1 and does_ancestor_sampling = True
    """

    def __init__(
        self,
        sg: StepGeneration,
        prm: AbstractProcessRewardModel,
        num_iterations: int = 1,
        selection_method: str | SelectionMethod = SelectionMethod.ARGMAX,
        num_ref_particles: int = 1,
        does_ancestor_sampling: bool = False,
    ):
        if isinstance(selection_method, str):
            selection_method = SelectionMethod(selection_method)

        self.sg = sg
        self.prm = prm
        self.num_iterations = num_iterations
        self.selection_method = selection_method
        self.num_ref_particles = num_ref_particles
        self.does_ancestor_sampling = does_ancestor_sampling

    def _propagate(
        self,
        lm: AbstractLanguageModel,
        particles: list[Particle],
        prompt: str,
        batched: bool = False,
    ) -> list[Particle]:
        if not batched:
            # forward each particle
            for p in particles:
                if p.is_stopped:
                    continue

                next_step, is_stopped = self.sg.forward(lm, prompt, p.steps)
                p.steps.append(next_step)
                p.is_stopped = is_stopped
                score = self.prm.score(
                    prompt, self.sg._post_process(p.steps, stopped=True)
                )
                p.partial_log_weights.append(_inv_sigmoid(score))

            return particles

        is_stopped_in_the_beginning = [p.is_stopped for p in particles]

        # collect batch inputs
        prompts, steps_so_far = [], []
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            prompts.append(prompt)
            steps_so_far.append(p.steps)

        # collect batch outputs
        sg_forward_results = self.sg.forward(lm, prompts, steps_so_far)

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
        scores = self.prm.score(
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

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | ParticleGibbsResult:
        assert budget % self.num_iterations == 0, (
            "budget must be divisible by num_iterations"
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

            current_step = 0  # Track current step outside the loop

            while not all(p.is_stopped for p in particles):
                particles = self._propagate(lm, particles, prompt, batched=True)
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
                resampled_particles = random.choices(
                    particles, weights=probabilities, k=num_free_particles
                )

                if self.does_ancestor_sampling:
                    raise NotImplementedError("Ancestor sampling is not implemented")

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
                [self.sg._post_process(p.steps, stopped=True) for p in particles]
            )
            log_weights_lst.append(log_weights)
            ref_indices_lst.append(ref_indices)
            steps_used_lst.append([len(p.steps) for p in particles])

        # select the chosen particle based on selection method
        # log_weights and probabilities are from the last iteration
        match self.selection_method:
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
        selection_method: str | SelectionMethod = SelectionMethod.ARGMAX,
    ):
        # initialize with num_iterations=1
        super().__init__(
            sg=sg,
            prm=prm,
            num_iterations=1,
            selection_method=selection_method,
            num_ref_particles=0,
            does_ancestor_sampling=False,
        )
