import copy
import logging
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np

from its_hub.api import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
    ChatMessage,
    ChatMessages,
)
from its_hub.core.lms.step_generation import StepGeneration


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


class ParticleFiltering(AbstractScalingAlgorithm):
    """Particle filtering for inference-time scaling.

    Each particle is one step-by-step generation. After every step, particles
    are weighted by the *generator* model's own token logprobs/entropy for the
    step it just produced (self-certainty — no separate reward model), then
    resampled in proportion to those weights.
    """

    def __init__(
        self,
        sg: StepGeneration,
        final_response_selection: str | SelectionMethod = SelectionMethod.ARGMAX,
        resampling_method: str | ResamplingMethod = ResamplingMethod.MULTINOMIAL,
        self_certainty_signal: str = "mean_logprob",
        self_certainty_style: str = "logit",
        top_logprobs: int | None = None,
    ):
        if isinstance(final_response_selection, str):
            final_response_selection = SelectionMethod(final_response_selection)

        if isinstance(resampling_method, str):
            resampling_method = ResamplingMethod(resampling_method)

        if self_certainty_signal not in ("mean_logprob", "entropy"):
            raise ValueError(
                f"self_certainty_signal must be 'mean_logprob' or 'entropy', got {self_certainty_signal!r}"
            )
        if self_certainty_style not in ("logit", "raw"):
            raise ValueError(
                f"self_certainty_style must be 'logit' or 'raw', got {self_certainty_style!r}"
            )
        # entropy needs the per-token top-k distribution from the API
        if self_certainty_signal == "entropy" and top_logprobs is None:
            top_logprobs = 20

        self.sg = sg
        self.final_response_selection = final_response_selection
        self.resampling_method = resampling_method
        self.self_certainty_signal = self_certainty_signal
        self.self_certainty_style = self_certainty_style
        self.top_logprobs = top_logprobs
        self.max_steps = self.sg.max_steps
        self._warned_missing_logprobs = False

    def _self_certainty_logweight(self, summary: dict) -> float:
        """Convert a generated step's logprob summary into a particle log-weight.

        Both signals reduce to a confidence in log-space, ``c <= 0``, where
        ``exp(c)`` in ``(0, 1]`` is a per-token confidence:
          - ``'mean_logprob'``: ``c`` = mean per-token logprob.
          - ``'entropy'``:      ``c`` = -mean per-token entropy (falls back to
            ``mean_logprob`` if top_logprobs were unavailable).
        Styles:
          - ``'raw'``:   use ``c`` directly as the log-weight.
          - ``'logit'``: ``s = exp(c)`` in ``(0, 1]``, log-weight = ``_inv_sigmoid(s)``
            (maps the confidence onto unbounded log-odds for softmax resampling).

        A step with no logprob signal (``num_tokens == 0`` — e.g. the endpoint
        ignored ``logprobs=true`` or the request errored and was replaced) gets a
        NEUTRAL log-weight of 0.0. Without this, the summarizer's 0.0 fallback for
        ``mean_logprob`` would read as *perfect* confidence and the signal-less
        particle would dominate every resampling round.
        """
        if summary.get("num_tokens", 0) == 0:
            if not self._warned_missing_logprobs:
                self._warned_missing_logprobs = True
                logging.warning(
                    "Generated step carried no token logprobs (endpoint may not "
                    "support logprobs=true, or the request errored). Using a "
                    "neutral particle weight; self-certainty weighting is "
                    "degraded for affected steps."
                )
            return 0.0

        if self.self_certainty_signal == "entropy":
            entropy = summary.get("entropy")
            c = (
                -float(entropy)
                if entropy is not None
                else float(summary.get("mean_logprob", 0.0))
            )
        else:
            c = float(summary.get("mean_logprob", 0.0))

        if self.self_certainty_style == "raw":
            return c
        s = float(np.exp(min(c, 0.0)))  # in (0, 1]
        return _inv_sigmoid(s)

    async def _apropagate(
        self,
        lm: AbstractLanguageModel,
        particles: list[Particle],
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        base_messages: list[ChatMessage] | None = None,
    ) -> list[Particle]:
        """propagate particles asynchronously (batched).

        ``base_messages`` (when set) carries structured/multimodal conversation
        content (e.g. an audio user turn) verbatim to the model; ``prompt`` is
        the flattened text used for logging and the plain-text path.
        """
        is_stopped_in_the_beginning = [p.is_stopped for p in particles]

        # collect batch inputs
        prompts, steps_so_far = [], []
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            prompts.append(prompt)
            steps_so_far.append(p.steps)

        # generate the next step for each active particle, with logprobs so the
        # particle can be weighted from the generator's own confidence
        sg_forward_results = await self.sg.aforward(
            lm,
            prompts,
            steps_so_far,
            tools=tools,
            tool_choice=tool_choice,
            return_logprobs=True,
            top_logprobs=self.top_logprobs,
            base_messages=base_messages,
        )

        i = 0
        for p, is_stopped in zip(particles, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            next_step, step_is_stopped, summary = sg_forward_results[i]
            p.steps.append(next_step)
            p.is_stopped = step_is_stopped
            p.partial_log_weights.append(self._self_certainty_logweight(summary))
            i += 1

        return particles

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
        # guard against floating-point cumsum ending slightly below the last
        # position, which would walk j past the end of the array
        cumulative_sum[-1] = max(cumulative_sum[-1], 1.0)
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

    def _weights_to_probabilities(
        self, log_weights: list[float], current_step: int, num_particles: int
    ) -> np.ndarray:
        """Turn log-weights into the resampling distribution.

        Plain particle filtering uses an untempered softmax; entropic particle
        filtering overrides this to apply temperature annealing.
        """
        return _softmax(log_weights)

    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | ParticleFilteringResult:
        """run inference asynchronously with particle filtering"""
        assert budget >= 1, "budget must be a positive integer"

        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        # Carry structured (e.g. audio) messages through the step path verbatim;
        # fall back to the legacy flattened string prompt for plain-text inputs.
        carry_structured = chat_messages.has_nontext_content()
        base_messages = chat_messages.base_user_messages() if carry_structured else None
        prompt_str = chat_messages.to_prompt()

        num_particles = budget
        particles = [
            Particle(steps=[], is_stopped=False, partial_log_weights=[])
            for _ in range(num_particles)
        ]

        current_step = 0
        while not all(p.is_stopped for p in particles):
            particles = await self._apropagate(
                lm,
                particles,
                prompt_str,
                tools=tools,
                tool_choice=tool_choice,
                base_messages=base_messages,
            )

            current_step += 1  # Increment after propagation

            # resampling: every particle's most recent weight is its weight at
            # the current step (stopped particles keep their final weight)
            log_weights = [p.log_weight for p in particles]
            probabilities = self._weights_to_probabilities(
                log_weights, current_step, num_particles
            )

            # resample with replacement and duplicate the resampled particles
            particles = [
                p.deepcopy()
                for p in self._resampling(particles, probabilities, num_particles)
            ]

        # select the chosen particle (untempered weights)
        log_weights = [p.log_weight for p in particles]
        probabilities = _softmax(log_weights)
        match self.final_response_selection:
            case SelectionMethod.SAMPLE:
                selected_index = random.choices(
                    range(len(particles)), weights=probabilities, k=1
                )[0]
            case SelectionMethod.ARGMAX:
                selected_index = np.argmax(log_weights).item()

        result = ParticleFilteringResult(
            responses=[
                {
                    "role": "assistant",
                    "content": self.sg._post_process(p.steps, stopped=True),
                }
                for p in particles
            ],
            log_weights_lst=log_weights,
            selected_index=selected_index,
            steps_used_lst=[len(p.steps) for p in particles],
        )

        return result.the_one if return_response_only else result


class EntropicParticleFiltering(ParticleFiltering):
    """Particle filtering with entropic annealing.

    When the particle weights collapse early (low effective sample size), the
    resampling distribution is tempered (flattened) so diversity survives the
    early phase of generation. Final response selection is untempered.
    """

    def __init__(
        self,
        sg: StepGeneration,
        final_response_selection: str | SelectionMethod = SelectionMethod.ARGMAX,
        resampling_method: str | ResamplingMethod = ResamplingMethod.SYSTEMATIC,
        temperature_method: str | TemperatureMethod = TemperatureMethod.ESS,
        ess_threshold: float = 0.5,
        early_phase: float = 0.5,
        self_certainty_signal: str = "mean_logprob",
        self_certainty_style: str = "logit",
        top_logprobs: int | None = None,
    ):
        super().__init__(
            sg=sg,
            final_response_selection=final_response_selection,
            resampling_method=resampling_method,
            self_certainty_signal=self_certainty_signal,
            self_certainty_style=self_certainty_style,
            top_logprobs=top_logprobs,
        )

        if isinstance(temperature_method, str):
            temperature_method = TemperatureMethod(temperature_method)

        self.temperature_method = temperature_method
        self.ess_threshold = ess_threshold
        self.early_phase = early_phase

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

        # Make normalization robust to edge cases (len(p) <= 1, where log(1) == 0)
        entropy_normalized = entropy / np.log(len(p)) if len(p) > 1 else entropy
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

    def _weights_to_probabilities(
        self, log_weights: list[float], current_step: int, num_particles: int
    ) -> np.ndarray:
        probabilities = _softmax(log_weights)
        temperature = self._temperature_annealing(
            probabilities, current_step, num_particles
        )
        # apply temperature annealing to the log weights
        return _softmax(np.asarray(log_weights) * (1 / temperature))
