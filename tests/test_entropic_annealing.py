"""Tests for entropic annealing in EntropicParticleFiltering."""

import pytest

from its_hub import StepGeneration
from its_hub.core.algorithms.particle_filtering import (
    EntropicParticleFiltering,
    Particle,
    ParticleFilteringResult,
    ResamplingMethod,
    SelectionMethod,
    TemperatureMethod,
)
from tests.mocks.language_models import LogprobMockLM


def _epf(
    resampling_method=ResamplingMethod.MULTINOMIAL,
    temperature_method=TemperatureMethod.ESS,
) -> EntropicParticleFiltering:
    return EntropicParticleFiltering(
        sg=StepGeneration(step_token="\n", max_steps=3),
        final_response_selection=SelectionMethod.ARGMAX,
        resampling_method=resampling_method,
        temperature_method=temperature_method,
        ess_threshold=0.5,
        early_phase=0.5,
    )


class TestEntropicAnnealing:
    """Test the entropic annealing."""

    def test_effective_sample_size(self):
        epf = _epf()
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        ess = epf._effective_sample_size(probabilities)
        assert isinstance(ess, float)
        assert ess == 1.0 / (0.1**2 + 0.2**2 + 0.3**2 + 0.4**2 + 0.5**2)

    def test_resampling(self):
        epf = _epf()
        particles = [
            Particle(steps=[f"p{i}"], is_stopped=False, partial_log_weights=[0.0])
            for i in range(1, 6)
        ]

        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        resampled_particles = epf._resampling_multinomial(
            particles, probabilities, len(probabilities)
        )
        assert isinstance(resampled_particles, list)
        assert len(resampled_particles) == len(probabilities)

        resampled_particles = epf._resampling_systematic(
            particles, probabilities, len(probabilities)
        )
        assert isinstance(resampled_particles, list)
        assert len(resampled_particles) == len(probabilities)

    def test_temperature_functions(self):
        """Test the temperature functions."""
        epf = _epf()

        # Test ESS temperature early phase
        t = epf._temperature_ess(ess_ratio=0.2, progress=0.2)
        assert isinstance(t, float)
        assert t == 4.0

        # Test ESS temperature late phase
        t = epf._temperature_ess(ess_ratio=0.5, progress=0.8)
        assert isinstance(t, float)
        assert t == 1.0

        # Test entropy temperature
        t = epf._temperature_entropy(entropy_n=0.5, progress=0.3)
        v = 1.0 / (0.5 + (1 - 0.5) * 0.3)
        assert isinstance(t, float)
        assert t == v

        # Test entropy temperature edge case
        t = epf._temperature_entropy(entropy_n=1.0, progress=0.2)
        assert isinstance(t, float)
        assert t == 1.0

        # Test base temperature
        t = epf._temperature_base(value_max=2.0, progress=0.5)
        assert isinstance(t, float)
        assert t == 1.50

        # Test base temperature edge case
        t = epf._temperature_base(value_max=0.8, progress=0.5)
        assert isinstance(t, float)
        assert t == 1.0

    @pytest.mark.parametrize(
        "resampling_method",
        [ResamplingMethod.MULTINOMIAL, ResamplingMethod.SYSTEMATIC],
    )
    @pytest.mark.parametrize(
        "temperature_method",
        [TemperatureMethod.ESS, TemperatureMethod.ENTROPY, TemperatureMethod.BASE],
    )
    def test_entropic_annealing_end_to_end(self, resampling_method, temperature_method):
        """EPF runs end-to-end for every temperature x resampling combination."""
        # varying mean logprobs => particles get a spread of self-certainty weights
        mock_lm = LogprobMockLM(mean_logprobs=(-0.1, -0.5, -1.0, -0.2))
        epf = _epf(
            resampling_method=resampling_method,
            temperature_method=temperature_method,
        )

        n = 4
        result = epf.infer(mock_lm, "Test prompt", budget=n, return_response_only=False)
        # Verify the result structure
        assert isinstance(result, ParticleFilteringResult)
        assert len(result.responses) == n
        assert len(result.log_weights_lst) == n
        assert isinstance(result.log_weights_lst, list)
        assert isinstance(result.selected_index, int)
        assert mock_lm.saw_logprobs is True
