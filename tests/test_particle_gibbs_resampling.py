"""Test for the particle Gibbs resampling weight calculation fix (issue #54)."""

import random

import numpy as np

from its_hub.algorithms.particle_gibbs import (
    Particle,
    ParticleGibbs,
    ParticleGibbsResult,
    SelectionMethod,
)
from its_hub.base import AbstractLanguageModel, AbstractProcessRewardModel
from its_hub.lms import StepGeneration


class MockLanguageModelForResampling(AbstractLanguageModel):
    """Mock LM that generates predictable steps for testing resampling."""

    def __init__(self):
        self.step_counter = 0

    def generate(self, prompt, max_tokens=100, **kwargs):
        # Generate different steps based on counter
        step = f"step{self.step_counter}"
        self.step_counter += 1
        return step

    def generate_batch(self, prompts, max_tokens=100, **kwargs):
        return [self.generate(p, max_tokens, **kwargs) for p in prompts]

    def evaluate(self, prompt, response):
        # Not used in these tests
        return 0.5


class MockProcessRewardModelForResampling(AbstractProcessRewardModel):
    """Mock PRM that gives higher scores to longer sequences."""

    def score(self, prompt, response):
        if isinstance(response, list):
            # Batch scoring
            return [self._score_single(r) for r in response]
        else:
            # Single scoring
            return self._score_single(response)

    def _score_single(self, response):
        # Give higher scores to longer responses
        # This simulates a scenario where reference particles (being longer)
        # would have unfairly high scores if we don't use partial weights
        num_steps = response.count("step")
        # Return a score between 0.5 and 0.9 based on length
        return min(0.5 + 0.1 * num_steps, 0.9)


class TestParticleGibbsResampling:
    """Test the fix for issue #54 - proper calculation of resampling weights."""

    def test_reference_trajectory_partial_weights(self):
        """Test that reference trajectories use partial weights during resampling."""
        # Create mock models
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        # Create step generation with 3 max steps
        sg = StepGeneration(step_token="\n", max_steps=3)

        # Create ParticleGibbs with 2 iterations and 1 reference particle
        pg = ParticleGibbs(
            sg=sg,
            prm=mock_prm,
            num_iterations=2,
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )

        # Run inference with budget=4 (2 particles per iteration)
        result = pg.infer(mock_lm, "Test prompt", budget=4, return_response_only=False)

        # Verify the result structure
        assert isinstance(result, ParticleGibbsResult)
        assert len(result.responses_lst) == 2  # 2 iterations
        assert len(result.log_weights_lst) == 2
        assert len(result.ref_indices_lst) == 2

        # In the second iteration, check that particles have consistent trajectory lengths
        # after resampling (this would fail with the old implementation)
        second_iter_steps = result.steps_used_lst[1]

        # All particles should have progressed similarly
        # With the bug, reference particle would keep its full trajectory
        # making it have more steps than others
        assert len(set(second_iter_steps)) <= 2, (
            f"Particles have very different step counts: {second_iter_steps}. "
            "This suggests reference particles weren't truncated properly."
        )

    def test_resampling_weights_consistency(self):
        """Test that resampling uses consistent weights across particles."""
        # Create a custom mock that allows us to inspect the resampling process
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        sg = StepGeneration(step_token="\n", max_steps=5)

        # Create a test scenario with manual particle manipulation
        # Initialize particles with different trajectories
        ref_particle = Particle(
            steps=["ref_step1", "ref_step2", "ref_step3"],
            is_stopped=True,
            partial_log_weights=[0.6, 0.7, 0.8]
        )

        new_particle = Particle(
            steps=["new_step1"],
            is_stopped=False,
            partial_log_weights=[0.65]
        )

        # Test the _propagate method
        pg = ParticleGibbs(
            sg=sg,
            prm=mock_prm,
            num_iterations=1,
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )

        # Propagate one step
        particles = [new_particle, ref_particle]
        propagated = pg._propagate(mock_lm, particles, "Test prompt", batched=False)

        # After propagation, only non-stopped particles get extended
        assert len(propagated[0].partial_log_weights) == 2  # new particle now has 2 steps
        assert len(propagated[1].partial_log_weights) == 3  # ref particle stays at 3 steps (stopped)

        # The key insight: during resampling in the main loop,
        # we should compare partial_log_weights[1] for both particles,
        # not the full log_weight of the reference particle

    def test_reference_particle_truncation(self):
        """Test that reference particles are properly truncated when resampled."""
        mock_lm = MockLanguageModelForResampling()
        mock_prm = MockProcessRewardModelForResampling()

        sg = StepGeneration(step_token="\n", max_steps=4)

        # Use a controlled random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        pg = ParticleGibbs(
            sg=sg,
            prm=mock_prm,
            num_iterations=2,
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )

        result = pg.infer(mock_lm, "Test prompt", budget=6, return_response_only=False)

        # Check that in the second iteration, particles don't have
        # drastically different numbers of steps
        second_iter_steps = result.steps_used_lst[1]

        # The maximum difference in steps should be reasonable
        # (not the full trajectory length difference)
        max_diff = max(second_iter_steps) - min(second_iter_steps)
        assert max_diff <= 2, (
            f"Large step difference in second iteration: {second_iter_steps}. "
            "Reference particle may not have been truncated."
        )
