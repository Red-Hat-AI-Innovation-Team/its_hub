"""Clean tests for algorithms with improved organization and shared utilities."""

from collections import Counter
from copy import deepcopy
from typing import List
import pytest

from its_hub.algorithms.self_consistency import _select_most_common_or_random
from its_hub.algorithms.beam_search import BeamSearch, BeamSearchResult, Path
from its_hub.algorithms.particle_gibbs import (
    ParticleGibbs, ParticleGibbsResult, ParticleFiltering, 
    SelectionMethod, Particle
)
from its_hub.algorithms.bon import BestOfN, BestOfNResult
from its_hub.lms import StepGeneration

# Import from our new shared utilities
from tests.mocks.language_models import StepMockLanguageModel
from tests.mocks.reward_models import MockOutcomeRewardModel, MockProcessRewardModel
from tests.mocks.test_data import TestDataFactory, ALGORITHM_CONFIGS


class TestSelfConsistency:
    """Test the self-consistency algorithm utility functions."""
    
    @pytest.mark.parametrize("test_list,expected_counts,expected_element", [
        (['a', 'b', 'a', 'c', 'a'], Counter({'a': 3, 'b': 1, 'c': 1}), 'a'),
        (['a', 'b', 'a', 'b', 'c'], Counter({'a': 2, 'b': 2, 'c': 1}), ['a', 'b']),
        (['a', 'b', 'c', 'd'], Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1}), ['a', 'b', 'c', 'd']),
    ])
    def test_select_most_common_or_random(self, test_list, expected_counts, expected_element):
        """Test selection of most common element with various scenarios."""
        counts, selected_index = _select_most_common_or_random(test_list)
        
        assert counts == expected_counts
        
        if isinstance(expected_element, list):
            # Multiple possible winners
            assert test_list[selected_index] in expected_element
        else:
            # Single winner
            assert test_list[selected_index] == expected_element


class TestDataStructures:
    """Test core data structures used by algorithms."""
    
    def test_path_deepcopy(self):
        """Test Path deepcopy functionality."""
        steps = ['a', 'b', 'c']
        is_stopped = False
        score = 1.0
        path = Path(steps=deepcopy(steps), is_stopped=is_stopped, score=score)
        path_copy = path.deepcopy()
        path.steps.append('d')
        
        assert path_copy.steps == steps
        assert path_copy.is_stopped == is_stopped
        assert path_copy.score == score

    def test_particle_deepcopy(self):
        """Test Particle deepcopy functionality."""
        steps = ['a', 'b', 'c']
        is_stopped = False
        partial_log_weights = [0.3, 0.6, 1.0]
        particle = Particle(
            steps=deepcopy(steps), 
            is_stopped=is_stopped, 
            partial_log_weights=deepcopy(partial_log_weights)
        )
        particle_copy = particle.deepcopy()
        particle.steps.append('d')
        particle.partial_log_weights.append(1.2)
        
        assert particle_copy.steps == steps
        assert particle_copy.is_stopped == is_stopped
        assert particle_copy.log_weight == 1.0  # Should return last value of partial_log_weights
        assert particle_copy.partial_log_weights == partial_log_weights


class TestBestOfN:
    """Test the Best-of-N algorithm."""
    
    def test_result_structure(self):
        """Test BestOfNResult data structure."""
        responses = ["response1", "response2", "response3"]
        scores = [0.5, 0.8, 0.3]
        selected_index = 1
        
        result = BestOfNResult(responses=responses, scores=scores, selected_index=selected_index)
        
        assert result.responses == responses
        assert result.scores == scores
        assert result.selected_index == selected_index
        assert result.the_one == "response2"

    @pytest.mark.parametrize("responses,scores,expected_index,expected_response", [
        (["response1", "response2", "response3"], [0.5, 0.8, 0.3], 1, "response2"),
        (["response1", "response2", "response3"], [0.8, 0.5, 0.8], 0, "response1"),  # Tie - first wins
        (["response1"], [0.7], 0, "response1"),  # Single response
    ])
    def test_selection_logic(self, responses, scores, expected_index, expected_response):
        """Test Best-of-N selection logic with various score scenarios."""
        mock_lm = StepMockLanguageModel(responses)
        mock_orm = MockOutcomeRewardModel(scores)
        
        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=len(responses), return_response_only=False)
        
        assert result.selected_index == expected_index
        assert result.the_one == expected_response

    def test_return_response_only(self):
        """Test return_response_only parameter."""
        mock_lm = StepMockLanguageModel(["response1", "response2", "response3"])
        mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])
        
        bon = BestOfN(mock_orm)
        result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=True)
        
        assert result == "response2"


class TestBeamSearch:
    """Test the Beam Search algorithm."""
    
    def test_result_structure(self):
        """Test BeamSearchResult data structure."""
        responses = ["response1", "response2", "response3"]
        scores = [0.5, 0.8, 0.3]
        selected_index = 1
        steps_used = [2, 3, 1]
        
        result = BeamSearchResult(responses=responses, scores=scores, selected_index=selected_index, steps_used=steps_used)
        
        assert result.responses == responses
        assert result.scores == scores
        assert result.selected_index == selected_index
        assert result.steps_used == steps_used
        assert result.the_one == "response2"

    def test_basic_functionality(self):
        """Test basic beam search functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "stepA", "stepB"])
        mock_prm = MockProcessRewardModel([0.7, 0.9])
        
        sg = StepGeneration(step_token="\n", max_steps=2)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)
        
        result = beam_search.infer(mock_lm, "Solve this problem:", budget=2, return_response_only=True)
        
        assert isinstance(result, str)

    def test_budget_validation(self):
        """Test budget validation constraints."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)
        
        # Test budget not divisible by beam_width
        with pytest.raises(AssertionError, match="budget must be divisible by beam_width"):
            beam_search.infer(mock_lm, "test prompt", budget=3)

    def test_path_selection(self):
        """Test that beam search selects the highest scoring path."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step", "good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1, 0.8, 0.2])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        beam_search = BeamSearch(sg, mock_prm, beam_width=2)
        
        result = beam_search.infer(mock_lm, "Solve this:", budget=4, return_response_only=False)
        
        assert isinstance(result, BeamSearchResult)
        assert result.selected_index == result.scores.index(max(result.scores))


class TestParticleGibbs:
    """Test the Particle Gibbs algorithm."""
    
    def test_result_structure(self):
        """Test ParticleGibbsResult data structure."""
        responses_lst = [["response1", "response2"], ["response3", "response4"]]
        log_weights_lst = [[0.1, 0.2], [0.3, 0.4]]
        ref_indices_lst = [[0], [1]]
        selected_index = 1
        steps_used_lst = [[2, 3], [1, 4]]
        
        result = ParticleGibbsResult(
            responses_lst=responses_lst,
            log_weights_lst=log_weights_lst,
            ref_indices_lst=ref_indices_lst,
            selected_index=selected_index,
            steps_used_lst=steps_used_lst
        )
        
        assert result.responses_lst == responses_lst
        assert result.log_weights_lst == log_weights_lst
        assert result.ref_indices_lst == ref_indices_lst
        assert result.selected_index == selected_index
        assert result.steps_used_lst == steps_used_lst
        assert result.the_one == "response4"

    def test_basic_functionality(self):
        """Test basic particle Gibbs functionality."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
        
        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)
        
        assert isinstance(result, str)

    def test_budget_validation(self):
        """Test budget validation for particle Gibbs."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=3)
        
        with pytest.raises(AssertionError, match="budget must be divisible by num_iterations"):
            particle_gibbs.infer(mock_lm, "test prompt", budget=4)

    @pytest.mark.parametrize("selection_method,expected_type", [
        (SelectionMethod.ARGMAX, str),
        (SelectionMethod.SAMPLE, str),
        ("argmax", str),  # Test string conversion
    ])
    def test_selection_methods(self, selection_method, expected_type):
        """Test different selection methods."""
        mock_lm = StepMockLanguageModel(["good_step", "bad_step"])
        mock_prm = MockProcessRewardModel([0.9, 0.1])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=selection_method)
        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)
        
        assert isinstance(result, expected_type)

    def test_multiple_iterations(self):
        """Test particle Gibbs with multiple iterations."""
        mock_lm = StepMockLanguageModel(["step1", "step2", "step3", "step4"])
        mock_prm = MockProcessRewardModel([0.7, 0.6, 0.8, 0.5])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(
            sg, mock_prm, 
            num_iterations=2, 
            selection_method=SelectionMethod.ARGMAX,
            num_ref_particles=1
        )
        
        result = particle_gibbs.infer(mock_lm, "Solve this:", budget=4, return_response_only=False)
        
        assert isinstance(result, ParticleGibbsResult)
        assert len(result.responses_lst) == 2  # num_iterations = 2
        assert len(result.log_weights_lst) == 2
        assert len(result.ref_indices_lst) == 2

    def test_ancestor_sampling_not_implemented(self):
        """Test that ancestor sampling raises NotImplementedError."""
        mock_lm = StepMockLanguageModel(["step1"])
        mock_prm = MockProcessRewardModel([0.5])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        particle_gibbs = ParticleGibbs(
            sg, mock_prm, 
            num_iterations=1, 
            does_ancestor_sampling=True
        )
        
        with pytest.raises(NotImplementedError, match="Ancestor sampling is not implemented"):
            particle_gibbs.infer(mock_lm, "test prompt", budget=1)


class TestParticleFiltering:
    """Test the Particle Filtering algorithm (special case of Particle Gibbs)."""
    
    def test_is_single_iteration_particle_gibbs(self):
        """Test that ParticleFiltering is equivalent to ParticleGibbs with 1 iteration."""
        mock_lm = StepMockLanguageModel(["step1", "step2"])
        mock_prm = MockProcessRewardModel([0.7, 0.6])
        
        sg = StepGeneration(step_token="\n", max_steps=1)
        
        particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)
        result = particle_filtering.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)
        
        assert isinstance(result, ParticleGibbsResult)
        assert len(result.responses_lst) == 1  # Only 1 iteration
        assert len(result.responses_lst[0]) == 2  # budget = 2