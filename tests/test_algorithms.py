from its_hub.algorithms.self_consistency import _select_most_common_or_random
from collections import Counter

def test_select_most_common_or_random_single_winner():
    # test case with a single most common element
    test_list = ['a', 'b', 'a', 'c', 'a']
    counts, selected_index = _select_most_common_or_random(test_list)
    
    # verify counts are correct
    assert counts == Counter({'a': 3, 'b': 1, 'c': 1})
    
    # verify selected index points to 'a'
    assert test_list[selected_index] == 'a'

def test_select_most_common_or_random_tie():
    # test case with multiple most common elements
    test_list = ['a', 'b', 'a', 'b', 'c']
    counts, selected_index = _select_most_common_or_random(test_list)
    
    # verify counts are correct
    assert counts == Counter({'a': 2, 'b': 2, 'c': 1})
    
    # verify selected index points to either 'a' or 'b'
    assert test_list[selected_index] in ['a', 'b']

def test_select_most_common_or_random_all_unique():
    # test case where all elements are unique
    test_list = ['a', 'b', 'c', 'd']
    counts, selected_index = _select_most_common_or_random(test_list)
    
    # verify counts are correct
    assert counts == Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1})
    
    # verify selected index points to one of the elements
    assert test_list[selected_index] in test_list

from copy import deepcopy
from its_hub.algorithms.beam_search import Path
from its_hub.algorithms.particle_gibbs import Particle
from its_hub.algorithms.bon import BestOfN, BestOfNResult
from its_hub.base import AbstractLanguageModel, AbstractOutcomeRewardModel
from typing import List, Union

def test_path_deepcopy():
    steps = ['a', 'b', 'c']
    is_stopped = False
    score = 1.0
    path = Path(steps=deepcopy(steps), is_stopped=is_stopped, score=score)
    path_copy = path.deepcopy()
    path.steps.append('d')
    assert path_copy.steps == steps
    assert path_copy.is_stopped == is_stopped
    assert path_copy.score == score

def test_particle_deepcopy():
    steps = ['a', 'b', 'c']
    is_stopped = False
    log_weight = 1.0
    particle = Particle(steps=deepcopy(steps), is_stopped=is_stopped, log_weight=log_weight)
    particle_copy = particle.deepcopy()
    particle.steps.append('d')
    assert particle_copy.steps == steps
    assert particle_copy.is_stopped == is_stopped
    assert particle_copy.log_weight == log_weight


class MockLanguageModel(AbstractLanguageModel):
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        
    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batched generation - messages is List[List[ChatMessage]]
            num_requests = len(messages)
            if self.call_count + num_requests > len(self.responses):
                # Cycle through responses if we run out
                responses = []
                for i in range(num_requests):
                    responses.append(self.responses[(self.call_count + i) % len(self.responses)])
            else:
                responses = self.responses[self.call_count:self.call_count + num_requests]
            self.call_count += num_requests
            return responses
        else:
            # Single generation - messages is List[ChatMessage]
            if self.call_count >= len(self.responses):
                # Cycle through responses if we run out
                response = self.responses[self.call_count % len(self.responses)]
            else:
                response = self.responses[self.call_count]
            self.call_count += 1
            return response
            
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        return [0.1] * len(generation.split())


class MockOutcomeRewardModel(AbstractOutcomeRewardModel):
    def __init__(self, scores: Union[List[float], float]):
        if isinstance(scores, float):
            self.scores = [scores]
        else:
            self.scores = scores
        self.call_count = 0
        
    def score(self, prompt: str, response: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(response, list):
            scores = self.scores[self.call_count:self.call_count + len(response)]
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count]
            self.call_count += 1
            return score


def test_best_of_n_result():
    responses = ["response1", "response2", "response3"]
    scores = [0.5, 0.8, 0.3]
    selected_index = 1
    
    result = BestOfNResult(responses=responses, scores=scores, selected_index=selected_index)
    
    assert result.responses == responses
    assert result.scores == scores
    assert result.selected_index == selected_index
    assert result.the_one == "response2"


def test_best_of_n_basic():
    mock_lm = MockLanguageModel(["response1", "response2", "response3"])
    mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])
    
    bon = BestOfN(mock_orm)
    result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=True)
    
    assert result == "response2"


def test_best_of_n_return_full_result():
    mock_lm = MockLanguageModel(["response1", "response2", "response3"])
    mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])
    
    bon = BestOfN(mock_orm)
    result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=False)
    
    assert isinstance(result, BestOfNResult)
    assert result.responses == ["response1", "response2", "response3"]
    assert result.scores == [0.5, 0.8, 0.3]
    assert result.selected_index == 1
    assert result.the_one == "response2"


def test_best_of_n_batched_scoring():
    mock_lm = MockLanguageModel(["response1", "response2", "response3"])
    mock_orm = MockOutcomeRewardModel([0.5, 0.8, 0.3])
    
    bon = BestOfN(mock_orm)
    result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=False)
    
    assert result.scores == [0.5, 0.8, 0.3]
    assert result.selected_index == 1


def test_best_of_n_tie_scores():
    mock_lm = MockLanguageModel(["response1", "response2", "response3"])
    mock_orm = MockOutcomeRewardModel([0.8, 0.5, 0.8])
    
    bon = BestOfN(mock_orm)
    result = bon.infer(mock_lm, "test prompt", budget=3, return_response_only=False)
    
    assert result.scores == [0.8, 0.5, 0.8]
    assert result.selected_index == 0  # should select first occurrence of max score
    assert result.the_one == "response1"


def test_best_of_n_single_response():
    mock_lm = MockLanguageModel(["response1"])
    mock_orm = MockOutcomeRewardModel([0.7])
    
    bon = BestOfN(mock_orm)
    result = bon.infer(mock_lm, "test prompt", budget=1, return_response_only=False)
    
    assert result.responses == ["response1"]
    assert result.scores == [0.7]
    assert result.selected_index == 0
    assert result.the_one == "response1"


# Mock Process Reward Model for beam search and particle Gibbs tests
class MockProcessRewardModel:
    def __init__(self, scores: Union[List[float], List[List[float]]]):
        if isinstance(scores[0], float):
            self.scores = scores
        else:
            self.scores = [score for sublist in scores for score in sublist]
        self.call_count = 0
        
    def score(self, prompt: str, response: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                scores.append(self.scores[(self.call_count + i) % len(self.scores)])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


# Beam Search Tests
from its_hub.algorithms.beam_search import BeamSearch, BeamSearchResult
from its_hub.lms import StepGeneration

def test_beam_search_result():
    responses = ["response1", "response2", "response3"]
    scores = [0.5, 0.8, 0.3]
    selected_index = 1
    
    result = BeamSearchResult(responses=responses, scores=scores, selected_index=selected_index)
    
    assert result.responses == responses
    assert result.scores == scores
    assert result.selected_index == selected_index
    assert result.the_one == "response2"


def test_beam_search_basic():
    # Mock LM that returns step-by-step responses
    mock_lm = MockLanguageModel(["step1", "step2", "stepA", "stepB"])
    mock_prm = MockProcessRewardModel([0.7, 0.9])
    
    sg = StepGeneration(step_token="\n", max_steps=2)
    beam_search = BeamSearch(sg, mock_prm, beam_width=2)
    
    result = beam_search.infer(mock_lm, "Solve this problem:", budget=2, return_response_only=True)
    
    assert isinstance(result, str)


def test_beam_search_return_full_result():
    # Mock LM that returns step-by-step responses
    mock_lm = MockLanguageModel(["step1", "step2", "stepA", "stepB"])
    mock_prm = MockProcessRewardModel([0.7, 0.9])
    
    sg = StepGeneration(step_token="\n", max_steps=2)
    beam_search = BeamSearch(sg, mock_prm, beam_width=2)
    
    result = beam_search.infer(mock_lm, "Solve this problem:", budget=4, return_response_only=False)
    
    assert isinstance(result, BeamSearchResult)
    # With budget=4 and beam_width=2, we get num_beams=2, and final candidates = num_beams * beam_width = 4
    assert len(result.responses) == 4
    assert len(result.scores) == 4
    assert result.selected_index in range(4)
    assert result.the_one in result.responses


def test_beam_search_budget_validation():
    mock_lm = MockLanguageModel(["step1"])
    mock_prm = MockProcessRewardModel([0.5])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    beam_search = BeamSearch(sg, mock_prm, beam_width=2)
    
    # Test budget not divisible by beam_width
    try:
        beam_search.infer(mock_lm, "test prompt", budget=3)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "budget must be divisible by beam_width" in str(e)
    
    # Test budget less than beam_width - this will also trigger divisible check since 1 % 2 != 0
    try:
        beam_search.infer(mock_lm, "test prompt", budget=1)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        # The first assertion that fails is the divisibility check
        assert "budget must be divisible by beam_width" in str(e)


def test_beam_search_path_selection():
    # Mock LM with predictable responses
    mock_lm = MockLanguageModel(["good_step", "bad_step", "good_step", "bad_step"])
    # Higher score for responses containing "good"
    mock_prm = MockProcessRewardModel([0.9, 0.1, 0.8, 0.2])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    beam_search = BeamSearch(sg, mock_prm, beam_width=2)
    
    result = beam_search.infer(mock_lm, "Solve this:", budget=4, return_response_only=False)
    
    assert isinstance(result, BeamSearchResult)
    # Should select the path with highest score
    assert result.selected_index == result.scores.index(max(result.scores))


# Particle Gibbs Tests  
from its_hub.algorithms.particle_gibbs import ParticleGibbs, ParticleGibbsResult, ParticleFiltering, SelectionMethod

def test_particle_gibbs_result():
    responses_lst = [["response1", "response2"], ["response3", "response4"]]
    log_weights_lst = [[0.1, 0.2], [0.3, 0.4]]
    ref_indices_lst = [[0], [1]]
    selected_index = 1
    
    result = ParticleGibbsResult(
        responses_lst=responses_lst,
        log_weights_lst=log_weights_lst,
        ref_indices_lst=ref_indices_lst,
        selected_index=selected_index
    )
    
    assert result.responses_lst == responses_lst
    assert result.log_weights_lst == log_weights_lst
    assert result.ref_indices_lst == ref_indices_lst
    assert result.selected_index == selected_index
    assert result.the_one == "response4"


def test_particle_gibbs_basic():
    mock_lm = MockLanguageModel(["step1", "step2"])
    mock_prm = MockProcessRewardModel([0.7, 0.6])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
    
    result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=True)
    
    assert isinstance(result, str)


def test_particle_gibbs_return_full_result():
    mock_lm = MockLanguageModel(["step1", "step2"])
    mock_prm = MockProcessRewardModel([0.7, 0.6])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
    
    result = particle_gibbs.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)
    
    assert isinstance(result, ParticleGibbsResult)
    assert len(result.responses_lst) == 1  # num_iterations = 1
    assert len(result.responses_lst[0]) == 2  # budget = 2
    assert len(result.log_weights_lst) == 1
    assert len(result.ref_indices_lst) == 1
    assert result.selected_index in range(len(result.responses_lst[-1]))


def test_particle_gibbs_multiple_iterations():
    mock_lm = MockLanguageModel(["step1", "step2", "step3", "step4"])
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


def test_particle_gibbs_budget_validation():
    mock_lm = MockLanguageModel(["step1"])
    mock_prm = MockProcessRewardModel([0.5])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=3)
    
    # Test budget not divisible by num_iterations
    try:
        particle_gibbs.infer(mock_lm, "test prompt", budget=4)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "budget must be divisible by num_iterations" in str(e)


def test_particle_gibbs_selection_methods():
    mock_lm = MockLanguageModel(["good_step", "bad_step"])
    mock_prm = MockProcessRewardModel([0.9, 0.1])  # Clear winner
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    
    # Test ARGMAX selection
    particle_gibbs_argmax = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.ARGMAX)
    result_argmax = particle_gibbs_argmax.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)
    
    # Should select the particle with highest log weight
    assert result_argmax.selected_index == 0  # First particle should have higher score
    
    # Test SAMPLE selection (just ensure it doesn't crash)
    particle_gibbs_sample = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method=SelectionMethod.SAMPLE)
    result_sample = particle_gibbs_sample.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)
    
    assert result_sample.selected_index in range(len(result_sample.responses_lst[-1]))


def test_particle_gibbs_string_selection_method():
    mock_lm = MockLanguageModel(["step1"])
    mock_prm = MockProcessRewardModel([0.7])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    
    # Test string selection method conversion
    particle_gibbs = ParticleGibbs(sg, mock_prm, num_iterations=1, selection_method="argmax")
    result = particle_gibbs.infer(mock_lm, "Solve this:", budget=1, return_response_only=True)
    
    assert isinstance(result, str)


def test_particle_filtering():
    mock_lm = MockLanguageModel(["step1", "step2"])
    mock_prm = MockProcessRewardModel([0.7, 0.6])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    
    # ParticleFiltering is a special case of ParticleGibbs with num_iterations=1
    particle_filtering = ParticleFiltering(sg, mock_prm, selection_method=SelectionMethod.ARGMAX)
    result = particle_filtering.infer(mock_lm, "Solve this:", budget=2, return_response_only=False)
    
    assert isinstance(result, ParticleGibbsResult)
    assert len(result.responses_lst) == 1  # num_iterations = 1 (built into ParticleFiltering)
    assert len(result.responses_lst[0]) == 2  # budget = 2


def test_particle_gibbs_ancestor_sampling_not_implemented():
    mock_lm = MockLanguageModel(["step1"])
    mock_prm = MockProcessRewardModel([0.5])
    
    sg = StepGeneration(step_token="\n", max_steps=1)
    particle_gibbs = ParticleGibbs(
        sg, mock_prm, 
        num_iterations=1, 
        does_ancestor_sampling=True
    )
    
    try:
        particle_gibbs.infer(mock_lm, "test prompt", budget=1)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Ancestor sampling is not implemented" in str(e)
