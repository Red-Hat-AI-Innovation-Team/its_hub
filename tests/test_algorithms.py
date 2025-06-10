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
        if isinstance(messages[0], list):
            responses = self.responses[self.call_count:self.call_count + len(messages)]
            self.call_count += len(messages)
            return responses
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
