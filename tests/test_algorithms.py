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
