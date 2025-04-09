from inference_time_scaling.algorithms import _select_most_common_or_random
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
