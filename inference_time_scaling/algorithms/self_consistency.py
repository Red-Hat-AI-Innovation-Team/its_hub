from typing import Callable, Union, List
from collections import Counter
from dataclasses import dataclass
import random
from tqdm import tqdm

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm


@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: List[str]
    response_counts: Counter[str]
    selected_index: int
    
    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]

def _select_most_common_or_random(list_to_select_from: List[str]) -> int:
    # count occurrences of each element
    counts = Counter(list_to_select_from)
    
    # find the element with maximum occurrences
    max_count = max(counts.values())
    
    # find indices of the most common elements
    most_common_indices = [i for i, r in enumerate(list_to_select_from) 
                           if counts[r] == max_count]
    
    # select a random index from the most common ones
    # note above implementation ensures that if there are multiple 
    #      elements with the same count, a random one is selected
    selected_index = random.choice(most_common_indices)

    return counts, selected_index

class SelfConsistency(AbstractScalingAlgorithm):
    def __init__(self, consistency_space_projection_func: Callable):
        self.consistency_space_projection_func = consistency_space_projection_func

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, SelfConsistencyResult]:
        # generate responses
        responses = lm.generate([prompt] * budget)
        
        # project responses into consistency space
        responses_projected = [self.consistency_space_projection_func(r) for r in responses]

        # select the most common or random response
        response_counts, selected_index = _select_most_common_or_random(responses_projected)
        
        # return the result
        result = SelfConsistencyResult(
            responses=responses, 
            response_counts=response_counts, 
            selected_index=selected_index, 
        )
        return result.the_one if return_response_only else result