from typing import Callable, Union, List
from collections import Counter
from dataclasses import dataclass
import random
from tqdm import tqdm
from .base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm

@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    responses: List[str]
    response_counts: Counter[str]
    selected_index: int
    
    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]

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
        responses = []
        for _ in tqdm(range(budget), desc="Generating responses", disable=(not show_progress)):
            responses.append(lm.generate(prompt))
        
        # project responses into consistency space
        responses_projected = [self.consistency_space_projection_func(r) for r in responses]
        
        # count occurrences of each projected response
        response_counts = Counter(responses_projected)
        
        # find the response with maximum occurrences
        max_count = max(response_counts.values())
        
        # find indices of the most common projected responses
        most_common_indices = [i for i, r in enumerate(responses_projected) 
                              if response_counts[r] == max_count]
        
        # select a random index from the most common ones
        # note above implementation ensures that if there are multiple 
        #      responses with the same count, a random one is selected
        selected_index = random.choice(most_common_indices)
        
        # return the result
        result = SelfConsistencyResult(
            responses=responses, 
            response_counts=response_counts, 
            selected_index=selected_index, 
        )
        return result.the_one if return_response_only else result
