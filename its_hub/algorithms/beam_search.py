from typing import Union, List
import copy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractProcessRewardModel
from ..lms import StepGeneration


@dataclass
class BeamSearchResult(AbstractScalingResult):
    responses: List[str]
    scores: List[float]
    selected_index: int

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]

@dataclass
class Path:
    steps: List[str]
    is_stopped: bool
    score: float
    
    def deepcopy(self):
        # create a deep copy of the path object
        return Path(
            steps=copy.deepcopy(self.steps),
            is_stopped=self.is_stopped,
            score=self.score
        )

class BeamSearch(AbstractScalingAlgorithm):
    def __init__(
        self, 
        sg: StepGeneration, 
        prm: AbstractProcessRewardModel, 
        beam_width: int,
    ):
        self.sg = sg
        self.prm = prm
        self.beam_width = beam_width

    def _search_one_level(
        self, 
        lm: AbstractLanguageModel, 
        candidates: List[Path], 
        prompt: str, 
        batched: bool = False
    ) -> List[Path]:
        if not batched:
            for c in candidates:
                if c.is_stopped:
                    continue
                
                next_step, is_stopped = self.sg.forward(lm, prompt, c.steps)
                c.steps.append(next_step)
                c.is_stopped = is_stopped
                score = self.prm.score(prompt, c.steps)
                c.score = score
            
            return candidates
        
        is_stopped_in_the_beginning = [c.is_stopped for c in candidates]
        
        # collect batch inputs
        prompts, steps_so_far = [], []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            
            prompts.append(prompt)
            steps_so_far.append(c.steps)
        
        # collect batch outputs
        sg_forward_results = self.sg.forward(lm, prompts, steps_so_far)
    
        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            
            next_step, is_stopped = sg_forward_results[i]
            c.steps.append(next_step)
            c.is_stopped = is_stopped
            i += 1

        # collect batch inputs for scoring
        steps_so_far = []
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue
            
            steps_so_far.append(c.steps)

        # collect batch outputs for scoring
        scores = self.prm.score(prompt, steps_so_far)
        
        # update candidates
        i = 0
        for c, is_stopped in zip(candidates, is_stopped_in_the_beginning):
            if is_stopped:
                continue

            c.score = scores[i]
            i += 1
        
        return candidates

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, BeamSearchResult]:
        assert budget % self.beam_width == 0, "budget must be divisible by beam_width"
        assert budget >= self.beam_width, "budget must be greater than or equal to beam_width"

        num_beams = budget // self.beam_width
        
        candidates = [Path(steps=[], is_stopped=False, score=0) for _ in range(num_beams)]
        
        # create progress bar with total steps from sg.max_steps
        progress_bar = tqdm(total=self.sg.max_steps, desc="Stepping", disable=(not show_progress))
        
        while not all(c.is_stopped for c in candidates):
            candidates = self._search_one_level(lm, candidates, prompt, batched=True)

            # get the top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[:self.beam_width]
            
            # duplicate the candidates with the highest score
            new_candidates = []
            for _ in range(num_beams):
                for c in candidates:
                    new_candidates.append(c.deepcopy())
            candidates = new_candidates
            
            # update progress bar
            progress_bar.update(1)
        
        # close the progress bar
        progress_bar.close()
        
        scores = [c.score for c in candidates]
        result = BeamSearchResult(
            responses=[self.sg.step_token.join(c.steps) for c in candidates],
            scores=scores,
            selected_index=int(np.argmax(scores)),
        )
        return result.the_one if return_response_only else result
