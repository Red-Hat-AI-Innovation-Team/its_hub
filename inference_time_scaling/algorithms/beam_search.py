from typing import Union, List
import copy
from dataclasses import dataclass
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
        
        candidates = [Path(steps=[], is_stopped=False, score=0) for _ in range(budget)]
        
        # create progress bar with total steps from sg.max_steps
        progress_bar = tqdm(total=self.sg.max_steps, desc="Stepping", disable=(not show_progress))
        
        while not all(c.is_stopped for c in candidates):
            for c in candidates:
                if c.is_stopped:
                    continue
                
                next_step, is_stopped = self.sg.forward(lm, prompt, c.steps)
                c.steps.append(next_step)
                c.is_stopped = is_stopped
                score = self.prm.score(prompt, c.steps)
                c.score = score[-1]

            # get the top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[:self.beam_width]
            
            # duplicate the candidates with the highest score
            new_candidates = []
            for _ in range(budget // self.beam_width):
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
            selected_index=scores.index(max(scores)),
        )
        return result if not return_response_only else result.the_one
