from typing import Union, List
import copy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..base import AbstractLanguageModel, AbstractScalingResult, AbstractScalingAlgorithm, AbstractProcessRewardModel
from ..lms import StepGeneration


@dataclass
class ParticleGibbsResult(AbstractScalingResult):
    responses: List[str]
    log_weights: List[float]
    selected_index: int

    @property
    def the_one(self) -> str:
        return self.responses[self.selected_index]

@dataclass
class Particle:
    steps: List[str]
    is_stopped: bool
    log_weight: float
    
    def deepcopy(self):
        # create a deep copy of the particle object
        return Particle(
            steps=copy.deepcopy(self.steps), 
            is_stopped=self.is_stopped, 
            log_weight=self.log_weight, 
        )

def _inv_sigmoid(x):
    assert 0 <= x <= 1, "x must be between 0 and 1"
    # clip values to avoid numerical issues when x is close to 0 or 1
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))

def _softmax(x):
    # shift x by the maximum value for numerical stability
    x_shifted = x - np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))

class ParticleGibbs(AbstractScalingAlgorithm):
    """
    Particle-based Monte Carlo methods for inference time scaling.
    It supports the following variants:
    - Particle Filtering (PF): num_iterations = 1
    - Particle Gibbs (PG): num_iterations > 1
    - PG with ancestor sampling (PGAS): num_iterations > 1 and does_ancestor_sampling = True
    """
    def __init__(
        self, 
        sg: StepGeneration, 
        prm: AbstractProcessRewardModel, 
        num_iterations: int = 1, 
        does_ancestor_sampling: bool = False, 
    ):
        self.sg = sg
        self.prm = prm
        self.num_iterations = num_iterations
        self.does_ancestor_sampling = does_ancestor_sampling

    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, ParticleGibbsResult]:
        assert budget % self.num_iterations == 0, "budget must be divisible by num_iterations"

        if self.num_iterations > 1:
            raise NotImplementedError("Particle Gibbs is not implemented")
        
        num_particles = budget // self.num_iterations
        particles = [Particle(steps=[], is_stopped=False, log_weight=0) 
                      for _ in range(num_particles)]
        
        # create progress bar with total steps from sg.max_steps
        progress_bar = tqdm(total=self.sg.max_steps, desc="Stepping", disable=(not show_progress))
        
        while not all(p.is_stopped for p in particles):
            for p in particles:
                if p.is_stopped:
                    continue
                
                next_step, is_stopped = self.sg.forward(lm, prompt, p.steps)
                p.steps.append(next_step)
                p.is_stopped = is_stopped
                score = self.prm.score(prompt, p.steps)
                # TODO generalize the PRM score aggregation
                p.log_weight = _inv_sigmoid(score[-1])

            # resampling
            log_weights = [p.log_weight for p in particles]
            probabilities = _softmax(log_weights)
            particles = np.random.choice(particles, size=num_particles, p=probabilities)

            if self.does_ancestor_sampling:
                raise NotImplementedError("Ancestor sampling is not implemented")
            
            # duplicate the particles
            particles = [p.deepcopy() for p in particles]
            
            # update progress bar
            progress_bar.update(1)
        
        # close the progress bar
        progress_bar.close()
        
        log_weights = [p.log_weight for p in particles]
        result = ParticleGibbsResult(
            responses=[self.sg.step_token.join(p.steps) for p in particles],
            log_weights=log_weights,
            selected_index=int(np.argmax(log_weights)),
        )
        return result if not return_response_only else result.the_one

