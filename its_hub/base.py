from typing import Union, List
from abc import ABC, abstractmethod


class AbstractLanguageModel(ABC):
    """abstract base class for (autoregressive) language models"""
    
    @abstractmethod
    def generate(self, prompt: str, stop: str = None) -> str:
        """
        generate a response from the model
        
        Args:
            prompt: the input prompt
            
        Returns:
            the generated output string
        """
        pass

    @abstractmethod
    def evaluate(self, prompt: str, generation: str) -> List[float]:
        """
        evaluate the likelihoods of the generation given the prompt
        
        Args:
            prompt: the input prompt
            generation: the generated output string
            
        Returns:
            the likelihoods of the generation per token
        """
        pass

class AbstractScalingResult(ABC):
    """abstract base class for scaling result"""
    
    @property
    @abstractmethod
    def the_one(self) -> str:
        """the selected response"""
        pass

class AbstractScalingAlgorithm(ABC):
    """abstract base class for inference-time scaling algorithms"""
    
    @abstractmethod
    def infer(
        self, 
        lm: AbstractLanguageModel, 
        prompt: str, 
        budget: int, 
        show_progress: bool = False, 
        return_response_only: bool = True, 
    ) -> Union[str, AbstractScalingResult]:
        """
        run inference with the given language model and prompt under the specified budget
        
        Args:
            lm: a language model that takes a prompt and returns a response
            prompt: the input prompt
            budget: the computational budget for inference
            show_progress: whether to show a progress bar
            return_response_only: whether to return only the selected response
            
        Returns:
            the generated output string or the complete scaling result
        """
        pass 

class AbstractOutcomeRewardModel(ABC):
    """abstract base class for outcome reward models"""

    @abstractmethod
    def score(self, prompt: str, response: str) -> float:
        """the score for a given prompt and response"""
        pass

# TODO(GX) deal with aggregation of PRM scores somehow in a common place, e.g. here
class AbstractProcessRewardModel(ABC):
    """abstract base class for process reward models"""

    @abstractmethod
    def score(self, prompt: str, steps: List[str]) -> List[float]:
        """the score for a given prompt and steps"""
        pass