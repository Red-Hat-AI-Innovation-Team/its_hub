from typing import Union
from abc import ABC, abstractmethod


class AbstractLanguageModel(ABC):
    """abstract base class for (autoregressive) language models"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        generate a response from the model
        
        Args:
            prompt: the input prompt
            
        Returns:
            the generated output string
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