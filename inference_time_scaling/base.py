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

class AbstractScalingMethod(ABC):
    """abstract base class for inference-time scaling methods"""
    
    @abstractmethod
    def infer(self, lm: AbstractLanguageModel, prompt: str, budget: int) -> str:
        """
        run inference with the given language model and prompt under the specified budget
        
        Args:
            lm: a language model that takes a prompt and returns a response
            prompt: the input prompt
            budget: the computational budget for inference
            
        Returns:
            the generated output string
        """
        pass 