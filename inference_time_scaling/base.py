from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

# define a protocol for the llm type
class LLMProtocol(Protocol):
    def __call__(self, prompt: str) -> str:
        ...

T = TypeVar('T')

class AbstractScalingMethod(ABC):
    """abstract base class for inference-time scaling methods"""
    
    @abstractmethod
    def inference(self, llm: LLMProtocol, prompt: str, budget: int) -> str:
        """
        run inference with the given llm and prompt under the specified budget
        
        Args:
            llm: a callable that takes a prompt and returns a string
            prompt: the input prompt
            budget: the computational budget for inference
            
        Returns:
            the generated output string
        """
        pass 