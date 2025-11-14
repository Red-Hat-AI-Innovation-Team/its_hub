"""
A Python library for inference-time scaling LLMs
"""

from importlib.metadata import version

__version__ = version("its_hub")

# Core abstractions - always available
from .base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    AbstractScalingResult,
)

# Core algorithms - always available
from .algorithms.self_consistency import SelfConsistency
from .algorithms.bon import BestOfN

# Dummy reward model - always available
from .reward_models import DummyRewardModel

# Start with core exports
__all__ = [
    # Version
    "__version__",
    # Abstractions
    "AbstractLanguageModel",
    "AbstractScalingAlgorithm",
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
    "AbstractScalingResult",
    # Algorithms
    "SelfConsistency",
    "BestOfN",
    # Reward Models
    "DummyRewardModel",
]

# Optional LM implementations - only available if [lm] extra is installed
try:
    from .lms import OpenAICompatibleLanguageModel, StepGeneration
    __all__.extend(["OpenAICompatibleLanguageModel", "StepGeneration"])
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
