"""
A Python library for inference-time scaling LLMs
"""

from importlib.metadata import version

__version__ = version("its_hub")

# Core abstractions - always available
from .algorithms.bon import BestOfN

# Core algorithms - always available
from .algorithms.self_consistency import SelfConsistency
from .base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)

# Start with core exports
__all__ = [  # noqa: RUF022
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
]

# Optional LM implementations - only available if [lm] extra is installed
try:
    from .lms import OpenAICompatibleLanguageModel, StepGeneration
    from .reward_models import LLMJudge

    __all__.extend(["LLMJudge", "OpenAICompatibleLanguageModel", "StepGeneration"])
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
