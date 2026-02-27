"""
A Python library for inference-time scaling LLMs
"""

from importlib.metadata import version

__version__ = version("its_hub")

# Core - Algorithm implementations (always available)
from its_hub.api import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.core.algorithms.bon import BestOfN
from its_hub.core.algorithms.self_consistency import SelfConsistency

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
    from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
    from its_hub.core.lms.step_generation import StepGeneration
    from its_hub.core.reward_models.llm_judge import LLMJudge

    __all__.extend(["LLMJudge", "OpenAICompatibleLanguageModel", "StepGeneration"])
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
