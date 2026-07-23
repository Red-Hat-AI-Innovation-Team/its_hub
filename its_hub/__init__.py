"""
A Python library for inference-time scaling LLMs
"""

from importlib.metadata import version

# Core - Algorithm implementations (always available)
from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.core.algorithms.adaptive_self_consistency import AdaptiveSelfConsistency
from its_hub.core.algorithms.bon import BestOfN
from its_hub.core.algorithms.self_consistency import SelfConsistency

# Optional - requires scipy (install with: pip install its_hub[experimental])
try:
    from its_hub.core.algorithms.beta_self_consistency import BetaSelfConsistency

    _has_beta = True
except ImportError:
    _has_beta = False

__version__ = version("its_hub")

# Start with core exports
__all__ = [  # noqa: RUF022
    # Version
    "__version__",
    # Abstractions
    "AbstractLanguageModel",
    "AbstractOrchestrator",
    "AbstractScalingAlgorithm",
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
    "AbstractScalingResult",
    # Algorithms
    "AdaptiveSelfConsistency",
    "SelfConsistency",
    "BestOfN",
]

if _has_beta:
    __all__.append("BetaSelfConsistency")

# Optional LM implementations - only available if [lm] extra is installed
try:
    from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
    from its_hub.core.lms.step_generation import StepGeneration
    from its_hub.core.orchestrator import LMOrchestrator
    from its_hub.core.reward_models.llm_judge import LLMJudge

    __all__.extend(
        [
            "LLMJudge",
            "LMOrchestrator",
            "OpenAICompatibleLanguageModel",
            "StepGeneration",
        ]
    )
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
