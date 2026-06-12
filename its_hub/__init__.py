"""
A Python library for inference-time scaling LLMs with particle filtering
"""

from importlib.metadata import version

# Core - abstractions and algorithm implementations (always available)
from its_hub.api import (
    AbstractLanguageModel,
    AbstractOrchestrator,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.core.algorithms.particle_filtering import (
    EntropicParticleFiltering,
    ParticleFiltering,
    ParticleFilteringResult,
)
from its_hub.core.lms.step_generation import StepGeneration
from its_hub.core.orchestrator import LMOrchestrator

__version__ = version("its_hub")

# Start with core exports
__all__ = [  # noqa: RUF022
    # Version
    "__version__",
    # Abstractions
    "AbstractLanguageModel",
    "AbstractOrchestrator",
    "AbstractScalingAlgorithm",
    "AbstractScalingResult",
    # Algorithms
    "ParticleFiltering",
    "EntropicParticleFiltering",
    "ParticleFilteringResult",
    # Step generation and orchestration
    "StepGeneration",
    "LMOrchestrator",
]

# Optional LM implementations - only available if [lm] extra is installed
try:
    from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel

    __all__.extend(
        [
            "OpenAICompatibleLanguageModel",
        ]
    )
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
