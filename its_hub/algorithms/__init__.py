import warnings

from its_hub.core.algorithms.bon import BestOfN, BestOfNResult
from its_hub.core.algorithms.self_consistency import SelfConsistency, SelfConsistencyResult

warnings.warn(
    "The algorithms module is deprecated and will be removed in a future version. "
    "The default implementations are now in the core module. Refer to docs/algorithms.md "
    "and BREAKING_CHANGES.md for more information.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BestOfN",
    "BestOfNResult",
    "SelfConsistency",
    "SelfConsistencyResult",
]

# Optional experimental algorithms - only available if [experimental] extra is installed
try:
    from its_hub.core.algorithms.beam_search import BeamSearch, BeamSearchResult
    from its_hub.core.algorithms.particle_gibbs import (
        EntropicParticleFiltering,
        ParticleFiltering,
        ParticleFilteringResult,
        ParticleGibbs,
        ParticleGibbsResult,
    )

    __all__.extend(
        [
            "BeamSearch",
            "BeamSearchResult",
            "EntropicParticleFiltering",
            "ParticleFiltering",
            "ParticleFilteringResult",
            "ParticleGibbs",
            "ParticleGibbsResult",
        ]
    )
except ImportError:
    # experimental algorithms not available - install with: pip install its_hub[experimental]
    pass
