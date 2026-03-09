# Breaking Changes - Directory Restructure (api/core)

This document outlines breaking changes introduced by the api/core directory restructure and provides migration guidance.

## Summary

The its_hub library has been restructured to separate public API from core implementation. This improves maintainability and makes it easier to understand what's part of the stable public API.

## Breaking Changes

### 1. Import Paths Changed

All import paths have changed due to the api/core separation. Old imports will raise `ImportError`.

### 2. Reward-Hub Integration Removed from Core

`its_hub.integration` directory has been removed. LocalVllmProcessRewardModel is now in `examples/local_vllm_prm.py` as sample code.

### 3. LLMJudgeRewardModel Renamed

The `LLMJudgeRewardModel` class has been renamed to `LLMJudge` and simplified to reuse the language model instance.

## Migration Guide

### Quick Reference Table

| Old Import | New Import | Notes |
|------------|-----------|-------|
| `from its_hub.lms import OpenAICompatibleLanguageModel` | `from its_hub import OpenAICompatibleLanguageModel` | Top-level export |
| `from its_hub.algorithms import SelfConsistency` | `from its_hub import SelfConsistency` | Top-level export |
| `from its_hub.algorithms import BestOfN` | `from its_hub import BestOfN` | Top-level export |
| `from its_hub.lms import StepGeneration` | `from its_hub import StepGeneration` | Top-level export |
| `from its_hub.types import ChatMessage, ChatMessages` | `from its_hub.api import ChatMessage, ChatMessages` | In API module |
| `from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT` | `from its_hub.core.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT` | In core.utils |
| `from its_hub.algorithms import ParticleFiltering` | `from its_hub.core.algorithms.particle_gibbs import ParticleFiltering` | Not in top-level |
| `from its_hub.algorithms import BeamSearch` | `from its_hub.core.algorithms.beam_search import BeamSearch` | Not in top-level |
| `from its_hub.integration.reward_hub import LocalVllmProcessRewardModel` | `from local_vllm_prm import LocalVllmProcessRewardModel` | Now in examples/ |
| `from its_hub.integration.reward_hub import LLMJudgeRewardModel` | `from its_hub import LLMJudge` | Renamed and simplified |

## Code Examples

### Example 1: Self-Consistency (Core Algorithm)

**Before:**
```python
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import SelfConsistency
from its_hub.types import ChatMessage, ChatMessages

lm = OpenAICompatibleLanguageModel(...)
sc = SelfConsistency()
result = sc.infer(lm, "Your prompt", budget=5)
```

**After:**
```python
from its_hub import OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

lm = OpenAICompatibleLanguageModel(...)
sc = SelfConsistency()
result = sc.infer(lm, "Your prompt", budget=5)
```

### Example 2: Best-of-N with LLM Judge

**Before:**
```python
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel

lm = OpenAICompatibleLanguageModel(...)
judge = LLMJudgeRewardModel(
    model="gpt-4o-mini",
    criterion="overall_quality",
    api_key="...",
)
bon = BestOfN(judge)
result = bon.infer(lm, "Your prompt", budget=4)
```

**After:**
```python
from its_hub import OpenAICompatibleLanguageModel, BestOfN, LLMJudge

lm = OpenAICompatibleLanguageModel(...)
judge = LLMJudge(lm=lm)
bon = BestOfN(judge)
result = bon.infer(lm, "Your prompt", budget=4)
```

### Example 3: Particle Filtering (Experimental)

**Before:**
```python
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

lm = OpenAICompatibleLanguageModel(...)
sg = StepGeneration(step_token="\n\n", max_steps=32)
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)
pf = ParticleFiltering(sg, prm)
result = pf.infer(lm, "Your prompt", budget=8)
```

**After:**
```python
from its_hub import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

# NOTE: sample prm implementation is located in examples/local_vllm_prm.py
from local_vllm_prm import LocalVllmProcessRewardModel

lm = OpenAICompatibleLanguageModel(...)
sg = StepGeneration(step_token="\n\n", max_steps=32)
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)
pf = ParticleFiltering(sg, prm)
result = pf.infer(lm, "Your prompt", budget=8)
```

**Note**: For particle filtering and beam search examples, see `examples/test_math_example.py` for complete working code.

### Example 4: Beam Search (Experimental)

**Before:**
```python
from its_hub.algorithms import BeamSearch
from its_hub.lms import StepGeneration
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

sg = StepGeneration(step_token="\n\n", max_steps=50)
prm = LocalVllmProcessRewardModel(...)
bs = BeamSearch(sg, prm, beam_width=4)
```

**After:**
```python
from its_hub import StepGeneration
from its_hub.core.algorithms.beam_search import BeamSearch

from local_vllm_prm import LocalVllmProcessRewardModel

sg = StepGeneration(step_token="\n\n", max_steps=50)
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)
bs = BeamSearch(sg, prm, beam_width=4)
```

## Installation Changes

### Package Extras Renamed

**Before:**
```bash
pip install its_hub[prm]  # Process reward models
```

**After:**
```bash
pip install its_hub[experimental]  # Experimental features including reward-hub
```

### What's Included in Each Extra

| Extra | Included | Use Case |
|-------|----------|----------|
| **Core** (no extra) | SelfConsistency, BestOfN | Core algorithm implementations |
| **[lm]** | LLMJudge, OpenAI-compatible LM | Required for OpenAI-compatible LM support |
| **[experimental]** | reward-hub, transformers | ParticleFiltering, BeamSearch, process reward models |
| **[dev]** | pytest, ruff, jupyter | Development and testing |
| **[research]** | math-verify, datasets | Benchmarking and evaluation |

## API Design Changes

### API vs Core Modules

- **its_hub.api**: Abstract interfaces, types, and errors (public API)
- **its_hub.core**: Concrete implementations (internal, may change)

**Recommendation**: Import from top-level (`from its_hub import ...`) when possible. Fall back to `its_hub.core` only for algorithms not in top-level exports.

## Experimental Features

### What's Experimental?

Features in `its_hub.core.algorithms` that require reward-hub integration:
- ParticleFiltering
- BeamSearch
- EntropicParticleFiltering

### How to Use Experimental Features

1. Install experimental extra:
   ```bash
   pip install its_hub[experimental]
   ```

2. Copy `examples/local_vllm_prm.py` to your project or add examples directory to Python path.

3. Import experimental algorithms from core:
   ```python
   from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
   ```

### Why Are These Experimental?

- Require external reward-hub library
- Not part of minimal viable product (MVP)
- API may change based on feedback
- Kept separate to maintain core library simplicity

## Troubleshooting

### Import Error: No module named 'its_hub.lms'

**Problem**: Using old import paths.
**Solution**: Update imports following the migration guide above.

### Import Error: cannot import name 'LLMJudgeRewardModel'

**Problem**: Class was renamed to `LLMJudge`.
**Solution**: Use `from its_hub import LLMJudge`.

### Import Error: No module named 'local_vllm_prm'

**Problem**: local_vllm_prm.py is not in Python path.
**Solution**: Either:
1. Copy `examples/local_vllm_prm.py` to your project directory
2. Add examples directory to Python path
3. Run your script from the examples directory

### Import Error: No module named 'its_hub.integration'

**Problem**: The integration directory was removed.
**Solution**: Refer to or use `local_vllm_prm` module from examples directory as a reference.

## Support

For issues or questions:
- Check the examples in `examples/` directory
- Review `docs/quick-start.md` for updated examples
- Open a GitHub issue
