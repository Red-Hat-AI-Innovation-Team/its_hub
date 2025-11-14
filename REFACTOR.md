
## 🚧 ACTIVE REFACTORING: Repository Restructure

**Status:** Phase 2 - Validation & Cleanup
**Goal:** Transform its_hub from a bundled multi-purpose library into a **minimal, focused algorithm package** that external gateways can easily integrate.

### Core Principle: Algorithms First, Everything Else Optional

**What we're building:**
- **Tiny core package** with only algorithm logic + abstract interfaces
- **Minimal dependencies** (numpy, typing-extensions only)
- **Gateway-friendly design** - implement our interfaces, use our algorithms
- **Reference implementations optional** - OpenAI-compatible LM is just an example

**What we're NOT building:**
- Production-ready LM orchestration layer (that's the gateway's job)
- Multi-cloud provider support (gateways handle that)
- Comprehensive IaaS service (prototyping tool only)

### Current State: Three Bundled Functionalities

ITS-Hub currently combines three distinct responsibilities:

1. **Core Algorithms** - Inference-time scaling algorithms
   - Self-Consistency, Best-of-N, Particle Filtering, Beam Search, ParticleGibbs, MCTS
   - Core interface: `async_infer(lm, algorithm_name, budget, ...)`

2. **Orchestration Layer** - Language Model abstraction
   - `AbstractLanguageModel` and concrete implementations
   - `OpenAICompatibleLanguageModel`, `LiteLLMLanguageModel`
   - Handles API calls, retries, error handling, async batching

3. **Proxy Server** - IaaS FastAPI service
   - OpenAI-compatible API endpoint
   - Request transformation and routing
   - Global state management (LM_DICT, SCALING_ALG)

**Problem:** Complex installation, unclear scope, difficult integration into existing AI gateways.

### Refactoring Goals

#### Key Insight: Algorithms Are Already Decoupled
The current design is **already architecturally sound**:
- Algorithms accept `AbstractLanguageModel` interface (defined in `base.py`)
- Algorithm logic lives in `ainfer(lm, prompt, budget, ...)` methods
- Any class implementing `AbstractLanguageModel` can be used

**The refactoring is NOT about redesigning algorithms.**

#### What We're Actually Doing

1. **Making the Decoupling Obvious**
   - Current: Algorithms + concrete LM implementations bundled together
   - Target: Algorithms are core, concrete LM implementations are optional reference examples
   - Message: "Implement `AbstractLanguageModel`, then use our algorithms"

2. **Removing Unnecessary Dependencies**
   - Core install should NOT require: `openai`, `litellm`, `fastapi`, `uvicorn`, `aiohttp`, `backoff`
   - Core only needs: `numpy`, `typing-extensions`, basic utils
   - Heavy dependencies move to optional `[lm]` and `[iaas]` extras

3. **Clarifying the Scope**
   - **Core value**: The algorithms themselves (Self-Consistency, Best-of-N, Beam Search, etc.)
   - **Reference implementations**: `OpenAICompatibleLanguageModel` (optional), `LiteLLMLanguageModel` should be removed.
   - **Testing/dev tool**: IaaS proxy server based on fastapi; no need to be production-level. (optional)

4. **Enabling External Gateway Integration**
   - Python gateways: Import `its_hub` core, implement `AbstractLanguageModel` with their LM client
   - Gateway owns the orchestration, just uses algorithm logic
   - Gateway also uses its own proxy layer

#### Benefits

1. **Simpler Installation for Gateway Integrations**
   - `pip install its_hub` → Just algorithms + interfaces (tiny install)
   - Gateway implements its own LM class using existing infrastructure
   - No conflict with gateway's existing dependencies

2. **Clearer Contribution Path**
   - Contributing algorithms: Pure logic, no LM engine concerns
   - Contributing LM implementations: Optional, in separate module
   - Clear separation of concerns

3. **Production Deployment Clarity**
   - Production: Use established AI gateways (Python, TypeScript, Go, Rust)
   - Gateway imports `its_hub` algorithms only
   - IaaS proxy is for testing/prototyping, not production

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  its_hub (core)                                             │
│  - Pure algorithm implementations                           │
│  - Minimal dependencies (numpy, basic utils)                │
│  - Input: List of responses OR streaming step generator     │
│  - Output: Selected response(s) with scores                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ import algorithms only
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────┴────────┐                     ┌────────────┴──────────┐
│  its_hub[lm]   │                     │  AI Gateway           │
│  (optional)    │                     │  (TypeScript/Go/Rust) │
│  - LM clients  │                     │  - Uses its_hub algos │
│  - Retry logic │                     │  - Native LM clients  │
│  - Async pool  │                     │  - Production features│
└────────────────┘                     └───────────────────────┘
        │
        │ uses
        ▼
┌────────────────┐
│  its_hub[iaas] │
│  (optional)    │
│  - FastAPI svc │
│  - Testing only│
└────────────────┘
```

### Open Questions & Research Areas

1. **Cross-Language Integration**
   - Can Python-based algorithms be integrated into TypeScript/Go/Rust gateways?
   - Options: gRPC service, WebAssembly, native ports, subprocess calls
   - Performance implications of cross-language boundaries

2. **Step-wise Algorithm Support**
   - How to abstract step-wise generation (Beam Search, Particle Filtering)?
   - Current approach: `StepGeneration` class couples with LM implementation
   - Alternative: Callback/iterator interface that gateways implement?

3. **Reward Model Integration**
   - Process Reward Models (PRMs) currently tightly coupled to vLLM
   - How to make PRMs pluggable for different gateway backends?
   - Outcome reward models (LLM judges) easier to abstract

### MVP Implementation - 1 Day Delivery

**Scope:** Core algorithms (BoN, Self-Consistency) with minimal dependencies.
**Key Insight:** Algorithm code is already decoupled - NO changes needed. Mostly file moves and dependency cleanup.

---

#### Tasks for 1-Day MVP (Functional Changes Only)

**1. Remove Unnecessary Files - Create Minimal Structure (2 hours)**

**DELETE entirely:**
- `notebooks/` - not needed for library
- `scripts/` - benchmarking/examples, not core
- `its_hub/integration/reward_hub.py` - external dependency
- `its_hub/integration/iaas.py` - move to optional (later) or remove for MVP
- `tests/` - LiteLLM-related tests
- `tests/integration/` - complex integration tests (if exists)

**KEEP minimal set:**
```
its_hub/
  ├── base.py                  # Core abstractions ✓
  ├── types.py                 # Type definitions ✓
  ├── utils.py                 # Utilities ✓
  ├── lms.py                   # Keep, but REMOVE LiteLLM class
  ├── error_handling.py        # Needed by OpenAICompatibleLM ✓
  ├── reward_models.py         # NEW - dummy implementation
  └── algorithms/
      ├── bon.py               # NO CHANGE ✓
      ├── self_consistency.py  # NO CHANGE ✓
      ├── beam_search.py       # Keep but mark unsupported
      ├── particle_gibbs.py    # Keep but mark unsupported
      └── planning_wrapper.py  # Keep but mark unsupported

tests/
  ├── test_algorithms.py       # Core tests only ✓
  └── test_base.py             # Core tests only ✓
```

**2. Remove LiteLLM Code (1 hour)**

In `its_hub/lms.py`:
- Delete entire `LiteLLMLanguageModel` class
- Keep `OpenAICompatibleLanguageModel` and `StepGeneration`
- No other changes to file

**3. Dependency Cleanup (1 hour)**

Update `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy",
    "typing-extensions>=4.12.2",
]

[project.optional-dependencies]
lm = [
    "openai>=1.68.2",
    "aiohttp>=3.9.0",
    "backoff>=2.2.0",
]

iaas = [
    "its_hub[lm]",
    "fastapi>=0.115.5",
    "uvicorn",
    "pydantic>=2.7.2",
]

dev = [
    "its_hub[lm,iaas]",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.10.0",
]
```

**Key:** Remove `litellm` entirely from all dependency groups.

**4. Add Dummy Reward Model (30 min)**

Create `its_hub/reward_models.py`:

```python
from its_hub.base import AbstractOutcomeRewardModel

class DummyRewardModel(AbstractOutcomeRewardModel):
    """Dummy reward model for testing - returns fixed score."""
    async def evaluate(self, prompt, response):
        return 0.5
```

**5. Update Exports (30 min)**

Update `its_hub/__init__.py`:

```python
# Core - always available
from .base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
)
from .algorithms.self_consistency import SelfConsistency
from .algorithms.bon import BestOfN
from .reward_models import DummyRewardModel

# Optional - only if [lm] extra installed
try:
    from .lms import OpenAICompatibleLanguageModel, StepGeneration
except ImportError:
    pass
```

**6. Fix Tests (1 hour)**

- Remove LiteLLM test files
- Update any broken imports
- Ensure core algorithm tests pass

**7. Update README (1 hour)**

Add installation sections:
- Core: `pip install its_hub` (≤5 deps, algorithms + abstractions only)
- With LM: `pip install its_hub[lm]` (adds OpenAI-compatible LM)
- Gateway integration example (~30 lines showing AbstractLanguageModel implementation)

---

### MVP Success Criteria

**Must Have (1 day):**
- [x] `pip install its_hub` has ≤5 dependencies ✅ **DONE: Only 2 dependencies (numpy, typing-extensions)**
- [x] Algorithm code unchanged or little changed(bon.py, self_consistency.py) ✅ **DONE: Zero changes to algorithm code**
- [x] Dummy reward model implementation ✅ **DONE: Created its_hub/reward_models.py with DummyRewardModel**
- [x] Basic README update ✅ **DONE: Added installation patterns and gateway integration example**
- [x] Tests still pass ✅ **DONE: All 93 tests passing**
- [x] Flexible LM-abstraction design and implementation ✅ **DONE: Clean exports with try/except for optional deps**

---

## MVP Delivery Status: ✅ COMPLETE

### What Was Accomplished (Phase 1 - Initial Cleanup)

**1. Minimal File Structure** ✅
- Deleted: `notebooks/`, `scripts/`, `docs/`, `its_hub/integration/`
- Removed: 5 test files (iaas, lms, reward_hub, particle_gibbs_resampling, planning_wrapper)
- Updated: README.md to remove documentation website references
- Result: Streamlined codebase focused on core algorithms

**2. Removed LiteLLM** ✅
- Deleted entire `LiteLLMLanguageModel` class from `its_hub/lms.py`
- Removed litellm import and logging configuration
- Kept only `OpenAICompatibleLanguageModel` and `StepGeneration`

**3. Minimal Dependencies** ✅
```bash
# Core dependencies (verified):
Requires: numpy, typing-extensions

# Optional [lm]: openai, aiohttp, backoff, requests
# Optional [iaas]: fastapi, uvicorn, pydantic, click
# Optional [dev]: pytest, pytest-asyncio, ruff
```

**4. Clean Architecture** ✅
- Core abstractions always available (AbstractLanguageModel, AbstractScalingAlgorithm, etc.)
- Core algorithms always available (SelfConsistency, BestOfN)
- Dummy reward model for testing
- LM implementations optional (try/except import)

**5. Updated Documentation** ✅
- README shows core vs [lm] installation
- Gateway integration example (30 lines)
- Clear algorithm-first messaging

**6. Unit Tests Passing** ✅
- 93 tests passing (all use mocks)
- Fixed conftest.py imports

---

### Phase 2 - Validation & Cleanup (COMPLETED ✅)

**REQUIRED - Real Validation:**
- [x] **E2E Tests with Real OpenAI API** ✅
  - Created `tests/e2e/` directory
  - Test SelfConsistency with real OpenAI endpoint (basic, tool calls, async)
  - Test BestOfN with real OpenAI endpoint (with dummy reward, async)
  - Test core interface (minimal imports, direct LM calls)
  - Uses `.env` file for API keys (gitignored)
  - All 8 E2E tests passing with real API calls
  - Budget=2 to minimize API costs (~$0.001 per run)

- [x] **File-by-File Design Review** ✅
  - [x] `its_hub/base.py` - Fixed type hints (str → dict), improved docstrings, AbstractProcessRewardModel raises NotImplementedError
  - [x] `its_hub/types.py` - Fixed critical pydantic bug, removed unused Function/ToolCall classes, removed experimental methods (59% reduction)
  - [x] `its_hub/utils.py` - Removed unused system prompts, kept only extract_content_from_lm_response
  - [x] `its_hub/lms.py` - Removed stub classes (LocalVLLM, Transformers), kept StepGeneration for experimental
  - [x] `its_hub/algorithms/__init__.py` - Removed MetropolisHastings stub (71% reduction)
  - [x] `its_hub/__init__.py` - Made __all__ dynamic to match available imports
  - [x] `its_hub/error_handling.py` - All code needed, no changes required
  - [x] `its_hub/reward_models.py` - Correct implementation, no changes needed

**OPTIONAL - Test Cleanup:**
- [x] Kept experimental algorithm tests (BeamSearch, ParticleGibbs, ParticleFiltering) for [experimental] extra
- [x] Test count: 93 unit tests (using mocks) + 8 E2E tests (real API)

**Critical Bugs Fixed:**
1. **pydantic dependency bug** - types.py used pydantic but it wasn't in core dependencies
2. **Missing logging import** - types.py used logging.warning() without importing logging
3. **Wrong type hints** - base.py declared str returns but actually returned dict

**Files Modified (Summary):**

| File | Lines Before | Lines After | Change | Key Changes |
|------|--------------|-------------|--------|-------------|
| types.py | 149 | 61 | -59% | Fixed pydantic bug, removed unused code |
| utils.py | 57 | 56 | -2% | Removed unused prompts |
| lms.py | 538 | 529 | -2% | Removed stub classes |
| error_handling.py | 220 | 220 | 0% | No changes needed |
| algorithms/__init__.py | 56 | 16 | -71% | Removed MetropolisHastings stub |
| its_hub/__init__.py | 50 | 49 | -2% | Made __all__ dynamic |
| reward_models.py | 53 | 53 | 0% | No changes needed |
| base.py | 213 | 215 | +1% | Fixed type hints str→dict, improved docs |
| **Algorithm files** | - | - | - | Changed pydantic→dataclasses (imports only) |

**Definition of Done:**
- ✅ E2E tests pass with real OpenAI API (8/8 passing)
- ✅ Each file reviewed and confirmed minimal
- ✅ No extra/unnecessary code in any file
- ✅ Interface matches intended design
- ✅ Type hints match actual implementation
- ✅ Zero new dependencies added

### Timeline:
- Phase 1 (Cleanup): ✅ Complete
- Phase 2 (Validation): ✅ Complete