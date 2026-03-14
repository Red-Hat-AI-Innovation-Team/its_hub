# Installation

## Prerequisites

- Python 3.10+ (3.11+ recommended)
- pip or uv package manager
- GPU with CUDA 11.8+ (only for `[prm]` installation)

## Installation Options

| Option | Command | Use Case |
|--------|---------|----------|
| **Core** | `pip install its_hub` | Best-of-N, Self-Consistency, cloud APIs |
| **Experimental** | `pip install its_hub[experimental]` | Particle Filtering, Beam Search, reward-hub integration |
| **Cloud** | `pip install its_hub[cloud]` | AWS Bedrock, Google Vertex AI |
| **Research** | `pip install its_hub[research]` | Benchmarks, evaluation tools |
| **Dev** | `pip install -e ".[dev]"` | Contributing, testing |

---

## Core + LM Installation

```bash
pip install its_hub[lm]
```

### What's Included

**Algorithms**: Best-of-N, Self-Consistency, LLM Judge
**Language Models**: OpenAI-compatible

### When to Use

**Use if**: Working with cloud APIs (OpenAI, Anthropic, etc.), no GPU needed
**Skip if**: Need Particle Filtering/Beam Search or local process reward models

### Under the Hood

- **Size**: ~50MB (no vLLM or CUDA dependencies)
- **Installation time**: 1-2 minutes
- **GPU required**: No
- **What's excluded**: vLLM, local reward model inference

```python
# Verify installation
from its_hub import BestOfN, LLMJudge, SelfConsistency
```

---

## Experimental Installation (Reward-Hub Integration)

```bash
pip install its_hub[experimental]
```

### What's Added

**Algorithms**: Particle Filtering, Beam Search (+ all core algorithms)
**Reward Models**: `LocalVllmProcessRewardModel` for step-by-step scoring
**Additional Dependencies**: `reward-hub`, `transformers`

### When to Use

**Use if**: Need step-by-step reasoning with local reward models, have GPU
**Skip if**: Only using cloud APIs or outcome-based scoring

### Under the Hood

- **Size**: ~2-3GB (includes vLLM + CUDA dependencies)
- **Installation time**: 5-10 minutes
- **GPU required**: Yes (10-20GB VRAM for typical 7B reward models)
- **Version pinning**: `reward-hub[prm]` pins compatible vLLM + transformers + PyTorch versions

```python
# Verify installation
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.algorithms.beam_search import BeamSearch
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

# Check GPU
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
```

---

## Cloud Installation

```bash
pip install its_hub[cloud]
```

**Adds**: AWS Bedrock (`boto3`) and Google Vertex AI (`google-cloud-aiplatform`) SDKs
**Use if**: Need direct SDK access to Bedrock or Vertex AI (most cloud providers work with core via LiteLLM)

---

## Research Installation

```bash
pip install its_hub[research]
```

**Adds**: `math-verify`, `datasets`, `matplotlib`
**Use if**: Running benchmarks on MATH500/AIME or evaluating algorithm performance
**Includes**: Benchmark scripts in `scripts/benchmark.py`

---

## Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Recommended: uv
uv sync --extra dev

# Alternative: pip
pip install -e ".[dev]"
```

**Includes**: All core + experimental + `pytest`, `ruff`, `jupyter`, notebooks
**Use if**: Contributing, testing, or developing new features

```bash
# Run tests
uv run pytest tests/
uv run pytest tests/ --cov=its_hub

# Code quality
uv run ruff check its_hub/ --fix
uv run ruff format its_hub/
```

---

## Combining Extras

```bash
pip install its_hub[experimental,research]  # Experimental + benchmarking
pip install its_hub[cloud,research]          # Cloud + benchmarking
pip install -e ".[dev,research,cloud]"       # Everything
```

---

## Verification

```bash
# Core
python -c "from its_hub import BestOfN; print('✅ Core OK')"

# Experimental
python -c "from its_hub.core.algorithms.particle_gibbs import ParticleFiltering; print('✅ PRM OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Best-of-N and Particle Filtering examples
- [IaaS Service Guide](iaas-service.md) - Deploy as OpenAI-compatible API
- [Development Guide](development.md) - Contributing guidelines

For runtime issues (CUDA OOM, server errors, etc.), see the troubleshooting sections in the Quick Start or IaaS Service guides.