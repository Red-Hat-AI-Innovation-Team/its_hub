# Installation

## Prerequisites

- Python 3.11+
- pip or uv package manager
- Rust toolchain (only for development/source builds — install via [rustup](https://rustup.rs/))
- GPU with CUDA 11.8+ (only for `[experimental]` installation)

## Installation Options

| Option | Command | Use Case |
|--------|---------|----------|
| **Core** | `pip install its_hub` | Algorithms and interfaces only (2 dependencies) |
| **LM** | `pip install its_hub[lm]` | OpenAI-compatible LM, LLMJudge, StepGeneration |
| **IaaS** | `pip install its_hub[iaas]` | FastAPI Inference-as-a-Service server |
| **ext_proc** | `pip install its_hub[ext_proc]` | Envoy external processor gateway (gRPC + protobuf) |
| **envoy-iaas** | `pip install its_hub[envoy-iaas]` | Full Envoy + IaaS stack (iaas + ext_proc combined) |
| **Experimental** | `pip install its_hub[experimental]` | Particle Filtering, Beam Search, reward-hub integration |
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
from its_hub import OpenAICompatibleLanguageModel, StepGeneration
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

## Research Installation

```bash
pip install its_hub[research]
```

**Adds**: `math-verify`, `datasets`, `matplotlib`
**Use if**: Running benchmarks on MATH500/AIME or evaluating algorithm performance
**Includes**: Benchmark scripts in `scripts/benchmark.py`

---

## Development Installation

Requires a Rust toolchain (the build backend is [maturin](https://www.maturin.rs/), which compiles the native extension automatically).

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Recommended: uv
uv sync --extra dev

# Alternative: pip
pip install -e ".[dev]"
```

**Includes**: All core + experimental + `pytest`, `ruff`, `jupyter`, notebooks
**Use if**: Contributing, testing, or developing new features

> **Rust rebuilds are automatic with uv.** The `[tool.uv] cache-keys` config in `pyproject.toml` tracks `.rs` sources and `Cargo.toml`, so `uv sync` rebuilds the native extension whenever Rust code changes — no separate `maturin develop` step required.

```bash
# Run tests
uv run pytest tests/
uv run pytest tests/ --cov=its_hub

# Code quality
uv run ruff check its_hub/ --fix
uv run ruff format its_hub/
```

---

## Gateway Installation

The gateway extras provide two deployment modes for inference-time scaling. Both require
the `lm` extra as a base dependency.

### Standalone IaaS (FastAPI)

```bash
pip install its_hub[iaas]
```

Provides the `its-iaas` CLI command to run a standalone OpenAI-compatible API server with
ITS. No Envoy or gRPC dependencies.

```python
# Verify
from its_hub.integration.iaas.app import app
print("IaaS OK")
```

See [IaaS Service Guide](iaas-service.md) for configuration and usage.

### Envoy ext_proc Gateway

```bash
pip install its_hub[ext_proc]
```

Provides the `envoy-grpc` CLI command for the Envoy external processor. Requires compiled
proto files and an Envoy proxy.

**Additional prerequisites**: git, make, [Envoy proxy](https://www.envoyproxy.io/docs/envoy/latest/start/install)

```bash
# Initialize submodules and compile proto files
make setup-envoy

# Verify
python -c "from its_hub.integration.ext_proc.processor import ExternalProcessor; print('ext_proc OK')"
```

See [ext_proc Gateway Guide](ext-proc-gateway.md) for configuration and usage.

### Envoy + IaaS Combined

```bash
pip install its_hub[envoy-iaas]
```

Installs both `iaas` and `ext_proc` extras for the combined deployment where Envoy routes
`X-ITS-*` header requests to the IaaS backend via an ext_proc filter. Provides three CLI
commands: `its-iaas`, `its-iaas-ext-proc`, and `envoy-grpc`.

```bash
make setup-envoy  # Still needed for proto files

# Start the full stack
make envoy-iaas-stack
```

See the [Envoy Integration](iaas-service.md#envoy-integration) section in the IaaS guide.

### Proto Generation

Both `ext_proc` and `envoy-iaas` require compiled Envoy proto files. The `make setup-envoy`
target handles this automatically:

1. Initializes git submodules (`envoy-data-plane-api`, `xds`, `protoc-gen-validate`)
2. Compiles `.proto` files to Python using `grpc_tools.protoc`
3. Outputs to `its_hub/integration/proto/`

To recompile after proto changes:

```bash
make proto-clean    # Remove generated files
make proto-compile  # Recompile
```

To restore submodules to pinned commits (after a `git pull` updates `.gitmodules`):

```bash
make upgrade-protos
```

---

## Combining Extras

```bash
pip install its_hub[experimental,research]  # Experimental + benchmarking
pip install its_hub[lm,research]            # LM + benchmarking
pip install -e ".[dev,research]"            # Everything
```

---

## Verification

```bash
# Core
python -c "from its_hub import BestOfN, SelfConsistency; print('Core OK')"

# LM
python -c "from its_hub import OpenAICompatibleLanguageModel, LLMJudge; print('LM OK')"

# Experimental
python -c "from its_hub.core.algorithms.particle_gibbs import ParticleFiltering; print('Experimental OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Next Steps

- [Quick Start Guide](quick-start.md) - Best-of-N and Particle Filtering examples
- [IaaS Service Guide](iaas-service.md) - Deploy as OpenAI-compatible API
- [Development Guide](development.md) - Contributing guidelines

For runtime issues (CUDA OOM, server errors, etc.), see the troubleshooting sections in the Quick Start or IaaS Service guides.