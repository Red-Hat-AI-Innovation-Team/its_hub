# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

## 📚 Documentation

For comprehensive documentation, including installation guides, tutorials, and API reference, visit:

**[https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**

## Quick Start

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize language model (requires vLLM server)
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1", 
    api_key="NO_API_KEY", 
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
)

# Set up inference-time scaling
sg = StepGeneration("\n\n", 32, r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B", 
    device="cuda:0", 
    aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

# Solve with inference-time scaling
result = scaling_alg.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=8)
```

## Installation

```bash
# Production
pip install its_hub

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **OpenAI-Compatible API**: Easy integration with existing applications
- 🧮 **Math-Optimized**: Built for mathematical reasoning with specialized prompts
- 📊 **Benchmarking Tools**: Compare algorithms on MATH500 and AIME-2024 datasets
- ⚡ **Async Support**: Concurrent generation with limits and error handling
- 🌐 **Envoy Gateway** [Experimental]: Transparent ITS integration via Envoy External Processor

## Envoy Gateway Integration 

The Envoy External Processor (ext_proc) provides a transparent gateway for applying inference-time scaling to OpenAI API compatible inference API. Deploy it in front of your existing LLM infrastructure with minimal code changes.

### Architecture

```
Client → Envoy (port 8108) → ext_proc gRPC (port 50051) → LLM API
```

### Quick Start with Envoy Locally

**Prerequisites:**
- [Envoy proxy](https://www.envoyproxy.io/docs/envoy/latest/start/install) installed
- An OpenAI-compatible LLM API endpoint
- `LLM_API_KEY` environment variable to authenticate with LLM APIs

**Step 1: Setup**

```bash
# Clone and install dependencies
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
just setup

# setup grpc dependencies
just submodule-init
just proto-compile
```

**Step 2: Start Services (in separate terminals)**

Make sure Envoy is installed, see [Envoy Install](https://www.envoyproxy.io/docs/envoy/latest/start/install).

```bash
# Terminal 1: Start Envoy proxy
just envoy-start

# Terminal 2: Start ext_proc gRPC service
just envoy-grpc-start
```

**Step 3: Test with ITS**
```bash
# Use ITS with self-consistency (budget=3)
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ITS-Budget: 3" \
  -H "X-ITS-Endpoint: https://api.openai.com/v1" \
  -H "X-ITS-API-Key: $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

**Without ITS (standard pass-through):**
```bash
# Omit ITS headers for normal processing
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### ITS Header API

Control inference-time scaling via HTTP headers:

| Header | Required | Description | Example |
|--------|----------|-------------|---------|
| `X-ITS-Budget` | Yes | Number of LLM calls (1-1000) | `3` |
| `X-ITS-Endpoint` | Yes | Full LLM API base URL | `https://api.openai.com/v1` |
| `X-ITS-API-Key` | No | API key for authentication | `sk-...` |

**How it works:**
1. Requests **with** ITS headers → ext_proc applies self-consistency algorithm
2. Requests **without** ITS headers → pass through to upstream LLM unchanged
3. On errors → fail-safe pass-through to upstream

### Testing & Monitoring

```bash
# Run test suite
just envoy-grpc-test

# Check Envoy cluster health
just envoy-health

# View Envoy admin interface
open http://localhost:9901
```

### Configuration

The Envoy configuration is located at `config/envoy/ext_proc.yaml`. Key settings:

- **Client listener**: Port 8108
- **ext_proc gRPC**: Port 50051
- **LLM upstream**: Port 8100 (configurable)
- **Admin interface**: Port 9901
- **Timeouts**: 120s for ITS processing, 300s for upstream
- **Failure mode**: `allow` (fail-safe pass-through)

For detailed configuration options and troubleshooting, see `its_hub/integration/ext_proc/HANDOVER.md`.

## Development

### Prerequisites

Before setting up the development environment, please ensure you have the following tools installed:

- **git**: For version control and managing submodules. [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **just**: A command runner used to execute project-specific commands. [Installation Guide](https://github.com/casey/just#installation)
- **uv**: An extremely fast Python package installer and resolver. [Installation Guide](https://astral.sh/uv/install/)

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
pytest tests
```

### Submodule and Protobuf Management

This project uses Git submodules to manage third-party protobuf definitions. The generated Python stub files are build artifacts and are not committed to the repository.

**Initial Setup for Developers:**

To set up your development environment, including initializing submodules and compiling protobufs, use the `just setup` command:

```bash
just setup
```

This command performs the following steps:
1.  Installs Python dependencies.
2.  Initializes and updates all Git submodules to the versions recorded in this repository.
3.  Compiles the `.proto` files from the submodules into Python stub files.

**Updating Submodules:**

If you need to update the submodules to their latest upstream versions (e.g., to get new protobuf definitions), use the `just submodule-update` command:

```bash
just submodule-update
```

After running this command, the submodules will be updated to the latest commit on their tracked branches. You must then record these changes in the main repository:

```bash
git add third_party/envoy-data-plane-api third_party/xds third_party/protoc-gen-validate
git commit -m "Update submodules to latest upstream versions"
```

**Important**: The generated Python protobuf stub files (located in `its_hub/integration/ext_proc/proto/`) are intentionally excluded from version control via `.gitignore`. They should always be generated as part of the build process to ensure consistency with the `.proto` definitions.

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
