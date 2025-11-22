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

### Using its_hub as a Python Library

**Recommended: Using uv**
```bash
# Install uv if you haven't already
# See: https://astral.sh/uv/install/

# Create virtual environment and install its_hub
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install its_hub
```

**Alternative: Using pip with venv**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install its_hub
pip install its_hub
```

### Contributing to its_hub (Algorithm Development)

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Using Make (recommended)
make setup

# Or manually with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using ITS With Envoy

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub

# Full setup including submodules and proto compilation
make setup-envoy
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

# One-command setup (initializes submodules + compiles protos)
make setup-envoy
```

**Step 2: Start Services**

Make sure Envoy is installed, see [Envoy Install](https://www.envoyproxy.io/docs/envoy/latest/start/install).

**Option A: Start Both Services Together (Recommended)**
```bash
# Starts Envoy proxy and gRPC service in parallel
make envoy-stack

# Logs are written to envoy.log and envoy-grpc.log
# Press Ctrl+C to stop both services
```

**Option B: Start Services Separately (in different terminals)**
```bash
# Terminal 1: Start Envoy proxy
make envoy-start

# Terminal 2: Start ext_proc gRPC service
make envoy-grpc
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
# Run test suite for ext_proc
make envoy-test

# Check Envoy cluster health
make envoy-health

# Stop services (if using envoy-stack)
make envoy-stack-stop

# View Envoy admin interface
open http://localhost:9901
```

### Configuration

The Envoy configuration is located at `config/envoy/ext_proc.yaml` with comprehensive inline documentation.

**Default Settings:**

- **Client listener**: Port 8108 (where you send requests)
- **ext_proc gRPC**: Port 50051 (ITS algorithm service)
- **LLM upstream**: Port 8100 (fallback LLM endpoint)
- **Admin interface**: Port 9901 (monitoring dashboard)
- **Timeouts**: 120s for ITS processing, 300s for upstream
- **Failure mode**: `allow` (fail-safe pass-through)

**Common Customizations:**

The config file includes detailed comments with `**CUSTOMIZE**` markers. Common changes:

1. **Change Envoy listening port** (default: 8108)
   ```yaml
   listeners[0].address.socket_address.port_value: 8108
   ```

2. **Point to your LLM endpoint** (examples provided for vLLM, Kubernetes, remote)
   ```yaml
   clusters[llm_upstream].load_assignment.endpoints[0].lb_endpoints[0].endpoint.address.socket_address:
     address: 127.0.0.1  # Your LLM host
     port_value: 8100    # Your LLM port
   ```

3. **Adjust ITS processing timeout** (increase for large budgets)
   ```yaml
   http_filters[ext_proc].timeout: 120s
   http_filters[ext_proc].message_timeout: 120s
   ```

4. **Change failure behavior** (reject requests when ext_proc is down)
   ```yaml
   http_filters[ext_proc].failure_mode_allow: false
   ```

**Configuration File Structure:**
- Comprehensive header with 6-point customization guide
- Inline comments on every important setting
- Examples for local, Kubernetes, and remote deployments
- **CRITICAL** annotations for required settings

For detailed configuration options and troubleshooting, see `config/envoy/ext_proc.yaml` (inline docs) and `its_hub/integration/ext_proc/HANDOVER.md`.

## Development

### Prerequisites

Before setting up the development environment, ensure you have:

- **git**: For version control and managing submodules. [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- **make**: Build automation tool (usually pre-installed on Unix systems)
- **uv**: An extremely fast Python package installer and resolver. [Installation Guide](https://astral.sh/uv/install/)
- **just** (optional): Provides backward-compatible command aliases. [Installation Guide](https://github.com/casey/just#installation)

### Build System

This project uses a **Make-based build system** for dependency management and proto compilation:

```bash
# View all available Make targets
make help

# General development setup (algorithm work)
make setup

# Full setup including Envoy protos (gateway work)
make setup-envoy

# Run all tests
make test
```

**Common Make Commands:**

```bash
# Setup
make setup          # Python deps + requirements.txt
make setup-envoy    # Full setup with submodules + protos

# Proto Management
make proto-compile  # Compile proto files (incremental)
make proto-clean    # Remove generated proto files
make upgrade-protos # Restore submodules to pinned commits

# Services
make envoy-stack      # Start Envoy + gRPC together
make envoy-stack-stop # Stop Envoy stack
make iaas-start       # Start IaaS service
make test             # Run all tests
```

### Protobuf Management

This project uses Git submodules for proto definitions, pinned to specific commits for reproducible builds.

**Automatic Setup:**

```bash
# Make handles everything automatically
make setup-envoy
```

**Updating Proto Versions:**

```bash
# 1. Edit .gitmodules and update pinned-commit comments
# 2. Restore to new commits
make upgrade-protos

# 3. Test compilation
make proto-compile

# 4. Commit changes
git add .gitmodules third_party/
git commit -s -m "chore: update proto definitions"
```

**Important**: Generated proto stubs (`its_hub/integration/ext_proc/proto/`) and logs (`*.log`) are gitignored and regenerated during builds.

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
