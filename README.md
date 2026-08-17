# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

<p align="center">
  <video src="https://github.com/user-attachments/assets/f92de395-f3c0-49a7-b1a9-caa265ffe2c2" width="80%" controls autoplay loop muted playsinline>
    ITS Hub algorithms: Self-Consistency, Best-of-N, and Particle Filtering
  </video>
</p>

## 📚 Documentation

For comprehensive documentation, including installation guides, tutorials, and API reference, visit:

**[https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**

## Installation

**its_hub** provides a minimal core focused on algorithms, with optional language model implementations.

### Core Installation (Algorithms Only)

For **gateway integration** - just algorithms and interfaces, minimal dependencies:

```bash
pip install its_hub
```

This includes:
- ✓ Self-Consistency (and its variants) and Best-of-N algorithms — see [Algorithms](docs/algorithms.md) for the full list
- ✓ Abstract base classes (`AbstractLanguageModel`, `AbstractOutcomeRewardModel`)
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes OpenAI-compatible language model implementation:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel`, `LLMJudge`, `StepGeneration` (requires `openai`, `aiohttp`, `backoff`)

> **vLLM users:** its_hub uses the `max_completion_tokens` parameter (the OpenAI API standard), which requires **vLLM >= 0.6.2**. We recommend **vLLM >= 0.14.0**.

### With Experimental Algorithms

For **experimental features** - includes beam search and particle filtering:

```bash
pip install its_hub[experimental]
```

Adds: Process reward models, beam search, particle filtering algorithms

### Development Installation

Requires a [Rust toolchain](https://rustup.rs/) (the build backend compiles our rust code into a python module).

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
# or using uv:
uv sync --extra dev
```

To use ITS as an external processor with Envoy:
```bash
make setup-envoy
```

For more information, refer to [docs/ext-proc-gateway.md](docs/ext-proc-gateway.md) and [docs/iaas-service.md](docs/iaas-service.md).

## Quick Start

### Example 1: Gateway Integration (Core Installation)

**Installation required:** `pip install its_hub` (core only, minimal dependencies)

Gateway integration requires implementing two interfaces: `AbstractLanguageModel` for LM calls and `AbstractOrchestrator` for managing parallel execution with concurrency control and rate limiting.

```python
import asyncio

from its_hub import AbstractLanguageModel, AbstractOrchestrator, SelfConsistency

# Step 1: Implement AbstractLanguageModel with your gateway's LM client
class MyGatewayLM(AbstractLanguageModel):
    def __init__(self, gateway_client):
        self.client = gateway_client

    async def agenerate_single(self, messages, stop=None, **kwargs):
        response = await self.client.generate(messages, stop=stop, **kwargs)
        return {"role": "assistant", "content": response}

# Step 2: Implement AbstractOrchestrator for concurrency control
# (or use the built-in LMOrchestrator from its_hub[lm])
class MyGatewayOrchestrator(AbstractOrchestrator):
    async def agenerate(self, lm, messages_lst, **kwargs):
        # Manage parallel calls with your gateway's rate limits
        ...

async def main():
    lm = MyGatewayLM(your_gateway_client)
    orchestrator = MyGatewayOrchestrator()
    algorithm = SelfConsistency(orchestrator=orchestrator)
    result = await algorithm.ainfer(lm, "What is 2+2?", budget=5)
    print(result)  # {"role": "assistant", "content": "4", ...}

asyncio.run(main())
```

The `AbstractOrchestrator` is the central coordination point — it controls how algorithms fan out parallel LM calls, enforces rate limits, and provides structured error handling. See [Orchestration](docs/orchestration.md) for details.

### Example 2: Standalone Use with OpenAI-Compatible LM

**Installation required:** `pip install its_hub[lm]`

```python
import asyncio

from its_hub import OpenAICompatibleLanguageModel, SelfConsistency

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

algorithm = SelfConsistency()
result = algorithm.infer(lm, "What is the capital of France?", budget=3)
print(result)  # Most common answer from 3 generations

# Close lm for resource cleanup
asyncio.run(lm.close())
```

### Example 3: Best-of-N with LLM Judge

**Installation required:** `pip install its_hub[lm]`

```python
import asyncio

from its_hub import BestOfN, LLMJudge, OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

judge = LLMJudge(lm=lm, fallback_score=5.0)
algorithm = BestOfN(orm=judge)
result = algorithm.infer(lm, "Write a sorting function", budget=5)
print(result)  # Best response as judged by LLM

# Close lm for resource cleanup
asyncio.run(lm.close())
```

### Proxy Configuration

Like `requests`, `httpx`, and the OpenAI SDK, `OpenAICompatibleLanguageModel`
respects the standard proxy environment variables, so you can reach an upstream
that is only accessible through a proxy without any code changes. Set them in the
environment before creating the LM:

| Variable | Purpose |
| --- | --- |
| `HTTP_PROXY` / `HTTPS_PROXY` | Proxy URL for `http://` / `https://` requests |
| `NO_PROXY` | Comma-separated hosts/domains to connect to directly, bypassing the proxy |

```bash
export HTTPS_PROXY="http://proxy.example.com:8080"
export NO_PROXY="localhost,127.0.0.1"
```

Credentials in `.netrc` are also honored. Lowercase variants (`http_proxy`, etc.)
work as well.

## Key Features

- 🔬 **Multiple Algorithms**: Voting, confidence-based selection, Best-of-N, Beam Search (experimental), Particle Filtering (experimental) — see [Algorithms](docs/algorithms.md) for the full list
- 🚀 **Gateway Integration**: Clean abstractions (`AbstractLanguageModel`, `AbstractOrchestrator`) for easy integration with AI gateways
- 🔄 **Orchestration**: `AbstractOrchestrator` provides structured concurrency, rate limiting, and error propagation for parallel LM calls — essential for production gateway deployments
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async-First**: `ainfer()` is the primary method; `infer()` is a sync wrapper. Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install

## Coding Agent Plugin

its-hub is available as a plugin for two coding agents, bringing inference-time scaling directly into your coding workflow.

<details>
<summary><strong>Claude Code</strong></summary>

**Via org marketplace** (recommended — includes all Red Hat AI plugins):
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
/plugin install its-hub@Red-Hat-AI-Innovation-Team/plugins
```

**Via this repo directly:**
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/its_hub
/plugin install its-hub@Red-Hat-AI-Innovation-Team/its_hub
```

**From a local clone:**
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
/plugin marketplace add /path/to/its_hub
```
</details>

<details>
<summary><strong>Codex CLI</strong></summary>

```bash
codex plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
```

Then install the plugin from the marketplace. See `.codex-plugin/INSTALL.md` for manual installation.
</details>

### After Installing

Invoke the `setup-guide` skill to configure your model endpoint and algorithm.

| Skill | Description |
|---|---|
| `setup-guide` | Guided first-time configuration |
| `inference-scaling` | Run inference-time scaling on a single prompt |
| `batch-scaling` | Batch scaling from a JSONL/CSV/TXT file |

## Demo

See the library in action with a walkthrough of inference-time scaling algorithms:

[![Demo walkthrough](https://img.youtube.com/vi/qaXyvmR-YBU/maxresdefault.jpg)](https://www.youtube.com/watch?v=qaXyvmR-YBU)

Try it in your browser: [https://red.ht/its-hub-demo](https://red.ht/its-hub-demo)

To run the demo yourself, see the [demo setup instructions](https://github.com/lukeinglis/its_hub_demo/blob/main/demo_ui/README.md).

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
