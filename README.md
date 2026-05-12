# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

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
- ✓ Self-Consistency and Best-of-N algorithms
- ✓ Abstract base classes (`AbstractLanguageModel`, `AbstractOutcomeRewardModel`)
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes OpenAI-compatible language model implementation:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel`, `LLMJudge`, `StepGeneration` (requires `openai`, `aiohttp`, `backoff`)

### With Experimental Algorithms

For **experimental features** - includes beam search and particle filtering:

```bash
pip install its_hub[experimental]
```

Adds: Process reward models, beam search, particle filtering algorithms

### Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
# or using uv:
uv sync --extra dev
```

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

## Key Features

- 🔬 **Multiple Algorithms**: Self-Consistency, Best-of-N, Beam Search (experimental), Particle Filtering (experimental)
- 🚀 **Gateway Integration**: Clean abstractions (`AbstractLanguageModel`, `AbstractOrchestrator`) for easy integration with AI gateways
- 🔄 **Orchestration**: `AbstractOrchestrator` provides structured concurrency, rate limiting, and error propagation for parallel LM calls — essential for production gateway deployments
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async-First**: `ainfer()` is the primary method; `infer()` is a sync wrapper. Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install

## Coding Agent Plugin

its-hub is available as a plugin for five major coding agents, bringing inference-time scaling directly into your coding workflow.

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
<summary><strong>Cursor</strong></summary>

Clone the repo and open it — Cursor discovers the plugin via `.cursor-plugin/plugin.json` automatically.
</details>

<details>
<summary><strong>Gemini CLI</strong></summary>

```bash
gemini extensions install https://github.com/Red-Hat-AI-Innovation-Team/its_hub
```
</details>

<details>
<summary><strong>Codex CLI</strong></summary>

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git ~/.codex/its-hub
mkdir -p ~/.agents/skills
ln -s ~/.codex/its-hub/skills ~/.agents/skills/its-hub
```

Restart Codex to discover the skills. See `.codex-plugin/INSTALL.md` for full instructions.
</details>

<details>
<summary><strong>OpenCode</strong></summary>

Add to your `opencode.json`:

```json
{
  "plugin": ["its-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git"]
}
```

Restart OpenCode. See `.opencode-plugin/INSTALL.md` for full instructions.
</details>

### After Installing

Run `/its-setup` (or invoke the `setup-guide` skill) to configure your model endpoint and algorithm.

| Command | Description |
|---|---|
| `/its-setup` | Guided first-time configuration |
| `/its-scale <prompt>` | Run inference-time scaling on a single prompt |
| `/its-scale-batch <file>` | Batch scaling from a JSONL/CSV/TXT file |

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
