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

Gateway integration is simple: implement the `AbstractLanguageModel` interface with your existing LM client.

```python
from its_hub import AbstractLanguageModel, SelfConsistency

# Implement AbstractLanguageModel interface - this is the ONLY integration work required
class MyGatewayLM(AbstractLanguageModel):
    def __init__(self, gateway_client):
        self.client = gateway_client

    async def agenerate(self, messages, stop=None, **kwargs):
        response = await self.client.generate(messages, stop=stop, **kwargs)
        return {"role": "assistant", "content": response}

# Use its_hub algorithms with your gateway's LM
lm = MyGatewayLM(your_gateway_client)
algorithm = SelfConsistency()
result = await algorithm.ainfer(lm, "What is 2+2?", budget=5)
print(result)  # {"role": "assistant", "content": "4", ...}
```

### Example 2: Standalone Use with OpenAI-Compatible LM

**Installation required:** `pip install its_hub[lm]`

```python
from its_hub import OpenAICompatibleLanguageModel, SelfConsistency

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

algorithm = SelfConsistency()
result = algorithm.infer(lm, "What is the capital of France?", budget=3)
print(result)  # Most common answer from 3 generations
```

### Example 3: Best-of-N with LLM Judge

**Installation required:** `pip install its_hub[lm]`

```python
from its_hub import BestOfN, LLMJudge, OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

judge = LLMJudge(lm=lm, fallback_score=5.0)
algorithm = BestOfN(reward_model=judge)
result = algorithm.infer(lm, "Write a sorting function", budget=5)
print(result)  # Best response as judged by LLM
```

## Key Features

- 🔬 **Multiple Algorithms**: Self-Consistency, Best-of-N, Beam Search (experimental), Particle Filtering (experimental)
- 🚀 **Gateway Integration**: Clean abstractions for easy integration with AI gateways
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async Support**: Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
