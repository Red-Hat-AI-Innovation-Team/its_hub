# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

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
- ✓ Dummy reward model for testing
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes OpenAI-compatible language model implementation:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel`, `StepGeneration` (requires `openai`, `aiohttp`, `backoff`)

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

Integrate its_hub algorithms into your AI gateway by implementing the `AbstractLanguageModel` interface:

```python
from its_hub import AbstractLanguageModel, SelfConsistency, BestOfN, DummyRewardModel
from its_hub.types import ChatMessage

# Step 1: Implement the language model interface with your gateway's LM client
class MyGatewayLM(AbstractLanguageModel):
    def __init__(self, gateway_client):
        self.client = gateway_client

    async def agenerate(self, messages, stop=None, max_tokens=None, temperature=None, **kwargs):
        """Use your gateway's existing LM client."""
        response = await self.client.generate(messages, stop=stop, max_tokens=max_tokens)
        return {"role": "assistant", "content": response}

# Step 2: Use its_hub algorithms
lm = MyGatewayLM(your_gateway_client)

# Self-Consistency example
sc_algorithm = SelfConsistency()
result = await sc_algorithm.ainfer(lm, "What is 2+2?", budget=5)
print(result.the_one)  # Most common answer from 5 generations

# Best-of-N example with dummy reward model
bon_algorithm = BestOfN(DummyRewardModel(fixed_score=0.8))
result = await bon_algorithm.ainfer(lm, "Explain quantum physics", budget=4)
print(result.the_one)  # Best response from 4 generations
```

### Example 2: Standalone Use with OpenAI-Compatible LM

**Installation required:** `pip install its_hub[lm]`

Use the provided OpenAI-compatible language model implementation:

```python
from its_hub import SelfConsistency
from its_hub.lms import OpenAICompatibleLanguageModel

# Initialize with any OpenAI-compatible API (OpenAI, vLLM, etc.)
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

# Use Self-Consistency algorithm
algorithm = SelfConsistency()
result = algorithm.infer(lm, "What is the capital of France?", budget=3)
print(result.the_one)  # Most common answer from 3 generations
```

### Example 3: Custom Reward Model for Best-of-N

Implement your own reward model for Best-of-N selection:

```python
from its_hub import AbstractOutcomeRewardModel, BestOfN

class MyCustomRewardModel(AbstractOutcomeRewardModel):
    async def evaluate(self, prompt: str, response: str) -> float:
        # Your custom scoring logic here
        score = len(response) / 100.0  # Example: prefer longer responses
        return score

# Use with Best-of-N
reward_model = MyCustomRewardModel()
algorithm = BestOfN(reward_model)
result = await algorithm.ainfer(lm, "Write a story", budget=5)
```

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **Gateway Integration**: Clean abstractions for easy integration with AI gateways
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async Support**: Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install
