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
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes OpenAI-compatible language model implementation:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel`, `LLMJudge`, `StepGeneration` (requires `openai`, `aiohttp`, `backoff`)

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

Gateway integration is simple: implement the `AbstractLanguageModel` interface with your existing LM client. Reference the `OpenAICompatibleLanguageModel` implementation (available with `[lm]` extra) as an example.

```python
from its_hub import AbstractLanguageModel, SelfConsistency

# Implement AbstractLanguageModel interface - this is the ONLY integration work required
class MyGatewayLM(AbstractLanguageModel):
    """
    Wrap your gateway's LM client to follow the AbstractLanguageModel contract.

    See OpenAICompatibleLanguageModel in its_hub.lms for a reference implementation.
    """
    def __init__(self, gateway_client):
        self.client = gateway_client

    async def agenerate(self, messages, stop=None, **kwargs):
        """
        Use your gateway's existing LM client.

        Args:
            messages: list[ChatMessage] or list[list[ChatMessage]]

        Returns:
            dict or list[dict] with format: {"role": "assistant", "content": "..."}
        """
        response = await self.client.generate(messages, stop=stop, **kwargs)
        return {"role": "assistant", "content": response}

# Use its_hub algorithms with your gateway's LM
lm = MyGatewayLM(your_gateway_client)

# Self-Consistency - vote on most common answer
sc_algorithm = SelfConsistency()
result = await sc_algorithm.ainfer(lm, "What is 2+2?", budget=5)
print(result.the_one)  # Most common answer from 5 generations
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

### Example 3: Best-of-N with LLM Judge

**Installation required:** `pip install its_hub[lm]`

Use LLM-based judging to select the best response. Customize with your own scoring prompt and judge model:

```python
from its_hub import BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.reward_models import LLMJudge

# Initialize language model for generation
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

# Option 1: Use default judge prompt (scores 0-10 for conversation quality)
judge_lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
    temperature=0.3,
)
judge = LLMJudge(lm=judge_lm, fallback_score=5.0)

# Option 2: Customize judge prompt for your use case
custom_prompt = """Evaluate the following conversation for code quality.
Consider: correctness, efficiency, readability, and best practices.

Conversation:
{conversation}

Return a JSON object with a score from 0-10.
Format: {{"score": <number>}}"""

judge_custom = LLMJudge(
    lm=judge_lm,
    judge_prompt=custom_prompt,
    fallback_score=5.0
)

# Use with Best-of-N
algorithm = BestOfN(judge_custom)
result = await algorithm.ainfer(lm, "Write a Python function to sort a list", budget=5)
print(result.the_one)  # Best code as judged by custom criteria
```

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **Gateway Integration**: Clean abstractions for easy integration with AI gateways
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async Support**: Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install
