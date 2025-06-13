# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml)
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

## Development

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
pytest tests
```

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)