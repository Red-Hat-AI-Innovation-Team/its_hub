# `its-hub`: A Python library for inference-time scaling

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

## 📚 Documentation

For comprehensive documentation, including installation guides, tutorials, and API reference, visit:

**[https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**

## Installation

Choose the installation option based on which algorithms you need:

```bash
# Core installation - includes:
#   - Best-of-N with LLM Judge
#   - Self-Consistency
#   - OpenAI-compatible language models
pip install its_hub

# Process Reward Model installation - adds:
#   - Particle Filtering Algorithms
#   - Beam Search
#   - LocalVllmProcessRewardModel
#   - Required for step-by-step reasoning with process reward models
pip install its_hub[prm]

# Development installation
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

## Quick Start

### Example 1: Best-of-N with LLM Judge

**Installation required:** `pip install its_hub` (core)

Use Best-of-N algorithm with an LLM judge for response selection - works with any OpenAI-compatible API:

```python
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel

# Initialize language model
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

# Set up LLM judge for scoring
judge = LLMJudgeRewardModel(
    model="gpt-4o-mini",
    criterion="overall_quality",
    judge_type="groupwise",
    api_key="your-api-key",
)
scaling_alg = BestOfN(judge)

# Generate multiple responses and select the best
result = scaling_alg.infer(lm, "Explain quantum entanglement in simple terms", budget=4)
print(result)
```

### Example 2: Particle Filtering with Process Reward Model

**Installation required:** `pip install its_hub[prm]`

Use Particle Filtering for step-by-step reasoning with process reward models:

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize language model (requires vLLM server running)
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8100/v1",
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
)

# Set up step generation and process reward model
sg = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

# Solve with step-by-step reasoning
result = scaling_alg.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=8)
print(result)
```

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **OpenAI-Compatible API**: Easy integration with existing applications  
- 🧮 **Math-Optimized**: Built for mathematical reasoning with specialized prompts
- 📊 **Benchmarking Tools**: Compare algorithms on MATH500 and AIME-2024 datasets
- ⚡ **Async Support**: Concurrent generation with limits and error handling


## Demo

See the library in action with a walkthrough of inference-time scaling algorithms:

[![Demo walkthrough](https://img.youtube.com/vi/qaXyvmR-YBU/maxresdefault.jpg)](https://www.youtube.com/watch?v=qaXyvmR-YBU)

Try it in your browser: [https://red.ht/its-hub-demo](https://red.ht/its-hub-demo)

To run the demo yourself, see the [demo setup instructions](https://github.com/lukeinglis/its_hub_demo/blob/main/demo_ui/README.md).

For detailed documentation, visit: [https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)
