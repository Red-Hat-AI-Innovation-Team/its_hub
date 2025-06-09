# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Development installation (includes all dependencies)
pip install -e ".[dev]"

# Production installation
pip install its_hub
```

### Testing
```bash
# Run all tests
pytest tests

# Run tests with coverage
pytest tests --cov=its_hub
```

### Code Quality
```bash
# Format code (Black configuration in pyproject.toml)
black its_hub/

# Sort imports (isort configuration in pyproject.toml)  
isort its_hub/
```

### Running Examples
```bash
# Test basic functionality
python scripts/test_math_example.py

# Benchmark algorithms (see script help for full options)
python scripts/benchmark.py --help
```

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with unified `infer()` method
- `AbstractScalingResult`: Base for algorithm results with `the_one` property
- `AbstractOutcomeRewardModel` / `AbstractProcessRewardModel`: Reward model interfaces

### Main Components

#### Language Models (`its_hub/lms.py`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions
- Supports async generation with concurrency limits and backoff strategies

#### Algorithms (`its_hub/algorithms/`)
All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True)`

- **Self-Consistency**: Generate multiple responses, select most common answer
- **Best-of-N**: Generate N responses, select highest scoring via outcome reward model  
- **Beam Search**: Step-by-step generation with beam width, uses process reward models
- **Particle Filtering/Gibbs**: Probabilistic resampling with process reward models

#### Integration (`its_hub/integration/`)
- `LocalVllmProcessRewardModel`: Integrates with reward_hub library for process-based scoring

### Budget Interpretation
- **Self-Consistency/Best-of-N**: Number of parallel generations
- **Beam Search**: Total generations divided by beam width  
- **Particle Filtering**: Number of particles

### Step Generation Pattern
The `StepGeneration` class enables incremental text generation:
- Configure step tokens (e.g., "\n\n" for reasoning steps)
- Set max steps and stop conditions
- Post-processing for clean output formatting

### Typical Workflow
1. Start vLLM server with instruction model
2. Initialize `OpenAICompatibleLanguageModel` pointing to server
3. Create `StepGeneration` with step/stop tokens appropriate for the task
4. Initialize reward model (e.g., `LocalVllmProcessRewardModel`)
5. Create scaling algorithm with step generation and reward model
6. Call `infer()` with prompt and budget

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets