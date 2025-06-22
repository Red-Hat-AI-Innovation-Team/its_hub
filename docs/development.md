# Development

## Getting Started

### Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

The development installation includes:
- All core dependencies
- Testing frameworks (pytest, coverage)
- Code formatting and linting tool (ruff)
- Development tools and scripts

### Running Tests

```bash
# Run all tests
pytest tests

# Run with coverage
pytest tests --cov=its_hub

# Run specific test modules
pytest tests/test_algorithms.py
pytest tests/test_lms.py
pytest tests/test_iaas.py
```

### Code Quality

```bash
# Run linter checks (Ruff configuration in pyproject.toml)
ruff check its_hub/

# Fix auto-fixable linting issues
ruff check its_hub/ --fix

# Format code
ruff format its_hub/
```

## Architecture

### Core Design Principles

**its-hub** follows a clean architecture with abstract base classes defining interfaces between components:

1. **Separation of Concerns**: Language models, algorithms, and reward models are independent
2. **Extensibility**: Easy to add new algorithms and models via abstract interfaces
3. **Async-First**: Built for high-performance concurrent inference
4. **Mathematical Focus**: Optimized for reasoning tasks with specialized prompts and evaluation

### Key Base Classes

Located in `its_hub/base.py`:

```python
# Language model interface
class AbstractLanguageModel:
    def generate(self, prompt: str) -> str: ...
    def generate_batch(self, prompts: list[str]) -> list[str]: ...

# Algorithm interface  
class AbstractScalingAlgorithm:
    def infer(self, lm, prompt, budget, return_response_only=True): ...

# Result interface
class AbstractScalingResult:
    @property
    def the_one(self) -> str: ...  # Best response

# Reward model interfaces
class AbstractOutcomeRewardModel:
    def score(self, prompt: str, response: str) -> float: ...

class AbstractProcessRewardModel:
    def score_steps(self, prompt: str, steps: list[str]) -> list[float]: ...
```

### Component Overview

```
its_hub/
├── base.py              # Abstract interfaces
├── lms.py              # Language model implementations
├── algorithms/         # Scaling algorithms
│   ├── self_consistency.py
│   ├── bon.py
│   ├── beam_search.py
│   └── particle_gibbs.py
├── integration/        # External integrations
│   ├── reward_hub.py   # Reward model integration
│   └── iaas.py        # API server
└── utils.py           # Utilities and prompts
```

## Adding New Algorithms

### 1. Implement Abstract Interface

```python
from its_hub.base import AbstractScalingAlgorithm, AbstractScalingResult

class MyAlgorithmResult(AbstractScalingResult):
    def __init__(self, responses: list[str], scores: list[float]):
        self.responses = responses
        self.scores = scores
    
    @property
    def the_one(self) -> str:
        # Return best response based on your criteria
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.responses[best_idx]

class MyAlgorithm(AbstractScalingAlgorithm):
    def __init__(self, custom_param: float = 1.0):
        self.custom_param = custom_param
    
    def infer(self, lm, prompt: str, budget: int, return_response_only: bool = True):
        # Implement your algorithm logic here
        responses = []
        scores = []
        
        for i in range(budget):
            response = lm.generate(prompt)
            score = self._score_response(response)
            responses.append(response)
            scores.append(score)
        
        result = MyAlgorithmResult(responses, scores)
        return result.the_one if return_response_only else result
    
    def _score_response(self, response: str) -> float:
        # Implement your scoring logic
        return len(response)  # Example: prefer longer responses
```

### 2. Add to Algorithms Module

```python
# its_hub/algorithms/__init__.py
from .my_algorithm import MyAlgorithm

__all__ = ['SelfConsistency', 'BestOfN', 'BeamSearch', 'ParticleFiltering', 'MyAlgorithm']
```

### 3. Write Tests

```python
# tests/test_my_algorithm.py
import pytest
from its_hub.algorithms import MyAlgorithm
from its_hub.lms import OpenAICompatibleLanguageModel

def test_my_algorithm():
    # Mock language model for testing
    class MockLM:
        def generate(self, prompt):
            return f"Response to: {prompt}"
    
    lm = MockLM()
    algorithm = MyAlgorithm(custom_param=2.0)
    
    result = algorithm.infer(lm, "test prompt", budget=3)
    assert isinstance(result, str)
    assert "Response to: test prompt" in result
```

## Adding New Language Models

### 1. Implement Abstract Interface

```python
from its_hub.base import AbstractLanguageModel

class MyLanguageModel(AbstractLanguageModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Initialize your model here
    
    def generate(self, prompt: str) -> str:
        # Implement single generation
        pass
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        # Implement batch generation
        return [self.generate(p) for p in prompts]
    
    def score(self, prompt: str, response: str) -> float:
        # Implement response scoring (optional)
        return 0.0
```

### 2. Add Async Support

```python
import asyncio
from typing import Optional

class MyAsyncLanguageModel(AbstractLanguageModel):
    async def generate_async(self, prompt: str, **kwargs) -> str:
        # Implement async generation
        pass
    
    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        # Implement async batch generation
        tasks = [self.generate_async(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)
```

## Adding New Reward Models

### Process Reward Model

```python
from its_hub.base import AbstractProcessRewardModel

class MyProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Load your reward model
    
    def score_steps(self, prompt: str, steps: list[str]) -> list[float]:
        """Score each reasoning step"""
        scores = []
        context = prompt
        
        for step in steps:
            score = self._score_step(context, step)
            scores.append(score)
            context += f"\\n{step}"
        
        return scores
    
    def _score_step(self, context: str, step: str) -> float:
        # Implement step scoring logic
        return 1.0  # Placeholder
```

### Outcome Reward Model

```python
from its_hub.base import AbstractOutcomeRewardModel

class MyOutcomeRewardModel(AbstractOutcomeRewardModel):
    def score(self, prompt: str, response: str) -> float:
        """Score the final response"""
        # Implement outcome scoring logic
        return self._evaluate_correctness(prompt, response)
    
    def _evaluate_correctness(self, prompt: str, response: str) -> float:
        # Custom evaluation logic
        return 1.0 if "correct" in response.lower() else 0.0
```

## Testing Guidelines

### Unit Tests

```python
# Test individual components
def test_algorithm_basic():
    algorithm = MyAlgorithm()
    # Test basic functionality

def test_algorithm_edge_cases():
    algorithm = MyAlgorithm()
    # Test edge cases and error conditions

def test_algorithm_with_mock():
    # Use mocks to isolate component under test
    pass
```

### Integration Tests

```python
# Test component interactions
def test_algorithm_with_real_lm():
    lm = OpenAICompatibleLanguageModel(...)
    algorithm = MyAlgorithm()
    result = algorithm.infer(lm, "test", budget=2)
    # Verify end-to-end behavior
```

### Performance Tests

```python
import time

def test_algorithm_performance():
    start_time = time.time()
    # Run algorithm
    elapsed = time.time() - start_time
    assert elapsed < 10.0  # Performance requirement
```

## Git Workflow

### Commits

Always use the sign-off flag for commits:

```bash
git commit -s -m "feat: add new algorithm implementation"
```

### Branch Naming

- `feat/algorithm-name` - New features
- `fix/issue-description` - Bug fixes  
- `docs/section-name` - Documentation updates
- `refactor/component-name` - Code refactoring

### Pull Request Process

1. Create feature branch from `main`
2. Make changes with signed commits
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass
6. Submit pull request with clear description

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Brief description of the function.
    
    Longer description if needed, explaining the purpose
    and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default value
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this exception is raised
        
    Example:
        >>> result = my_function("hello", 5)
        >>> print(result)
        True
    """
    return len(param1) > param2
```

### Code Comments

- Explain **why**, not **what**
- Use comments for complex algorithms or non-obvious logic
- Keep comments up-to-date with code changes

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def profile_algorithm():
    pr = cProfile.Profile()
    pr.enable()
    
    # Run your algorithm here
    algorithm.infer(lm, prompt, budget=10)
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(10)
```

### Memory Optimization

```python
import tracemalloc

tracemalloc.start()

# Run your code
result = algorithm.infer(lm, prompt, budget=10)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
```

### GPU Memory Management

```python
import torch

def optimize_gpu_memory():
    # Clear cache periodically
    torch.cuda.empty_cache()
    
    # Monitor memory usage
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    print(f"GPU Memory - Allocated: {allocated/1e9:.2f}GB, Cached: {cached/1e9:.2f}GB")
```

## Release Process

### Version Bumping

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"
```

### Creating Releases

1. Update version number
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions will handle PyPI publishing

## Contributing

### Issues

- Use issue templates when available
- Provide minimal reproducible examples
- Include environment details (OS, Python version, GPU type)

### Feature Requests

- Explain the use case and motivation
- Provide examples of desired API
- Consider backwards compatibility

### Code Review

- Review for correctness, performance, and maintainability
- Suggest improvements constructively
- Test the changes locally when possible