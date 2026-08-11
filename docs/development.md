# Development

## Getting Started

### Development Installation

Requires a Rust toolchain — the project uses [maturin](https://www.maturin.rs/) to build an extension from `rust/`.

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"

git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

The development installation includes:
- All core dependencies
- Rust native extension
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

Located in `its_hub/api/`:

```python
# Language model interface (its_hub/api/lm.py)
class AbstractLanguageModel:
    async def agenerate_single(self, messages, stop=None, **kwargs) -> dict: ...
    # Deprecation warning! agenerate is being deprecated in favor of agenerate_single
    async def agenerate(self, messages, stop=None, **kwargs) -> dict | list[dict]: ...

# Algorithm interface (its_hub/api/algorithm.py)
class AbstractScalingAlgorithm:
    async def ainfer(self, lm, prompt_or_messages, budget,
                     return_response_only=True, tools=None, tool_choice=None): ...
    def infer(self, ...): ...  # Sync wrapper via asyncio.run()

# Result interface (its_hub/api/algorithm.py)
class AbstractScalingResult:
    @property
    def the_one(self) -> dict: ...  # Best response as dict

# Orchestrator interface (its_hub/api/orchestrator.py)
class AbstractOrchestrator:
    async def agenerate(self, lm, messages_lst, ...) -> list[dict]: ...

# Reward model interfaces (its_hub/api/reward_models/)
class AbstractOutcomeRewardModel:
    def score(self, messages, **kwargs) -> list[float] | float: ...
    async def ascore(self, messages, orchestrator=None, **kwargs) -> list[float] | float: ...

class AbstractProcessRewardModel:
    def score(self, prompt_or_messages, steps) -> list[float]: ...
    async def ascore(self, prompt_or_messages, steps) -> list[float]: ...
```

### Component Overview

```
its_hub/
├── __init__.py             # Top-level exports (import from here)
├── _rust.*                 # Native extension (built by maturin)
├── algorithms/__init__.py  # Deprecated, backward compatibility only
├── api/                    # Public interfaces (stable API)
│   ├── lm.py              # AbstractLanguageModel
│   ├── algorithm.py       # AbstractScalingAlgorithm, AbstractScalingResult
│   ├── orchestrator.py    # AbstractOrchestrator
│   ├── types.py           # ChatMessage, ChatMessages
│   ├── errors.py          # APIError, RateLimitError, etc.
│   └── reward_models/
│       ├── orm.py         # AbstractOutcomeRewardModel
│       └── prm.py         # AbstractProcessRewardModel
├── core/                   # Implementations (internal)
│   ├── algorithms/
│   │   ├── self_consistency.py
│   │   ├── bon.py
│   │   ├── beam_search.py
│   │   ├── particle_gibbs.py
│   │   └── planning_wrapper.py
│   ├── lms/
│   │   ├── openai_lm.py   # OpenAICompatibleLanguageModel
│   │   └── step_generation.py
│   ├── reward_models/
│   │   ├── llm_judge.py   # LLMJudge
│   │   └── local_vllm_prm.py
│   ├── orchestrator.py    # LMOrchestrator
│   └── utils.py           # System prompts, helpers
rust/
├── Cargo.toml              # Rust crate manifest
└── src/
    └── lib.rs              # PyLMOrchestrator (PyO3 native extension)
```

## Adding New Algorithms

### 1. Implement Abstract Interface

```python
from its_hub import AbstractScalingAlgorithm, AbstractScalingResult
from its_hub.api import ChatMessages

class MyAlgorithmResult(AbstractScalingResult):
    def __init__(self, responses: list[dict], scores: list[float]):
        self.responses = responses
        self.scores = scores
    
    @property
    def the_one(self) -> dict:
        best_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        return self.responses[best_idx]

class MyAlgorithm(AbstractScalingAlgorithm):
    def __init__(self, custom_param: float = 1.0):
        self.custom_param = custom_param
    
    async def ainfer(self, lm, prompt_or_messages, budget, return_response_only=True):
        # Implement your algorithm logic here
        messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        responses = []
        scores = []
        
        for i in range(budget):
            response = await lm.agenerate_single(messages)
            score = self._score_response(response)
            responses.append(response)
            scores.append(score)
        
        result = MyAlgorithmResult(responses, scores)
        return result.the_one if return_response_only else result
    
    def _score_response(self, response: dict) -> float:
        # Implement your scoring logic
        return len(response.get("content", ""))  # Example: prefer longer responses
```

> The base class provides a sync `infer()` wrapper that calls `asyncio.run(self.ainfer(...))` automatically.

### 2. Add to Core Algorithms Module

```python
# its_hub/core/algorithms/my_algorithm.py
# Place your implementation here, then export from its_hub/__init__.py
```

### 3. Write Tests

```python
# tests/test_my_algorithm.py
import asyncio
from its_hub import AbstractLanguageModel, MyAlgorithm

class MockLM(AbstractLanguageModel):
    async def agenerate_single(self, messages, stop=None, **kwargs):
        return {"role": "assistant", "content": "mock response"}

def test_my_algorithm():
    lm = MockLM()
    algorithm = MyAlgorithm(custom_param=2.0)
    
    result = algorithm.infer(lm, "test prompt", budget=3)
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
```

## Adding New Language Models

### Implement Abstract Interface

The key method to implement is `agenerate_single()`, which the `AbstractOrchestrator` calls to fan out parallel LM requests. This is the contract between your LM and the orchestration layer — the orchestrator handles concurrency control, and your LM handles a single request:

```python
from its_hub import AbstractLanguageModel
from its_hub.api import ChatMessage

class MyLanguageModel(AbstractLanguageModel):
    def __init__(self, api_client):
        self.client = api_client
    
    async def agenerate_single(
        self, messages: list[ChatMessage], stop=None, **kwargs
    ) -> dict:
        # Convert ChatMessage objects to your API format and call your backend
        response = await self.client.generate(
            [m.to_dict() for m in messages], stop=stop, **kwargs
        )
        return {"role": "assistant", "content": response}
    
    async def close(self):
        # Clean up resources (sessions, connections, etc.)
        await self.client.close()
```

### Resource Cleanup

Language models that hold async resources (HTTP sessions, connections) must be cleaned up after use:

```python
# Option 1: Async context manager
async with MyLanguageModel(client) as lm:
    result = await algorithm.ainfer(lm, prompt, budget=5)

# Option 2: Explicit close (sync context)
lm = MyLanguageModel(client)
result = algorithm.infer(lm, prompt, budget=5)
asyncio.run(lm.close())
```

## Adding New Reward Models

### Process Reward Model

```python
from its_hub import AbstractProcessRewardModel
from its_hub.api import ChatMessage, ChatMessages

class MyProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def score(self, prompt_or_messages, steps: list[str]) -> list[float]:
        """Score each reasoning step."""
        messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)
        scores = []
        for step in steps:
            score = self._score_step(messages.to_prompt(), step)
            scores.append(score)
        return scores
    
    async def ascore(self, prompt_or_messages, steps: list[str]) -> list[float]:
        """Async version of score."""
        return self.score(prompt_or_messages, steps)

    def _score_step(self, context: str, step: str) -> float:
        # Implement step scoring logic
        return 1.0  # Placeholder
```

### Outcome Reward Model

```python
from its_hub import AbstractOutcomeRewardModel
from its_hub.api import ChatMessage, ChatMessages

class MyOutcomeRewardModel(AbstractOutcomeRewardModel):
    def score(self, messages, **kwargs) -> list[float] | float:
        """Score conversation(s)."""
        msgs = ChatMessages.from_prompt_or_messages(messages)
        content = msgs.to_chat_messages()[-1].extract_text_content()
        return 1.0 if "correct" in content.lower() else 0.0
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
import asyncio

def test_algorithm_with_real_lm():
    lm = OpenAICompatibleLanguageModel(...)
    algorithm = MyAlgorithm()
    result = algorithm.infer(lm, "test", budget=2)
    # Verify end-to-end behavior
    assert isinstance(result, dict)
    asyncio.run(lm.close())
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

The package version comes from `rust/Cargo.toml` (maturin reads it from there):

```toml
[package]
version = "0.2.0"
```

This value is the **next unreleased version** — it should always be one step
ahead of the latest published release. Bump it as soon as you cut a release so
`main` starts building dev versions toward the next one (see below).

### Dev versions on `main`

Every push to `main` publishes to Test PyPI. Because (Test) PyPI releases are
write-once, the `set-dev-version` composite action rewrites the version to a
unique `<version>.devN` (where `N` is the commit count) before each build on
`main`, so every upload is a brand-new release and never collides or hits Test
PyPI's "no new files on releases older than 14 days" rule. Tag builds skip this
step and use the static `rust/Cargo.toml` version as-is.

So with `version = "1.2.1"` in `Cargo.toml`, merges to `main` publish
`1.2.1.dev1`, `1.2.1.dev2`, … to Test PyPI, and tagging `v1.2.1` publishes the
clean `1.2.1` to PyPI.

### Creating Releases

1. Make sure `rust/Cargo.toml` holds the exact version you're releasing (it
   should already, since it tracks the next unreleased version)
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v1.2.1 -m "Release v1.2.1"`
4. Push tag: `git push origin v1.2.1` (or publish a GitHub Release)
5. GitHub Actions will handle PyPI publishing
6. Bump `rust/Cargo.toml` to the next version (e.g. `1.2.2`) and merge to `main`
   so dev builds move on to `1.2.2.devN`

> The release workflow's `check-version` job fails the build if the release tag
> (e.g. `v1.2.1`) doesn't match the version in `rust/Cargo.toml`, so keep the two
> in sync when bumping.

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