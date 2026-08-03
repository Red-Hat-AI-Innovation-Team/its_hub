# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Development installation with uv (recommended)
uv sync --extra dev

# Alternative: pip installation
pip install -e ".[dev]"

# Production installation
pip install its_hub
```

`uv sync` automatically rebuilds the Rust extension (`its_hub._rust`) whenever `.rs` sources or `Cargo.toml` change, thanks to the `[tool.uv] cache-keys` config in `pyproject.toml`. No separate `maturin develop` step is needed during normal development.

### Contribution
When commit or raising PR, never mention it is by ClaudeCode.
never say 🤖 Generated with [Claude Code](https://claude.ai/code)" in the commit statment, don't mention claude!

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_algorithms.py

# Run tests with coverage
uv run pytest tests/ --cov=its_hub

# Run tests with verbose output
uv run pytest tests/ -v
```

### Code Quality
```bash
# Run linter checks
uv run ruff check its_hub/

# Fix auto-fixable linting issues
uv run ruff check its_hub/ --fix

# Format code with ruff
uv run ruff format its_hub/
```

### Git Workflow
```bash
# Create commits with sign-off
git commit -s -m "commit message"

# For any git commits, always use the sign-off flag (-s)
```

### Running Examples
```bash
# Test basic functionality
python scripts/test_math_example.py

# Benchmark algorithms (see script help for full options)
python scripts/benchmark.py --help
```

### IaaS Service (Inference-as-a-Service)
```bash
# Start IaaS service
uv run its-iaas --host 127.0.0.1 --port 8109

# Or using justfile (if available)
just iaas-start

# Check service health
curl -s http://localhost:8108/v1/models | jq .

# Configure the service (example: self-consistency algorithm)
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "http://localhost:8100/v1", "api_key": "NO_API_KEY", "model": "your-model-name", "alg": "self-consistency"}'

# For comprehensive IaaS setup (multi-GPU, reward models, etc.), see docs/iaas-service.md
```

## Additional Tips
- Use `rg` in favor of `grep` whenever it's available
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`
- In case of dependency issues during testing, try commenting out `reward_hub` and `vllm` temporarily in @pyproject.toml and retry.

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The architecture separates public interfaces (`its_hub/api/`) from implementations (`its_hub/core/`).

### Directory Structure

```
its_hub/
├── __init__.py                 # Top-level exports (import from here)
├── api/                        # Public interfaces (stable API)
│   ├── lm.py                  # AbstractLanguageModel
│   ├── algorithm.py           # AbstractScalingAlgorithm, AbstractScalingResult
│   ├── orchestrator.py        # AbstractOrchestrator
│   ├── types.py               # ChatMessage, ChatMessages
│   ├── errors.py              # APIError, RateLimitError, etc.
│   └── reward_models/
│       ├── orm.py             # AbstractOutcomeRewardModel
│       └── prm.py             # AbstractProcessRewardModel
├── core/                       # Implementations (internal)
│   ├── algorithms/
│   │   ├── self_consistency.py
│   │   ├── bon.py
│   │   ├── beam_search.py     # Experimental
│   │   ├── particle_gibbs.py  # Experimental
│   │   └── planning_wrapper.py
│   ├── lms/
│   │   ├── openai_lm.py      # OpenAICompatibleLanguageModel
│   │   └── step_generation.py # StepGeneration
│   ├── reward_models/
│   │   ├── llm_judge.py       # LLMJudge
│   │   └── local_vllm_prm.py # LocalVllmProcessRewardModel
│   ├── orchestrator.py        # LMOrchestrator
│   └── utils.py               # System prompts, helpers
```

### Key Base Classes (`its_hub/api/`)
- `AbstractLanguageModel`: Interface for async LM generation (`agenerate()`, `agenerate_single()`)
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with `ainfer()` (async) and `infer()` (sync wrapper)
- `AbstractScalingResult`: Base for algorithm results with `the_one` property returning a `dict`
- `AbstractOrchestrator`: Interface for managing parallel LM calls
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models
- `AbstractProcessRewardModel`: Interface for process-based reward models (step-by-step scoring)

### Main Components

#### Language Models (`its_hub/core/lms/`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs. Supports async context manager (`async with`) and requires `close()` for cleanup.
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions
- Async-first design with concurrency limits and backoff strategies

#### Algorithms (`its_hub/core/algorithms/`)
All algorithms follow the same interface: `ainfer(lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None)` (async primary) or `infer(...)` (sync wrapper)

- **Self-Consistency**: Generate multiple responses, select most common answer (supports tool voting)
- **Best-of-N**: Generate N responses, select highest scoring via outcome reward model
- **Beam Search** (experimental): Step-by-step generation with beam width, uses process reward models
- **Particle Filtering/Gibbs** (experimental): Probabilistic resampling with process reward models

#### Orchestrator (`its_hub/core/orchestrator.py`)
- `LMOrchestrator`: Built-in implementation of `AbstractOrchestrator` using `asyncio.TaskGroup` with thread-safe semaphore for concurrency control
- The orchestrator is central to gateway integration — it controls how algorithms fan out parallel LM calls, enforces rate limits, and handles error propagation. Gateway teams can implement `AbstractOrchestrator` with their own concurrency policies or use the built-in `LMOrchestrator`

#### Reward Models (`its_hub/core/reward_models/`)
- `LLMJudge`: LLM-based outcome reward model for scoring responses
- `LocalVllmProcessRewardModel`: Integrates with reward_hub library for process-based scoring (experimental)

### Budget Interpretation
The budget parameter controls computational resources allocated to each algorithm. Different algorithms interpret budget as follows:
- **Self-Consistency/Best-of-N**: Number of parallel generations to create
- **Beam Search**: Total generations divided by beam width (controls search depth)
- **Particle Filtering**: Number of particles maintained during sampling

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
6. Call `ainfer()` (async) or `infer()` (sync wrapper) with prompt and budget
7. Close LM with `await lm.close()` or `asyncio.run(lm.close())` for resource cleanup

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/core/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets

## Coding Agent Plugin

This repo serves as a plugin for Claude Code and Codex CLI. The plugin follows the standard structure below — skills are the primary interface, scripts handle execution.

### Standard Plugin Structure

```
.claude-plugin/                        # Claude Code discovery manifest
  plugin.json
.codex-plugin/                         # Codex CLI discovery manifest + install guide
  plugin.json
  INSTALL.md
.claude/skills/                        # Skills (auto-triggered + explicitly invokable)
  setup-guide/SKILL.md                 # First-time installation and configuration
  inference-scaling/SKILL.md           # Single-prompt scaling
  inference-scaling-guide/SKILL.md     # Algorithm selection and troubleshooting
  batch-scaling/SKILL.md               # Batch scaling from file
scripts/                               # Bash/Python helpers called by skills
  _env.sh                              # Shared Python resolution
  its_detect.sh                        # Environment detection (library, config)
  its_scale.sh                         # Single-prompt scaling (bash wrapper)
  its_batch_scale.sh                   # Batch scaling (bash wrapper)
  _its_scale_runner.py                 # Single-prompt scaling logic
  _its_batch_runner.py                 # Batch scaling logic
```

Both plugin manifests point to `./.claude/skills` for discovery.

### Plugin Config

User config is stored at `.its-hub/config.json` (auto-generated by the `setup-guide` skill, gitignored). API keys are **not** stored in the config — they are read from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

### Plugin Development

- Skills are markdown files — edit `.claude/skills/*/SKILL.md` directly
- Scripts are bash — edit `scripts/*.sh`
- Discovery manifests rarely change
- Test plugin changes by invoking skills in Claude Code or Codex