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

### Contribution
When commit or raising PR, never mention it is by ClaudeCode.
never say 🤖 Generated with [Claude Code](https://claude.ai/code)" in the commit statment, don't mention claude!

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_particle_filtering.py

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

### Benchmarking
```bash
# MMAU-Pro audio MCQ benchmark (requires a served audio LM, e.g. Qwen2.5-Omni on vLLM)
python -m benchmarking.mmau_pro.run_mmau --help

# E2E math tests against a served model
python tests/e2e/test_e2e.py --help
```

## Additional Tips
- Use `rg` in favor of `grep` whenever it's available
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs using Particle Filtering (PF) and Entropic Particle Filtering (EPF). Particle weights come from the *generator* model's own token logprobs (self-certainty) — no separate reward model. The architecture separates public interfaces (`its_hub/api/`) from implementations (`its_hub/core/`).

### Directory Structure

```
its_hub/
├── __init__.py                 # Top-level exports (import from here)
├── api/                        # Public interfaces (stable API)
│   ├── lm.py                  # AbstractLanguageModel
│   ├── algorithm.py           # AbstractScalingAlgorithm, AbstractScalingResult
│   ├── orchestrator.py        # AbstractOrchestrator
│   ├── types.py               # ChatMessage, ChatMessages (audio-aware helpers)
│   └── errors.py              # APIError, RateLimitError, etc.
├── core/                       # Implementations (internal)
│   ├── algorithms/
│   │   └── particle_filtering.py  # ParticleFiltering, EntropicParticleFiltering
│   ├── lms/
│   │   ├── openai_lm.py      # OpenAICompatibleLanguageModel (logprobs support)
│   │   └── step_generation.py # StepGeneration (logprobs + audio carry)
│   ├── orchestrator.py        # LMOrchestrator
│   └── utils.py               # System prompts, summarize_step_logprobs, helpers
benchmarking/
└── mmau_pro/                   # MMAU-Pro audio MCQ benchmark (Qwen2.5-Omni)
documentation/                  # In-depth design docs (weights, annealing, audio)
```

### Key Base Classes (`its_hub/api/`)
- `AbstractLanguageModel`: Interface for async LM generation (`agenerate()`, `agenerate_single()`)
- `AbstractScalingAlgorithm`: Base for scaling algorithms with `ainfer()` (async) and `infer()` (sync wrapper)
- `AbstractScalingResult`: Base for algorithm results with `the_one` property returning a `dict`
- `AbstractOrchestrator`: Interface for managing parallel LM calls

### Main Components

#### Language Models (`its_hub/core/lms/`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs. Supports async context manager (`async with`) and requires `close()` for cleanup. Requests token `logprobs` when asked (used for self-certainty particle weights).
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions. Optionally returns a per-step logprob summary, and carries structured `base_messages` (e.g. an audio user turn) verbatim to the model.
- Async-first design with concurrency limits and backoff strategies

#### Algorithms (`its_hub/core/algorithms/particle_filtering.py`)
Both follow the same interface: `ainfer(lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None)` (async primary) or `infer(...)` (sync wrapper)

- **ParticleFiltering (PF)**: step-by-step generation; after each step, particles are weighted from the generator's own logprobs (`self_certainty_signal="mean_logprob"` or `"entropy"`, `self_certainty_style="logit"` or `"raw"`) and resampled (multinomial by default)
- **EntropicParticleFiltering (EPF)**: PF plus entropic annealing — when the effective sample size collapses early, the resampling distribution is tempered (`temperature_method`: ess/entropy/base); systematic resampling by default

#### Orchestrator (`its_hub/core/orchestrator.py`)
- `LMOrchestrator`: Built-in implementation of `AbstractOrchestrator` using `asyncio.TaskGroup` with thread-safe semaphore for concurrency control
- Gateway teams can implement `AbstractOrchestrator` with their own concurrency policies or use the built-in `LMOrchestrator`

### Budget Interpretation
`budget` = number of particles maintained during sampling.

### Step Generation Pattern
The `StepGeneration` class enables incremental text generation:
- Configure step tokens (e.g., "\n\n" for reasoning steps) or `tokens_per_step`
- Set max steps and stop conditions
- Post-processing for clean output formatting

### Typical Workflow
1. Start vLLM server with the model (must support `logprobs`; for audio models like Qwen2.5-Omni serve with audio enabled)
2. Initialize `OpenAICompatibleLanguageModel` pointing to server
3. Create `StepGeneration` with step/stop tokens appropriate for the task
4. Create `ParticleFiltering` or `EntropicParticleFiltering` with the step generation
5. Call `ainfer()` (async) or `infer()` (sync wrapper) with the prompt (string or structured messages with audio) and budget
6. Close LM with `await lm.close()` or `asyncio.run(lm.close())` for resource cleanup

### Audio / Multimodal Support
- `ChatMessages.has_nontext_content()` detects structured (audio/image) content; the algorithms then carry the original messages verbatim (`base_user_messages()`) to the model at every step instead of flattening to text
- See `benchmarking/mmau_pro/` for an end-to-end audio benchmark and `documentation/audio-mmau-changes.md` for the design

### Mathematical Focus
- Predefined system prompts in `its_hub/core/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- E2E benchmarking on MATH500 and AIME-2024 datasets (`tests/e2e/`)
