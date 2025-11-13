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

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_algorithms.py

# Run specific test by name pattern
uv run pytest tests/ -k "test_function_name"

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

# Run both check and format together
uv run ruff check its_hub/ --fix && uv run ruff format its_hub/
```

### Git Workflow
```bash
# IMPORTANT: Always use sign-off flag (-s) for commits
git commit -s -m "commit message"

# When committing or raising PR, never mention Claude Code or AI assistance
# Do NOT include "🤖 Generated with Claude Code" or similar in commits
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
uv run its-iaas --host 0.0.0.0 --port 8108

# Or using justfile (extensive examples available)
just iaas-start

# Check service health
curl -s http://localhost:8108/v1/models | jq .

# Configure the service (example: self-consistency algorithm)
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "http://localhost:8100/v1", "api_key": "NO_API_KEY", "model": "your-model-name", "alg": "self-consistency"}'

# For comprehensive IaaS setup (multi-GPU, reward models, etc.), see docs/iaas-service.md
# The justfile contains many more examples: test-chat, test-math, test-calculator, etc.
```

## Package Installation Extras

Different algorithm configurations require different dependencies:

- **Core** (default): `pip install its_hub`
  - Includes: Best-of-N, Self-Consistency, OpenAICompatibleLanguageModel, LiteLLMLanguageModel
  - Works with any OpenAI-compatible API or cloud provider
- **PRM** (process reward models): `pip install its_hub[prm]`
  - Adds: Particle Filtering, Beam Search, LocalVllmProcessRewardModel
  - Requires GPU for local process reward models
- **Dev**: `pip install -e ".[dev]"` or `uv sync --extra dev`
  - Includes all testing, linting, and development dependencies
- **Research**: `pip install its_hub[research]`
  - Adds: math_verify, datasets, matplotlib for benchmarking
- **Cloud**: `pip install its_hub[cloud]`
  - Adds: boto3 (AWS), google-cloud-aiplatform (GCP) for direct cloud SDK usage

## Development Tips
- Use `uv` for Python environment management: always start with `uv sync --extra dev` to init the env and run stuff with `uv run`
- The justfile contains extensive test examples for IaaS - reference it for API usage patterns and curl commands
- For testing without GPU: core installation supports Best-of-N and Self-Consistency with cloud APIs
- For testing with GPU: use `[prm]` extra to test Particle Filtering and Beam Search with local models

---

## 🚧 ACTIVE REFACTORING: Repository Restructure

**Status:** In Progress
**Goal:** Transform its_hub from a bundled multi-purpose library into a focused algorithm package with optional integrations.

### Current State: Three Bundled Functionalities

ITS-Hub currently combines three distinct responsibilities:

1. **Core Algorithms** - Inference-time scaling algorithms
   - Self-Consistency, Best-of-N, Particle Filtering, Beam Search, ParticleGibbs, MCTS
   - Core interface: `async_infer(lm, algorithm_name, budget, ...)`

2. **Orchestration Layer** - Language Model abstraction
   - `AbstractLanguageModel` and concrete implementations
   - `OpenAICompatibleLanguageModel`, `LiteLLMLanguageModel`
   - Handles API calls, retries, error handling, async batching

3. **Proxy Server** - IaaS FastAPI service
   - OpenAI-compatible API endpoint
   - Request transformation and routing
   - Global state management (LM_DICT, SCALING_ALG)

**Problem:** Complex installation, unclear scope, difficult integration into existing AI gateways.

### Refactoring Goals

#### Primary Goal: Algorithm-First Architecture
- **Main scope**: Core scaling algorithms with minimal dependencies
- **Optional components**: LM orchestration and proxy server become separate packages/extras
- **Clear boundaries**: Algorithms don't dictate LM implementation details

#### Benefits of Decoupling

1. **Clearer Scope**
   - Algorithms are the core value proposition, not the LLM engine
   - Clean separation enables integration into any LLM orchestration system
   - Easier to contribute new algorithms without LM engine concerns

2. **Simplified Integration**
   - AI gateways can import only the algorithm components
   - Define orchestration logic locally using their existing LM clients
   - No forced dependency on OpenAI/LiteLLM patterns

3. **Reduced Installation Complexity**
   - Core algorithm install is lightweight (minimal dependencies)
   - LM orchestration is optional (for standalone usage)
   - Proxy server is optional (for testing/small deployments only)

4. **Gateway-First Production Strategy**
   - Production deployments should use established AI gateways (written in TypeScript, Go, Rust)
   - ITS-Hub provides the algorithm logic only
   - Gateways handle production concerns (auth, rate limiting, monitoring)

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  its_hub (core)                                             │
│  - Pure algorithm implementations                           │
│  - Minimal dependencies (numpy, basic utils)                │
│  - Input: List of responses OR streaming step generator     │
│  - Output: Selected response(s) with scores                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ import algorithms only
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────┴────────┐                     ┌────────────┴──────────┐
│  its_hub[lm]   │                     │  AI Gateway           │
│  (optional)    │                     │  (TypeScript/Go/Rust) │
│  - LM clients  │                     │  - Uses its_hub algos │
│  - Retry logic │                     │  - Native LM clients  │
│  - Async pool  │                     │  - Production features│
└────────────────┘                     └───────────────────────┘
        │
        │ uses
        ▼
┌────────────────┐
│  its_hub[iaas] │
│  (optional)    │
│  - FastAPI svc │
│  - Testing only│
└────────────────┘
```

### Open Questions & Research Areas

1. **Cross-Language Integration**
   - Can Python-based algorithms be integrated into TypeScript/Go/Rust gateways?
   - Options: gRPC service, WebAssembly, native ports, subprocess calls
   - Performance implications of cross-language boundaries

2. **Step-wise Algorithm Support**
   - How to abstract step-wise generation (Beam Search, Particle Filtering)?
   - Current approach: `StepGeneration` class couples with LM implementation
   - Alternative: Callback/iterator interface that gateways implement?

3. **Reward Model Integration**
   - Process Reward Models (PRMs) currently tightly coupled to vLLM
   - How to make PRMs pluggable for different gateway backends?
   - Outcome reward models (LLM judges) easier to abstract

### Implementation Stages (To Be Refined)

**Stage 1: API Design & Contracts**
- [ ] Define clean algorithm interfaces independent of LM details
- [ ] Specify input/output contracts for each algorithm type
- [ ] Document integration patterns for gateway developers

**Stage 2: Core Algorithm Extraction**
- [ ] Extract algorithms to standalone modules with no LM dependencies
- [ ] Algorithms accept pre-generated responses OR generator callbacks
- [ ] Minimal dependencies (remove OpenAI, LiteLLM from core)

**Stage 3: Optional LM Package**
- [ ] Move `lms.py` to `its_hub.lm` subpackage
- [ ] Create `its_hub[lm]` extra in pyproject.toml
- [ ] Maintain current LM implementations for backward compatibility

**Stage 4: Optional IaaS Package**
- [ ] Move `integration/iaas.py` to `its_hub.iaas` subpackage
- [ ] Create `its_hub[iaas]` extra in pyproject.toml
- [ ] Mark as "testing/development only" in docs

**Stage 5: Documentation & Migration Guide**
- [ ] Update README to reflect algorithm-first positioning
- [ ] Create gateway integration guide
- [ ] Provide migration path for existing users

### Contributing New LM Engines

After refactoring, contributing new LM engines should be straightforward:
1. Implement the algorithm callback interface (to be defined)
2. Handle LM-specific API calls in your orchestration layer
3. No need to modify core algorithm code

For gateway integrations:
1. Import algorithm functions from `its_hub`
2. Implement orchestration using gateway's existing LM clients
3. Example: `result = self_consistency_vote(responses, projection_fn)`

---

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
  - Key methods: `agenerate()` (async), `generate()` (sync), `evaluate()` (optional)
- `AbstractScalingAlgorithm`: Base for all scaling algorithms
  - Key methods: `ainfer()` (async, must implement), `infer()` (sync wrapper)
  - Unified signature: `(lm, prompt_or_messages, budget, return_response_only, tools, tool_choice)`
- `AbstractScalingResult`: Base for algorithm results
  - Key property: `the_one` - returns the selected best response
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models (score complete responses)
- `AbstractProcessRewardModel`: Interface for process-based reward models (score step-by-step)

### Main Components

#### Language Models (`its_hub/lms.py`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs
  - Uses `backoff` library for exponential retry with configurable max attempts
  - Async generation with concurrency limits via semaphores
  - Supports both single and batch generation (list of message lists)
  - Error handling via `its_hub/error_handling.py`: distinguishes retryable vs non-retryable errors
- `LiteLLMLanguageModel`: Alternative LM supporting multi-cloud providers (AWS Bedrock, Vertex AI, etc.)
  - Unified interface for multiple cloud providers
  - Same retry and error handling as OpenAICompatibleLanguageModel
- `StepGeneration`: Incremental generation for process-based algorithms
  - Two modes: step_token (e.g., "\n\n") or tokens_per_step (fixed token count)
  - Configurable stop tokens and temperature switching
  - Post-processing to clean up step delimiters

#### Algorithms (`its_hub/algorithms/`)
All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True, tools=None, tool_choice=None)`

- **Self-Consistency** (`self_consistency.py`): Generate N responses, vote on most common answer
  - Supports regex-based voting via `create_regex_projection_function()`
  - Tool-call voting: `tool_vote` parameter ("tool_name", "tool_args", "tool_hierarchical")
  - Hierarchical voting with `exclude_args` for filtering non-semantic arguments
- **Best-of-N** (`bon.py`): Generate N responses, select highest-scoring via outcome reward model
  - Requires an `AbstractOutcomeRewardModel` (e.g., LLM judge)
- **Beam Search** (`beam_search.py`): Step-by-step generation with beam width
  - Requires `StepGeneration` and `AbstractProcessRewardModel`
  - Budget interpreted as total steps / beam_width
- **Particle Filtering/Gibbs** (`particle_gibbs.py`): Probabilistic resampling
  - Requires `StepGeneration` and `AbstractProcessRewardModel`
  - Budget = number of particles maintained during sampling
- **Planning Wrapper** (`planning_wrapper.py`): Meta-algorithm that wraps any other algorithm
  - Generates multiple solution approaches first, then runs base algorithm on each
  - Budget is distributed across approaches
  - Best approach selected by scoring final results

#### Integration (`its_hub/integration/`)
- `reward_hub.py`: Integrates with reward_hub library
  - `LocalVllmProcessRewardModel`: Process reward model for step-by-step scoring
  - Supports different aggregation methods: "prod", "sum", "last", "model"
- `iaas.py`: Inference-as-a-Service FastAPI server
  - Provides OpenAI-compatible `/v1/chat/completions` endpoint
  - Adds `budget` parameter for inference-time scaling
  - `/configure` endpoint to set up algorithm, models, and reward models
  - Global state management (LM_DICT, SCALING_ALG) - refactor for production use

### Budget Parameter
The budget parameter controls computational effort. Interpretation varies by algorithm:
- **Self-Consistency/Best-of-N**: Number of parallel generations to create (budget=4 → 4 responses)
- **Beam Search**: Total generations = budget / beam_width (budget=8, beam=2 → 4 steps)
- **Particle Filtering**: Number of particles maintained during sampling

### Tool Call Support
The library supports OpenAI-style tool calls throughout:
- Language models: `tools` and `tool_choice` parameters in generate/agenerate
- Self-Consistency: Special voting modes for tool calls (tool_name, tool_args, tool_hierarchical)
- Message format: Preserves tool_calls in response objects (dict with "tool_calls" key)
- Extract content: `extract_content_from_lm_response()` handles both text and tool call responses

### Typical Workflow

#### For Math Problems (using vLLM + Process Reward Model)
1. Start vLLM server: `CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8100`
2. Initialize language model pointing to vLLM
3. Create `StepGeneration` (e.g., `StepGeneration("\n\n", 32, r"\boxed")`)
4. Initialize process reward model: `LocalVllmProcessRewardModel("Qwen/Qwen2.5-Math-PRM-7B", "cuda:1")`
5. Create algorithm: `ParticleFiltering(step_generation, prm)` or `BeamSearch(step_generation, prm, beam_width=4)`
6. Call `infer()` with math problem and budget

#### For General Tasks (using IaaS)
1. Start IaaS service: `uv run its-iaas --port 8108`
2. Configure via POST to `/configure` with algorithm choice and model settings
3. Send requests to `/v1/chat/completions` with `budget` parameter
4. Service handles algorithm selection, generation, and reward scoring automatically

### Testing Structure (`tests/`)
- `conftest.py`: Shared fixtures for tests
- `mocks/`: Mock implementations for testing without real models
  - `language_models.py`: Mock LMs with deterministic responses
  - `reward_models.py`: Mock reward models for testing
  - `test_data.py`: Sample prompts and responses
- Test files follow naming: `test_<component>.py` (e.g., `test_algorithms.py`, `test_lms.py`)
- Use `pytest-asyncio` for async test support

### Mathematical Focus
The library is optimized for mathematical reasoning:
- System prompts: `SAL_STEP_BY_STEP_SYSTEM_PROMPT`, `QWEN_SYSTEM_PROMPT` in `utils.py`
- Regex patterns for answers: `r"\boxed{([^}]+)}"` for LaTeX boxed notation
- Integration with `math_verify` library for answer verification (optional dependency)
- Benchmark scripts for MATH500 and AIME-2024 datasets in `scripts/`

## Inference-as-a-Service (IaaS)

The its_hub library includes an IaaS service that provides OpenAI-compatible API with inference-time scaling capabilities. For comprehensive setup instructions, usage examples, and troubleshooting, see [docs/iaas-service.md](./docs/iaas-service.md).

### Key IaaS Configuration Patterns
- **Self-Consistency with tool voting**: Set `tool_vote="tool_hierarchical"` and `exclude_args` to filter timestamps/IDs
- **Best-of-N with LLM judge**: Set `rm_name="llm-judge"` and configure judge model settings
- **Multi-GPU setup**: vLLM on GPU 0, IaaS + reward model on GPU 1 (see docs/iaas-service.md)
- **LiteLLM provider support**: Use `provider="litellm"` with `endpoint="auto"` for multi-cloud (AWS Bedrock, Vertex AI, etc.)
- Reference `justfile` for extensive configuration examples (config-bon-openai, config-self-consistency-bedrock, etc.)

### Provider Configuration
- **OpenAI**: Standard endpoint `https://api.openai.com/v1` with API key
- **vLLM (local)**: Point to local server (e.g., `http://localhost:8100/v1`)
- **LiteLLM (multi-cloud)**: Set `provider="litellm"`, `endpoint="auto"`, pass cloud credentials via `extra_args`
  - AWS Bedrock: Pass `aws_access_key_id`, `aws_secret_access_key`, `aws_region_name` in `extra_args`
  - Model format: `bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0`

## Key File Locations

**Core Library:**
- `its_hub/base.py` - Abstract base classes for all components
- `its_hub/lms.py` - Language model implementations
- `its_hub/types.py` - Type definitions (ChatMessage, ChatMessages, etc.)
- `its_hub/utils.py` - Utility functions and system prompts
- `its_hub/error_handling.py` - Retry logic and error classification

**Algorithms:**
- `its_hub/algorithms/self_consistency.py` - Self-consistency voting
- `its_hub/algorithms/bon.py` - Best-of-N selection
- `its_hub/algorithms/beam_search.py` - Beam search (requires PRM)
- `its_hub/algorithms/particle_gibbs.py` - Particle filtering (requires PRM)
- `its_hub/algorithms/planning_wrapper.py` - Meta-algorithm wrapper

**Integration:**
- `its_hub/integration/iaas.py` - FastAPI server (entry point: `its-iaas` command)
- `its_hub/integration/reward_hub.py` - Reward model implementations

**Testing:**
- `tests/conftest.py` - Shared test fixtures
- `tests/mocks/` - Mock implementations for testing

**Documentation:**
- `docs/` - Docsify-based documentation site (https://ai-innovation.team/its_hub)
- `README.md` - Main project README
- `CLAUDE.md` - This file

**Scripts:**
- `scripts/benchmark.py` - Benchmarking on MATH500/AIME datasets
- `scripts/test_math_example.py` - Quick functionality test
- `justfile` - Extensive IaaS configuration and test examples