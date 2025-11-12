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
uv run its-iaas --host 0.0.0.0 --port 8108

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

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with unified `infer()` method
- `AbstractScalingResult`: Base for algorithm results with `the_one` property
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models
- `AbstractProcessRewardModel`: Interface for process-based reward models (step-by-step scoring)

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
- `iaas.py`: Inference-as-a-Service FastAPI server providing OpenAI-compatible chat completions API with budget parameter for inference-time scaling

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
6. Call `infer()` with prompt and budget

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets

## Inference-as-a-Service (IaaS)

The its_hub library includes an IaaS service that provides OpenAI-compatible API with inference-time scaling capabilities. For comprehensive setup instructions, usage examples, and troubleshooting, see [docs/iaas-service.md](./docs/iaas-service.md).

## Envoy External Processor (ext_proc) Gateway

The Envoy ext_proc integration provides a **transparent gateway** for applying inference-time scaling to any LLM API without code changes. It uses Envoy's External Processor filter to intercept HTTP requests and apply ITS algorithms.

### Architecture

```
Client → Envoy (port 8108) → ext_proc gRPC (port 50051) → LLM API
```

**Key Components:**
- `its_hub/integration/ext_proc/processor.py`: gRPC service implementing Envoy's External Processor protocol
- `its_hub/integration/orchestrator.py`: Stateless orchestrator managing ITS execution
- `config/envoy/ext_proc.yaml`: Envoy proxy configuration

### Development Commands

```bash
# Initial setup (compile protobuf files)
just setup

# Start Envoy proxy
just envoy-start

# Start ext_proc gRPC service (in separate terminal)
just envoy-grpc-start

# Test with sample requests
just envoy-grpc-test

# Check Envoy health and statistics
just envoy-health

# Test with OpenAI API (requires OPENAI_API_KEY)
just test-chat
```

### How It Works

**Request Flow:**
1. Client sends request to Envoy (port 8108)
2. Envoy sends headers to ext_proc via gRPC
3. ext_proc checks for ITS headers (`X-ITS-Budget`, `X-ITS-Endpoint`)
4. If ITS headers present:
   - Envoy sends request body to ext_proc
   - ext_proc extracts model from body
   - ext_proc runs ITS algorithm (self-consistency)
   - ext_proc returns aggregated response to Envoy
   - Envoy returns response to client (upstream is bypassed)
5. If no ITS headers:
   - ext_proc signals "continue"
   - Envoy forwards request to upstream LLM unchanged

**Fail-Safe Behavior:**
- If ext_proc service is down → requests pass through to upstream
- If ext_proc encounters error → request passes through to upstream
- Configuration: `failure_mode_allow: true` in Envoy config

### ITS Header API

Configure ITS via HTTP headers (per-request):

| Header | Required | Description | Example |
|--------|----------|-------------|---------|
| `X-ITS-Budget` | Yes | Number of LLM calls | `3` |
| `X-ITS-Endpoint` | Yes | Full LLM API base URL with protocol | `https://api.openai.com/v1` |
| `X-ITS-API-Key` | No | API key for LLM authentication | `sk-...` |

**Important:** Model is specified in request body (standard OpenAI format), NOT in headers.

### Testing Strategy

**Unit Tests (Manual gRPC):**
```bash
# Test ext_proc directly without Envoy
just envoy-grpc-test
```

This runs `scripts/test_envoy_grpc.py` which simulates Envoy's gRPC protocol:
- Test 1: Request WITH ITS headers
- Test 2: Request WITHOUT ITS headers (passthrough)
- Test 3: Request with ITS headers but NO model in body
- Test 4: Request to non-chat endpoint

**Integration Tests (with Envoy):**
```bash
# Terminal 1: Start Envoy
just envoy-start

# Terminal 2: Start ext_proc
just envoy-grpc-start

# Terminal 3: Test end-to-end
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ITS-Budget: 3" \
  -H "X-ITS-Endpoint: https://api.openai.com/v1" \
  -H "X-ITS-API-Key: $OPENAI_API_KEY" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]}'
```

### Configuration Files

**Envoy Config (`config/envoy/ext_proc.yaml`):**
- Listener on port 8108
- ext_proc filter pointing to port 50051
- Upstream LLM cluster on port 8100 (fallback)
- Admin interface on port 9901
- Key setting: `processing_mode.request_body_mode: BUFFERED` (required for body access)

**Important:** Do NOT use quoted enum values in `processing_mode` (e.g., use `SEND` not `"SEND"`).

### Protobuf Compilation

The ext_proc service uses Envoy's protobuf definitions via git submodules:

```bash
# Initialize submodules
just submodule-init

# Compile proto files to Python
just proto-compile
```

Generated files: `its_hub/integration/ext_proc/proto/` (not committed to git)

### Common Issues & Solutions

**Issue: Envoy not sending request body to ext_proc**
- **Symptom**: Logs show "ITS headers detected" but no "Request body complete"
- **Cause**: Missing or incorrect `processing_mode` configuration
- **Solution**: Ensure `request_body_mode: BUFFERED` in Envoy config

**Issue: Invalid URL error (`aiohttp.client_exceptions.InvalidUrlClientError`)**
- **Symptom**: Error message shows `api.openai.com/chat/completions`
- **Cause**: Missing protocol in `X-ITS-Endpoint` header
- **Solution**: Use full URL with protocol: `https://api.openai.com/v1`

**Issue: 503 Service Unavailable**
- **Symptom**: Requests fail with 503 status
- **Cause**: ext_proc service not running OR upstream LLM not available
- **Check**: Run `just envoy-health` to verify cluster status

### Key Design Decisions

**Stateless Per-Request:**
- Unlike IaaS which uses POST /configure for global state
- All configuration via headers on each request
- Enables multi-tenant deployments

**Model from Body:**
- Model extracted from request body's `"model"` field
- Follows standard OpenAI API conventions
- Temperature removed (not needed for ITS)

**Self-Consistency Only (Phase 1):**
- Currently only implements self-consistency algorithm
- Future: Add `X-ITS-Algorithm` header for algorithm selection

### Documentation References

- **Detailed handover**: `its_hub/integration/ext_proc/HANDOVER.md`
- **Envoy ext_proc docs**: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter
- **Architecture**: See "Integration" section above