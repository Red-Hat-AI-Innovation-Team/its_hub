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

## Inference-as-a-Service (IaaS) Setup

The its_hub library includes a production-ready IaaS service that provides OpenAI-compatible API with inference-time scaling capabilities.

### Quick Start

```bash
# 1. Start vLLM server (main model on GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
  --dtype float16 --host 0.0.0.0 --port 8100

# 2. Start IaaS service (scaling + reward model on GPU 1)  
CUDA_VISIBLE_DEVICES=1 uv run its-iaas --host 0.0.0.0 --port 8108

# 3. Configure the service
curl -X POST http://localhost:8108/configure -H "Content-Type: application/json" -d '{
  "endpoint": "http://localhost:8100/v1",
  "api_key": "NO_API_KEY",
  "model": "Qwen/Qwen2.5-Math-1.5B-Instruct", 
  "alg": "best-of-n",
  "rm_name": "Qwen/Qwen2.5-Math-PRM-7B",
  "rm_device": "cuda:1",
  "rm_agg_method": "model"
}'
```

### Usage Example

```bash
# Generate response with inference-time scaling
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}],
    "budget": 4
  }'
```

### External Access via SSH Tunneling

For remote access to the service:

```bash
# Forward both services to local machine
ssh -L 8100:localhost:8100 -L 8108:localhost:8108 user@server-ip

# Then access locally
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Math-1.5B-Instruct", "messages": [{"role": "user", "content": "Hello"}], "budget": 2}'
```

### Service Architecture

```
Client → IaaS Service (GPU 1) → vLLM Server (GPU 0)
         ↓
    Reward Model (GPU 1)
         ↓
    Best Response Selection
```

### Key Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI chat completions
- **Budget Control**: `budget` parameter controls inference-time scaling intensity
- **Multi-GPU Support**: Distribute main model and reward model across GPUs
- **External Access**: SSH tunneling support for remote access
- **Production Ready**: Comprehensive error handling and logging

### Integration Examples

**Watson Orchestrate Integration:**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8108/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    messages=[{"role": "user", "content": "Solve this problem"}],
    extra_body={"budget": 4}
)
```

**Custom Python Client:**
```python
import requests

response = requests.post("http://localhost:8108/v1/chat/completions", json={
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "budget": 4  # Generate 4 responses, select best
})
```

### Performance Considerations

- **Response Time**: Scaling algorithms take longer than single generation
  - `budget=1`: ~2 seconds (no scaling)
  - `budget=4`: ~30-60 seconds (4x generation + reward scoring)
- **Quality Trade-off**: Higher budget = better quality but slower response
- **Memory Usage**: 
  - GPU 0: Main model (~74GB for Math-1.5B)
  - GPU 1: Reward model (~14GB for PRM-7B)

### Troubleshooting

Common issues and solutions:

```bash
# Check service status
ss -tlnp | grep 8108

# View service logs
tail -f iaas.log

# Check GPU usage
nvidia-smi

# Test basic connectivity
curl -X GET http://localhost:8108/docs
```

For comprehensive setup instructions, troubleshooting, and advanced configuration, see [IAAS_SERVICE_GUIDE.md](./IAAS_SERVICE_GUIDE.md).

### Available Algorithms

- **best-of-n**: Generate N responses, select highest scoring
- **particle-filtering**: Probabilistic sampling with resampling
- **beam-search**: Tree search with beam width control
- **self-consistency**: Multiple generations with majority voting

### Configuration Options

- `endpoint`: vLLM server URL
- `model`: Model name (must match vLLM)
- `alg`: Algorithm type
- `rm_name`: Reward model name
- `rm_device`: GPU device for reward model
- `rm_agg_method`: Reward aggregation method
- `budget`: Computational budget (1-1000)