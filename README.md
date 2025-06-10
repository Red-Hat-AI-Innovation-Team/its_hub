# A Python library for inference-time scaling LLMs

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)

**its_hub** provides inference-time scaling for LLMs through multiple approaches:

1. **Direct Library Usage** - For Python integration
2. **Inference-as-a-Service (IaaS) API** - OpenAI-compatible HTTP API (⚠️ Alpha)

## Direct Library Usage

Example: Using the particle filtering from `[1]` for inference-time scaling

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# NOTE launched via `CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --dtype float16`
lm = OpenAICompatibleLanguageModel(
    endpoint="http://0.0.0.0:8000/v1", 
    api_key="NO_API_KEY", 
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
)
prompt = r"Let $a$ be a positive real number such that all the roots of \[x^3 + ax^2 + ax + 1 = 0\]are real. Find the smallest possible value of $a.$" # question from MATH500
budget = 8

sg = StepGeneration("\n\n", 32, r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B", device="cuda:1", aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

scaling_alg.infer(lm, prompt, budget) # => gives output
```

## Inference-as-a-Service (IaaS) API ⚠️ Alpha

The IaaS integration provides an OpenAI-compatible HTTP API server that wraps inference-time scaling algorithms. This allows external applications to use scaling techniques through familiar chat completion endpoints.

### Starting the IaaS Server

```bash
# Start the API server
its-iaas --host 0.0.0.0 --port 8108 --dev

# Or using justfile (development)
just iaas-dev
```

### Configuring the Service

Before making chat completion requests, configure the service with your models and algorithms:

```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "http://localhost:8100/v1",
    "api_key": "NO_API_KEY", 
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "alg": "particle-filtering",
    "step_token": "\n",
    "stop_token": "<|end|>",
    "rm_name": "Qwen/Qwen2.5-Math-PRM-7B",
    "rm_device": "cuda:0",
    "rm_agg_method": "model"
  }'
```

### Making Requests

Use the standard OpenAI chat completions format with an additional `budget` parameter:

```bash
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}
    ],
    "budget": 8
  }'
```

### Supported Algorithms

- **particle-filtering**: Step-by-step generation with particle resampling
- **best-of-n**: Generate multiple responses, select the best using reward models

### API Endpoints

- `POST /configure` - Configure models and algorithms
- `GET /v1/models` - List available models (OpenAI-compatible)
- `POST /v1/chat/completions` - Generate chat completions with scaling (OpenAI-compatible + budget parameter)
- `GET /docs` - Interactive API documentation

### Development Recipes

The project includes a `justfile` with convenient development commands:

```bash
# Start vLLM server with different models
just serve-qwen    # Qwen2.5-Math-1.5B-Instruct
just serve-phi     # Phi-4-mini-instruct
just serve-llama   # Llama-3.2-1B-Instruct

# Run experiments
just exp-qwen budgets="8,16,32"
just exp-phi budgets="16"

# IaaS development
just iaas-dev                    # Start IaaS server
just iaas-dev-configure-pf       # Configure particle filtering
just iaas-dev-configure-bon      # Configure best-of-n
just iaas-dev-chat "solve 2+2"   # Test chat completion
```

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. "A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods", 2025, https://arxiv.org/abs/2502.01618.

## Installation

For development (recommended for running examples):
```sh
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

For production use:
```sh
pip install its_hub
```

## Quick Start Guide

This guide will help you run the example using a single H100 GPU. The example uses two models:
1. `Qwen/Qwen2.5-Math-1.5B-Instruct` (1.5B parameters) - Main model for math problem solving
2. `Qwen/Qwen2.5-Math-PRM-7B` (7B parameters) - Reward model for improving solution quality

Memory Requirements:
- Qwen2.5-Math-1.5B-Instruct: ~3GB GPU memory
- Qwen2.5-Math-PRM-7B: ~14GB GPU memory
- Total recommended GPU memory: 20GB or more (H100 80GB is ideal)

### 1. Environment Setup

First, create and activate a conda environment with Python 3.11:

```bash
conda create -n its_hub python=3.11
conda activate its_hub
```

Install the package in development mode (this includes all dependencies):

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

### 2. Starting the vLLM Server

First, identify your available GPU:
```bash
nvidia-smi
```

Start the vLLM server with optimized settings for H100 GPU (replace `$GPU_ID` with your GPU number, typically 0 if you have only one GPU). Note that we're using port 8100 as an example - you can change this if needed:

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dtype float16 \
    --port 8100 \  # This is the example port we'll use
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 128 \
    --tensor-parallel-size 1
```

### 3. Running the Example

The repository includes a test script in the `scripts` directory. You can find it at `scripts/test_math_example.py`. Here's what it contains:

```python
import os
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Get GPU ID from environment variable or default to 0
gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

# Initialize the language model
# Note: The endpoint port (8100) must match the port used when starting the vLLM server
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8100/v1",  # Make sure this matches your vLLM server port
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
)

# Test prompts
test_prompts = [
    "What is 2+2? Show your steps.",
    "Solve the quadratic equation x^2 + 5x + 6 = 0. Show your steps.",
    "Find the derivative of f(x) = x^2 + 3x + 2. Show your steps.",
    "Let a be a positive real number such that all the roots of x^3 + ax^2 + ax + 1 = 0 are real. Find the smallest possible value of a."
]

# Initialize step generation and reward model
sg = StepGeneration("\n\n", 32, r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device=f"cuda:{gpu_id}",  # Use the same GPU as the vLLM server
    aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

# Run tests
print("Testing Qwen Math Model with different approaches...")
print(f"Using GPU {gpu_id} with memory optimization settings\n")

for prompt in test_prompts:
    print(f"\nTesting: {prompt}")
    print("Response:", scaling_alg.infer(lm, prompt, budget=8))
```

Run the test script (make sure to use the same GPU as the server):

```bash
# From the its_hub directory
CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/test_math_example.py
```

### 4. Troubleshooting

If you encounter any issues:

1. **CUDA Out of Memory (OOM)**: 
   - The 7B reward model requires significant memory. If you encounter OOM errors:
     - Try reducing `--gpu-memory-utilization` to 0.6 or lower
     - Reduce `--max-num-seqs` to 64 or lower
     - Ensure no other processes are using the GPU
     - Consider using a smaller reward model if available
   - Monitor GPU memory usage with `nvidia-smi`

2. **Server Connection Issues**:
   - Verify the server is running with `curl http://localhost:8100/v1/models | cat`
   - Check if the port 8100 is available and not blocked by firewall
   - Ensure you're using the correct endpoint URL in the test script

3. **Model Loading Issues**:
   - Ensure you have enough disk space:
     - Qwen2.5-Math-1.5B-Instruct: ~3GB
     - Qwen2.5-Math-PRM-7B: ~14GB
   - Check your internet connection for model download
   - Verify you have the correct model names

### 5. Performance Notes

- The example uses two models:
  1. Qwen2.5-Math-1.5B-Instruct (1.5B parameters) for solving math problems
  2. Qwen2.5-Math-PRM-7B (7B parameters) for improving solution quality
- Total GPU memory requirement is about 20GB, making it suitable for:
  - H100 80GB (recommended)
  - A100 40GB (with reduced batch size)
  - Other GPUs with 20GB+ memory (with further optimizations)
- Memory usage is optimized to prevent OOM errors while maintaining good performance
- The particle filtering algorithm helps improve the quality of mathematical reasoning
- Response times may vary depending on the complexity of the math problem

## Benchmark

There is a script at `scripts/benchmark.py` that can be used to benchmark inference-time scaling algorithms.
The CLI of the script is self-contained so the usage can be checked via `python scripts/benchmark.py --help`.
Example command:
```
python scripts/benchmark.py --benchmark aime-2024 --model_name Qwen/Qwen2.5-Math-1.5B-Instruct --alg particle-filtering --rm_device cuda:1 --endpoint http://0.0.0.0:8000/v1 --shuffle_seed 1110 --does_eval --budgets 1,2,4,8,16,32,64 --rm_agg_method model
```

## Development

```sh
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
pytest tests
```

