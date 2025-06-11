# Usage

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
```

### Configuring the Service

Before making chat completion requests, configure the service with your models and algorithms.

**Particle Filtering Configuration:**
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

**Best-of-N Configuration:**
```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "http://localhost:8100/v1",
    "api_key": "NO_API_KEY", 
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "alg": "best-of-n",
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

### Testing the Service

You can test both the raw model and the IaaS service to compare results:

```bash
# Test raw model directly
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}]
  }'

# Test with inference-time scaling
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}],
    "budget": 8
  }'
```

---

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. "A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods", 2025, https://arxiv.org/abs/2502.01618.