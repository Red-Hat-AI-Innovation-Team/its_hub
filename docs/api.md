# API Reference

## Core Classes

### AbstractLanguageModel

Base interface for language model implementations.

```python
from its_hub.base import AbstractLanguageModel

class AbstractLanguageModel:
    def generate(self, prompt: str) -> str:
        """Generate a single response"""
        pass
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for multiple prompts"""
        pass
    
    def score(self, prompt: str, response: str) -> float:
        """Score a response given a prompt"""
        pass
```

### OpenAICompatibleLanguageModel

Primary implementation supporting vLLM and OpenAI APIs.

```python
from its_hub.lms import OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1",
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    system_prompt="You are a helpful assistant.",
    max_concurrent_requests=10,
    request_timeout=60.0,
    backoff_factor=2.0,
    max_retries=3
)
```

**Parameters:**
- `endpoint`: API endpoint URL
- `api_key`: API key (use "NO_API_KEY" for local servers)
- `model_name`: Model identifier
- `system_prompt`: System message for chat models
- `max_concurrent_requests`: Concurrency limit for async requests
- `request_timeout`: Request timeout in seconds
- `backoff_factor`: Exponential backoff multiplier for retries
- `max_retries`: Maximum number of retry attempts

### AbstractScalingAlgorithm

Base class for all scaling algorithms.

```python
from its_hub.base import AbstractScalingAlgorithm

class AbstractScalingAlgorithm:
    def infer(self, lm: AbstractLanguageModel, prompt: str, budget: int, 
              return_response_only: bool = True) -> str | AbstractScalingResult:
        """Perform inference with given budget"""
        pass
```

## Algorithms

### SelfConsistency

```python
from its_hub.algorithms import SelfConsistency

sc = SelfConsistency(step_generation=None)
result = sc.infer(lm, prompt, budget=8)
```

### BestOfN

```python
from its_hub.algorithms import BestOfN

bon = BestOfN(reward_model=None, step_generation=None)
result = bon.infer(lm, prompt, budget=16)
```

### BeamSearch

```python
from its_hub.algorithms import BeamSearch

beam = BeamSearch(step_generation, process_reward_model, beam_width=4)
result = beam.infer(lm, prompt, budget=32)
```

### ParticleFiltering

```python
from its_hub.algorithms import ParticleFiltering

pf = ParticleFiltering(step_generation, process_reward_model)
result = pf.infer(lm, prompt, budget=8)
```

## Step Generation

### StepGeneration

Handles incremental text generation with configurable step tokens.

```python
from its_hub.lms import StepGeneration

sg = StepGeneration(
    step_token="\\n\\n",           # Token marking step boundaries
    max_steps=32,                  # Maximum number of steps
    stop_pattern=r"\\boxed",       # Regex pattern to stop generation
    post_process=True,             # Enable post-processing
    include_stop_in_output=False   # Include stop pattern in output
)
```

**Methods:**
- `generate_step(lm, prompt, context="")`: Generate next step
- `generate_full(lm, prompt)`: Generate complete response with steps
- `post_process_response(response)`: Clean up generated text

## Reward Models

### LocalVllmProcessRewardModel

Process reward model using local vLLM inference.

```python
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod",    # "prod", "mean", "min", "max", "model"
    torch_dtype="float16",
    trust_remote_code=True
)
```

**Methods:**
- `score_steps(prompt, steps)`: Score individual reasoning steps
- `score_response(prompt, response)`: Score complete response

## Utilities

### System Prompts

Predefined system prompts optimized for mathematical reasoning:

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT

# Step-by-step reasoning prompt
lm = OpenAICompatibleLanguageModel(
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    ...
)
```

### Response Processing

Utilities for extracting answers from mathematical responses:

```python
from its_hub.utils import extract_boxed_answer

answer = extract_boxed_answer(response)  # Extract \\boxed{...} content
```

## IaaS API

### FastAPI Server

The Inference-as-a-Service API provides OpenAI-compatible endpoints:

```python
from its_hub.integration.iaas import create_app

app = create_app()
# Serves at /v1/chat/completions with budget parameter
```

### Configuration Endpoint

Configure the service before making requests:

```http
POST /configure
Content-Type: application/json

{
    "endpoint": "http://localhost:8100/v1",
    "api_key": "NO_API_KEY",
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "alg": "particle-filtering",
    "step_token": "\\n\\n",
    "stop_token": "<|end|>",
    "rm_name": "Qwen/Qwen2.5-Math-PRM-7B",
    "rm_device": "cuda:0",
    "rm_agg_method": "prod"
}
```

### Chat Completions

OpenAI-compatible chat completions with budget parameter:

```http
POST /v1/chat/completions
Content-Type: application/json

{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [
        {"role": "user", "content": "Solve x^2 + 5x + 6 = 0"}
    ],
    "budget": 8
}
```

## Error Handling

### Common Exceptions

```python
from its_hub.base import ScalingError

try:
    result = algorithm.infer(lm, prompt, budget)
except ScalingError as e:
    print(f"Scaling algorithm failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Async Error Handling

```python
import asyncio
from its_hub.lms import OpenAICompatibleLanguageModel

async def safe_generate():
    try:
        response = await lm.generate_async(prompt)
        return response
    except asyncio.TimeoutError:
        print("Request timed out")
    except Exception as e:
        print(f"Generation failed: {e}")
```