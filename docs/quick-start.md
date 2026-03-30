# Quick Start Guide

This guide shows examples of inference-time scaling. **Tool calling is the primary use case** for production applications.

## Example 1: Self-Consistency with Tool Calling (Recommended)

**Installation required:** `pip install its_hub`

This example shows how to use Self-Consistency for reliable tool calling in agent applications.

```python
import asyncio

from its_hub import OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

# Initialize language model
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini"
)

# Define tools (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Create messages
messages = ChatMessages([
    ChatMessage(
        role="system",
        content="You are a precise calculator. Always use the calculator tool for arithmetic."
    ),
    ChatMessage(
        role="user",
        content="What is 847 * 293 + 156?"
    )
])

# Use hierarchical tool voting
sc = SelfConsistency(tool_vote="tool_hierarchical")
result = sc.infer(
    lm,
    messages,
    budget=5,
    tools=tools,
    tool_choice="auto"
)
print(result)

# Close lm for resource cleanup
asyncio.run(lm.close())
```

**What happens:**
1. Generates 5 different responses with tool calls
2. Votes on tool names first (which tool to use)
3. Then votes on tool arguments (what parameters to pass)
4. Returns the most consistent tool call

---

## Example 2: Best-of-N with LLM Judge (Core Installation)

**Installation required:** `pip install its_hub`

This example uses Best-of-N algorithm with an LLM judge for response selection. Works with any OpenAI-compatible API and requires no GPU.

```python
import asyncio

from its_hub import BestOfN, OpenAICompatibleLanguageModel
from its_hub.core.reward_models.llm_judge import LLMJudge

# Initialize language model
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

# Set up LLM judge for scoring
judge = LLMJudge(lm=lm)
scaling_alg = BestOfN(judge)

# Generate multiple responses and select the best
result = scaling_alg.infer(
    lm,
    "Explain quantum entanglement in simple terms",
    budget=4
)
print(result)

# Close lm for resource cleanup
asyncio.run(lm.close())
```

**What happens:**
1. Generates 4 different responses to the prompt
2. LLM judge scores all responses
3. Returns the highest-scoring response

---

## Example 3: Particle Filtering with Process Reward Model

**Installation required:** `pip install its_hub[experimental]`

This example uses Particle Filtering for step-by-step mathematical reasoning with a local process reward model. Requires GPU.

### Prerequisites

- GPU with CUDA 11.8+
- 20GB+ GPU memory recommended (for 7B reward model)

### Step 1: Start vLLM Server

Start a vLLM server with your instruction model:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
    --dtype float16 \
    --port 8100 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7
```

### Step 2: Run Particle Filtering

```python
import asyncio

from its_hub import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel
from its_hub.core.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

# Initialize language model (points to vLLM server)
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8100/v1",
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
)

# Set up step generation and process reward model
sg = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

# Solve with step-by-step reasoning
result = scaling_alg.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=8)
print(result)

# Close lm for resource cleanup
asyncio.run(lm.close())
```

**What happens:**
1. Generates reasoning steps incrementally (separated by `\n\n`)
2. Process reward model scores each step
3. Maintains 8 particles, resampling based on scores
4. Returns best solution when `\boxed` pattern is found

### Memory Requirements

- **Instruction Model** (Qwen2.5-Math-1.5B): ~3GB GPU memory
- **Reward Model** (Qwen2.5-Math-PRM-7B): ~14GB GPU memory
- **Total**: 20GB+ recommended (H100, A100, or similar)

---

## Troubleshooting

### Examples 1 & 2 (Cloud APIs)

**API errors**: Verify API key and endpoint are correct
**Slow responses**: Reduce `budget` parameter (e.g., from 5 to 2)

### Example 3 (Particle Filtering)

**CUDA Out of Memory**:
- Reduce `--gpu-memory-utilization` to 0.6 or lower
- Reduce `--max-num-seqs` to 64
- Ensure no other processes are using GPU
- Check memory: `nvidia-smi`

**Server Connection Issues**:
```bash
# Verify vLLM server is running
curl http://localhost:8100/v1/models
```

**Model Loading Issues**:
- Ensure sufficient disk space (~20GB for both models)
- Check internet connection for model downloads
- Verify model names are correct

---

## Next Steps

- **Explore algorithms**: See [Algorithms](algorithms.md) for Beam Search, Self-Consistency, and other approaches
- **Deploy as API**: See [IaaS Service Guide](iaas-service.md) to deploy as OpenAI-compatible service
- **Contribute**: See [Development](development.md) for contribution guidelines