# Algorithms

its-hub provides several inference-time scaling algorithms, each optimized for different use cases and computational budgets.

## Overview

All algorithms follow the same interface:
- **Async (primary):** `ainfer(lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None)`
- **Sync wrapper:** `infer(...)` (calls `asyncio.run(ainfer(...))`)

The `budget` parameter controls computational resources allocated to each algorithm, with different interpretations:

| Algorithm | Budget Interpretation | Snippet |
|-----------|----------------------|---------|
| Self-Consistency | Number of parallel generations | `SelfConsistency()` |
| Best-of-N | Number of candidate responses | `BestOfN(rm)` |
| Beam Search | Total generations ÷ beam width | `BeamSearch(sg, prm, beam_width=4)` |
| Particle Filtering | Number of particles | `ParticleFiltering(sg, prm)` |
| Entropic Particle Filtering | Number of particles | `EntropicParticleFiltering(sg, prm)` |
| Beta Self-Consistency (experimental) | Maximum number of parallel generations | `BetaSelfConsistency()` |
| Adaptive Self-Consistency | Maximum number of parallel generations | `AdaptiveSelfConsistency()` |
| Confidence Selection | Number of parallel generations | `ConfidenceSelection(metric="entropy")` |
| Weighted Self-Consistency | Number of parallel generations | `WeightedSelfConsistency(metric="entropy")` |

## Self-Consistency

<video src="https://github.com/user-attachments/assets/f6239b67-15b6-4073-a0a2-73269328cf38" width="100%" controls muted playsinline></video>

Generates multiple responses and selects the most common answer through voting. **Especially powerful for tool-calling** where you want consistent tool usage patterns.

### Tool Calling Example (Recommended)

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

**Tool voting modes:**
- `"tool_name"`: Vote on which tool to call
- `"tool_args"`: Vote on tool arguments
- `"tool_hierarchical"` (recommended): First vote on tool name, then on arguments
- `exclude_args=["timestamp", "id"]`: Exclude non-semantic arguments from voting

### Text-Based Example

```python
# For mathematical problems with regex extraction
def extract_boxed(text):
    import re
    matches = re.findall(r'\\boxed\{([^{}]+)\}', text)
    return matches[-1] if matches else ""

sc = SelfConsistency(consistency_space_projection_func=extract_boxed)
result = sc.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=4)
```

**When to use:**
- Tool-calling applications (agents, function calling)
- Mathematical problems with clear final answers
- Tasks where multiple reasoning approaches are valid
- When you need fast inference with improved accuracy

## Best-of-N

<video src="https://github.com/user-attachments/assets/6f6e69bb-9540-4681-9e81-edda7c59fdfb" width="100%" controls muted playsinline></video>

Generates N candidate responses and selects the highest-scoring one using a reward model. **Works with both text and tool-calling responses.**

### With LLM Judge (Cloud APIs)

```python
import asyncio

from its_hub import BestOfN, LLMJudge, OpenAICompatibleLanguageModel

# Initialize language model
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini"
)

# Set up LLM judge for scoring
judge = LLMJudge(lm=lm)

# Best-of-N with LLM judge
bon = BestOfN(judge)

# Works with tool calls
tools = [{"type": "function", "function": {...}}]
result = bon.infer(
    lm,
    messages,
    budget=4,
    tools=tools,
    tool_choice="auto"
)

# Close lm for resource cleanup
asyncio.run(lm.close())
```

### With Local Process Reward Model

```python
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

# Initialize reward model (requires GPU)
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)

bon = BestOfN(prm)
result = bon.infer(lm, prompt, budget=16)
```

**When to use:**
- Tool-calling applications where quality matters most
- When you have a reliable reward model
- Quality is more important than speed
- Tasks where ranking responses is straightforward

## Beam Search

Performs step-by-step generation with beam width control, using process reward models to guide the search.

```python
from its_hub import StepGeneration
from its_hub.core.algorithms.beam_search import BeamSearch
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

# Initialize components
sg = StepGeneration(max_steps=32, step_token="\n\n", stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)

# Beam search with beam width of 4
beam_search = BeamSearch(sg, prm, beam_width=4)
result = beam_search.infer(lm, prompt, budget=32)  # 32 total generations
```

**Budget calculation:** `budget = beam_width × number_of_steps`

**When to use:**
- Step-by-step reasoning problems
- When you can evaluate partial solutions
- Mathematical proofs or derivations

## Particle Filtering

<video src="https://github.com/user-attachments/assets/5d37fb3b-acf9-4c3d-a15f-187414237e34" width="100%" controls muted playsinline></video>

Uses probabilistic resampling to maintain diverse reasoning paths while focusing on promising directions.

```python
from its_hub import StepGeneration
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

# Initialize components
sg = StepGeneration(max_steps=32, step_token="\n\n", stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)

# Particle filtering with 8 particles
pf = ParticleFiltering(sg, prm)
result = pf.infer(lm, prompt, budget=8)
```

**When to use:**
- Complex reasoning tasks with multiple valid approaches
- When exploration vs exploitation balance is important
- Mathematical problem solving with uncertainty


## Entropic Particle Filtering

Entropic Particle Filtering (ePF) is an advanced sampling algorithm that mitigates common failure modes in standard PF, like particle degeneracy and impoverishment. 
By leveraging Entropic Annealing (EA) to control the variance of the resampling distribution, ePF ensures a more robust and thorough exploration in the early phase of sampling, especially for complex long sequences and multi-step tasks.

```python
from its_hub import StepGeneration
from its_hub.core.algorithms.particle_gibbs import EntropicParticleFiltering
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

# Initialize components
sg = StepGeneration(max_steps=32, step_token="\n\n", stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)

# Entropic particle filtering with 8 particles
epf = EntropicParticleFiltering(sg, prm)
result = epf.infer(lm, prompt, budget=8)
```

**When to use:**
- When Reward Models Are Hard to Calibrate: 
    - If your Process Reward Model (PRM) tends to be *overconfident* early in the sampling process, ePF helps by keeping a wider range of options open for longer.

- For Complex, Long Multi-Step Tasks: 
    - When a problem requires many sequential steps to solve (> 20 steps), standard particle filters can lose diversity and generate greedy-like solutions. ePF is designed to handle these long-horizon tasks more effectively.

- To Avoid Early Convergence: 
    - If you notice that a standard filter is producing short, incomplete responses or underperforming, it is likely converging prematurely. ePF directly counteracts this by promoting particle diversity.


### Beta Self-Consistency

Beta Self-Consistency is an adaptive variant of Self-Consistency. It starts up to `budget` generations, evaluates the Beta posterior as responses complete, and cancels pending generations once the leading answer reaches `confidence_threshold` (default `0.95`). 

```python
from its_hub import BetaSelfConsistency

beta_sc = BetaSelfConsistency(confidence_threshold=0.95)
result = beta_sc.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=64)
```

The `budget` is the maximum number of generations. Like `SelfConsistency`, this experimental algorithm supports response projection and tool voting.

**Reference:** Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. 2023. [“Let’s Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs.”](https://aclanthology.org/2023.emnlp-main.761/) In *Proceedings of EMNLP 2023*, pages 12375–12396. Association for Computational Linguistics.


## Adaptive Self-Consistency

Self-Consistency with exponential doubling and early stopping. Starts with 2 samples, checks if a supermajority agree, and doubles the sample count each round until `budget` is reached or consensus is achieved. Previous samples are kept across rounds — no resampling.

```python
from its_hub import AdaptiveSelfConsistency

# Stop early when 75% of samples agree (default)
algo = AdaptiveSelfConsistency(supermajority_threshold=0.75)
result = algo.infer(lm, “Solve x^2 + 5x + 6 = 0”, budget=64)
```

Inherits `tool_vote`, `exclude_args`, and `consistency_space_projection_func` from `SelfConsistency`.

**When to use:**
- When many problems are easy and full budget would be wasteful
- High-throughput pipelines where average latency matters more than worst-case
- Drop-in replacement for `SelfConsistency` — same interface, lower average cost


## Confidence Selection

Selects the single best response based on the model’s own confidence, measured from token-level logprobs. No reward model needed.

```python
from its_hub import ConfidenceSelection

# Entropy metric (default) — lower tail entropy = more confident
algo = ConfidenceSelection(metric=”entropy”)
result = algo.infer(lm, “What is 2+2?”, budget=8)

# Self-certainty metric — higher KL(uniform || model) = more confident
algo = ConfidenceSelection(metric=”certainty”)
result = algo.infer(lm, “What is 2+2?”, budget=8)
```

The algorithm generates `budget` candidates with `logprobs=True`, computes per-token confidence scores from the top-k log probabilities, aggregates over an adaptive tail window, and picks the candidate with the best score.

**Key parameters:**
- `metric`: `”entropy”` (default) or `”certainty”`
- `top_logprobs`: Number of top logprobs to request (1–20, default 20)
- `agg`: Tail aggregation — `”median”` (default, robust) or `”mean”`

**When to use:**
- When logprobs are available but no reward model is
- Single-best-answer selection without voting
- Quick confidence-based filtering before more expensive evaluation


## Weighted Self-Consistency

Confidence-weighted majority voting, inspired by [DeepConf](https://arxiv.org/abs/2508.15260). Each candidate’s vote is weighted by its tail confidence score from logprobs, fusing the consistency signal (agreement across responses) with the confidence signal (model-internal certainty).

```python
from its_hub import WeightedSelfConsistency

# Entropy-weighted voting (default)
algo = WeightedSelfConsistency(metric=”entropy”)
result = algo.infer(lm, “What is 2+2?”, budget=16)

# With custom answer projection
import re

def extract_boxed(text):
    matches = re.findall(r’\\boxed\{([^{}]+)\}’, text)
    return matches[-1] if matches else “”

algo = WeightedSelfConsistency(
    consistency_space_projection_func=extract_boxed,
    metric=”certainty”,
)
result = algo.infer(lm, prompt, budget=32, return_response_only=False)
print(result.group_weights)  # {“42”: 5.83, “99”: 0.41}
```

**Degeneration properties:**
- When all candidates have **equal confidence** → degenerates to standard majority voting (`SelfConsistency`)
- When all candidates **disagree** → degenerates to picking the most confident response (`ConfidenceSelection`)

**When to use:**
- When a confident minority may have the right answer but would lose a simple vote
- Mathematical reasoning where some reasoning paths are more decisive than others
- Drop-in improvement over `SelfConsistency` when logprobs are available


## Advanced Configuration

### Step Generation

The `StepGeneration` class enables incremental text generation:

```python
from its_hub import StepGeneration

# For math problems with boxed answers
sg = StepGeneration(
    max_steps=32,               # Maximum number of steps
    step_token="\n\n",          # Split reasoning into steps
    stop_token=r"\boxed",       # Stop when final answer is found
)
```

### Reward Models

#### Process Reward Models
Evaluate reasoning steps incrementally:

```python
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"  # or "mean", "min", "max"
)
```

#### Outcome Reward Models
Evaluate final answers only:

```python
from its_hub import AbstractOutcomeRewardModel

# Custom outcome reward model
class MathOutcomeRewardModel(AbstractOutcomeRewardModel):
    def score(self, messages, **kwargs):
        # Extract answer and compute reward
        return score
```

## Resource Cleanup

When using `OpenAICompatibleLanguageModel`, always close the LM after use to clean up HTTP sessions:

```python
# Option 1: Async context manager (recommended)
async with OpenAICompatibleLanguageModel(...) as lm:
    result = await algorithm.ainfer(lm, prompt, budget=5)

# Option 2: Explicit close after sync usage
lm = OpenAICompatibleLanguageModel(...)
result = algorithm.infer(lm, prompt, budget=5)
asyncio.run(lm.close())
```

## Performance Tips

1. **Start with Self-Consistency** for quick improvements
2. **Try Adaptive Self-Consistency** to save compute on easy problems
3. **Use Confidence Selection** when logprobs are available but no reward model
4. **Use Weighted Self-Consistency** to combine voting with confidence signals
5. **Use Best-of-N** when you have a good reward model
6. **Try Beam Search** for step-by-step reasoning
7. **Use Particle Filtering** for the most complex problems
8. **Use Entropic Particle Filtering** to mitigate early exploitation
9. **Adjust budget** based on problem complexity and time constraints
10. **Monitor GPU memory** when using large reward models