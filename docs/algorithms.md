# Algorithms

its-hub provides several inference-time scaling algorithms, each optimized for different use cases and computational budgets.

## Overview

All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True)`

The `budget` parameter controls computational resources allocated to each algorithm, with different interpretations:

| Algorithm | Budget Interpretation | Snippet |
|-----------|----------------------|---------|
| Self-Consistency | Number of parallel generations | `SelfConsistency()` |
| Best-of-N | Number of candidate responses | `BestOfN(rm)` |
| Beam Search | Total generations ÷ beam width | `BeamSearch(sg, prm, beam_width=4)` |
| Particle Filtering | Number of particles | `ParticleFiltering(sg, prm)` |

## Self-Consistency

Generates multiple responses and selects the most common answer. Effective for tasks where multiple reasoning paths can lead to the same correct answer.

```python
from its_hub.algorithms import SelfConsistency

# Simple self-consistency without step generation
sc = SelfConsistency()
result = sc.infer(lm, prompt, budget=8)

# With step generation for better reasoning
from its_hub.lms import StepGeneration
sg = StepGeneration("\n\n", max_steps=32, stop_pattern=r"\boxed")
sc = SelfConsistency(sg)
result = sc.infer(lm, prompt, budget=8)
```

**When to use:**
- Mathematical problems with clear final answers
- Tasks where multiple reasoning approaches are valid
- When you need fast inference with improved accuracy

## Best-of-N

Generates N candidate responses and selects the highest-scoring one using a reward model.

```python
from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize reward model
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"
)

# Best-of-N with reward model scoring
bon = BestOfN(reward_model=prm)
result = bon.infer(lm, prompt, budget=16)
```

**When to use:**
- When you have a reliable reward model
- Quality is more important than speed
- Tasks where ranking responses is straightforward

## Beam Search

Performs step-by-step generation with beam width control, using process reward models to guide the search.

```python
from its_hub.algorithms import BeamSearch
from its_hub.lms import StepGeneration
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize components
sg = StepGeneration("\n\n", max_steps=32, stop_pattern=r"\boxed")
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

Uses probabilistic resampling to maintain diverse reasoning paths while focusing on promising directions.

```python
from its_hub.algorithms import ParticleFiltering
from its_hub.lms import StepGeneration
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Initialize components
sg = StepGeneration("\n\n", max_steps=32, stop_pattern=r"\boxed")
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

## Advanced Configuration

### Step Generation

The `StepGeneration` class enables incremental text generation:

```python
from its_hub.lms import StepGeneration

# For math problems with boxed answers
sg = StepGeneration(
    step_token="\n\n",        # Split reasoning into steps
    max_steps=32,               # Maximum number of steps
    stop_pattern=r"\boxed",    # Stop when final answer is found
    post_process=True           # Clean up output formatting
)
```

### Reward Models

#### Process Reward Models
Evaluate reasoning steps incrementally:

```python
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"  # or "mean", "min", "max"
)
```

#### Outcome Reward Models
Evaluate final answers only:

```python
# Custom outcome reward model
class MathOutcomeRewardModel:
    def score(self, prompt, response):
        # Extract answer and compute reward
        return score
```

## Performance Tips

1. **Start with Self-Consistency** for quick improvements
2. **Use Best-of-N** when you have a good reward model
3. **Try Beam Search** for step-by-step reasoning
4. **Use Particle Filtering** for the most complex problems
5. **Adjust budget** based on problem complexity and time constraints
6. **Monitor GPU memory** when using large reward models