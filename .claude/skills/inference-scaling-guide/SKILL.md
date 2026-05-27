---
name: inference-scaling-guide
description: Guides users through inference-time scaling with its_hub, including algorithm selection (Self-Consistency, Best-of-N, Beam Search, Particle Filtering), budget tuning, reward model setup, tool-calling integration, interpreting results, and troubleshooting. Use when the user is working with its_hub, asking about scaling algorithms, debugging scaling issues, or tuning inference quality.
---

# Inference-Time Scaling Guide

its_hub generates multiple LLM responses and selects the best one using voting, scoring, or search. All algorithms share the same interface: `ainfer(lm, prompt, budget)` (async) or `infer(...)` (sync).

For API reference and conceptual overviews, consult the docs at https://ai-innovation.team/its_hub and the `docs/` directory. This skill covers practical knowledge, decision frameworks, and troubleshooting.

## Algorithm Selection

| Need | Algorithm | Why |
|------|-----------|-----|
| Fast improvement, tool calling | **Self-Consistency** | Voting is cheap, no reward model needed, excellent for tool-call consensus |
| Highest quality single response | **Best-of-N** | Scores every candidate, picks the best — requires a reward model |
| Step-by-step reasoning | **Beam Search** | Evaluates partial solutions at each step — requires process reward model + GPU |
| Complex multi-path reasoning | **Particle Filtering** | Maintains diverse reasoning paths — requires process reward model + GPU |
| Long multi-step tasks | **Entropic Particle Filtering** | Avoids premature convergence on long sequences — requires process reward model + GPU |

### Decision framework

1. **No GPU for a reward model?** → Self-Consistency (no reward model needed)
2. **Have a judge model or API?** → Best-of-N with LLM Judge
3. **Have a local GPU + PRM?** → Beam Search or Particle Filtering depending on task complexity
4. **Tool-calling task?** → Self-Consistency with `tool_vote="tool_hierarchical"` is the recommended starting point

## Budget Tuning

The `budget` parameter controls how many LLM calls are made per prompt:

| Algorithm | Budget meaning | Starting point | Diminishing returns |
|-----------|---------------|----------------|---------------------|
| Self-Consistency | Number of parallel generations | 5-8 | Beyond 16 for most tasks |
| Best-of-N | Number of candidates to score | 4-8 | Beyond 16 |
| Beam Search | Total generations (= beam_width × steps) | 16-32 | Depends on step count |
| Particle Filtering | Number of particles | 8-16 | Beyond 32 |

**Budget vs cost**: each budget unit = 1 LLM call. Budget 8 costs 8x a single call. Start low, increase only if quality improves.

**Budget vs latency**: Self-Consistency and Best-of-N run in parallel (latency ≈ single call). Beam Search and Particle Filtering are sequential per step (latency ≈ budget × step time).

## Reward Models

### Outcome Reward Models (ORM)

Score complete responses. Used by Best-of-N.

**LLM Judge** (easiest setup — uses an LLM to score):
```python
from its_hub import LLMJudge, OpenAICompatibleLanguageModel

judge_lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-key",
    model_name="gpt-4o-mini"
)
judge = LLMJudge(lm=judge_lm, fallback_score=5.0)
```

The judge model can be the same as the generation model, but using a stronger model as judge improves quality.

### Process Reward Models (PRM)

Score each reasoning step. Used by Beam Search and Particle Filtering. Requires a local GPU.

```python
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel

prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"  # or "mean", "min", "max"
)
```

**Aggregation methods:**
- `prod`: Product of step scores (strict — one bad step kills the score)
- `mean`: Average of step scores (forgiving)
- `min`: Worst step score (conservative)
- `max`: Best step score (optimistic)

Start with `prod` for math, `mean` for general reasoning.

## Tool-Calling Integration

Self-Consistency supports voting on tool calls, not just text:

```python
sc = SelfConsistency(tool_vote="tool_hierarchical")
result = sc.infer(lm, messages, budget=5, tools=tools, tool_choice="auto")
```

**Tool voting modes:**
- `tool_name`: Vote on which tool to call
- `tool_args`: Vote on tool arguments
- `tool_hierarchical` (recommended): First vote on tool name, then on arguments within the winning tool
- `exclude_args=["timestamp", "id"]`: Exclude non-semantic arguments from voting

Best-of-N also works with tool calls when using an LLM Judge that understands tool-call quality.

## Step Generation

For Beam Search and Particle Filtering, configure how the LLM generates incrementally:

```python
from its_hub import StepGeneration

sg = StepGeneration(
    max_steps=32,          # Maximum reasoning steps
    step_token="\n\n",     # Split on double newlines
    stop_token=r"\boxed",  # Stop when final answer found
)
```

**Tuning:**
- `max_steps`: Higher for complex problems. 16-32 is typical for math.
- `step_token`: Use `"\n\n"` for chain-of-thought, `"\n"` for more granular steps.
- `stop_token`: Match your expected answer format (`\boxed` for math, custom for other tasks).

## Concurrency Control

All algorithms accept an optional `orchestrator` for controlling parallelism:

```python
from its_hub import LMOrchestrator

orchestrator = LMOrchestrator(max_concurrency=4)
sc = SelfConsistency(orchestrator=orchestrator)
```

**When to tune:**
- **Rate-limited APIs**: Set `max_concurrency` to stay under the limit
- **Local vLLM**: Higher concurrency (16-32) is fine
- **Gateway integration**: Implement `AbstractOrchestrator` with your own rate limiting

## Interpreting Results

### Self-Consistency
- **Good sign**: Most responses agree (e.g., 6/8 voted for the same answer)
- **Bad sign**: No clear majority — problem may be ambiguous or model is uncertain. Try higher budget or a better model.

### Best-of-N
- **Good sign**: Top score is significantly higher than average
- **Bad sign**: All scores are similar — the judge can't differentiate. Try a stronger judge or different scoring criteria.

### Beam Search / Particle Filtering
- **Good sign**: Final beam scores are high and diverse
- **Bad sign**: All particles collapsed to the same path — try Entropic Particle Filtering for more diversity.

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All responses identical | Temperature too low or budget too low | Increase temperature (0.7-1.0) or budget |
| Self-Consistency ties | Budget too low for the task | Increase budget to odd number (5, 7, 9) |
| Best-of-N picks poor response | Judge model not strong enough | Use a stronger judge model or tune the prompt |
| Beam Search OOM | PRM too large for GPU | Use a smaller PRM or offload to different GPU (`device="cuda:1"`) |
| Particle Filtering slow | Sequential step generation | Reduce `max_steps` or switch to Self-Consistency for speed |
| Rate limit errors | Too many parallel calls | Set `LMOrchestrator(max_concurrency=N)` |
| Empty or null results | LM endpoint unreachable or API key invalid | Verify endpoint with a single `lm.agenerate_single()` call |

## Resource Cleanup

Always close the LM after use:

```python
# Async context manager (recommended)
async with OpenAICompatibleLanguageModel(...) as lm:
    result = await algorithm.ainfer(lm, prompt, budget=5)

# Sync usage — explicit close
lm = OpenAICompatibleLanguageModel(...)
result = algorithm.infer(lm, prompt, budget=5)
asyncio.run(lm.close())
```

## Performance Tips

1. **Start with Self-Consistency** — cheapest, fastest, no reward model needed
2. **Upgrade to Best-of-N** when you have a judge — better quality, same latency
3. **Use Beam Search** for step-by-step math/reasoning — highest quality on those tasks
4. **Try Entropic Particle Filtering** if standard PF converges too early
5. **Monitor GPU memory** when using local reward models — PRMs are 7B+ parameters
6. **Benchmark** with `scripts/benchmark.py` on MATH500 or AIME-2024 to compare algorithms for your model

## Reference Documentation

Detailed documentation for specific topics lives in the `docs/` directory:

- `docs/algorithms.md` — Full code examples for every algorithm (Self-Consistency, Best-of-N, Beam Search, Particle Filtering, Entropic PF), tool-calling integration, step generation config, and reward model setup
- `docs/orchestration.md` — Concurrency control, custom orchestrator implementation for gateway deployments, async/sync usage patterns
- `docs/benchmarking.md` — How to benchmark algorithms on MATH500 and AIME-2024, budget scaling analysis
- `docs/iaas-service.md` — Running the Inference-as-a-Service HTTP server
- `docs/quick-start.md` — Getting started from zero
