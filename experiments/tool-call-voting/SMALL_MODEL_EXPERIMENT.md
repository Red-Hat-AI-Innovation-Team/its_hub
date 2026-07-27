# Small Model Experiment: SC/BoN for Tool Calling

## Hypothesis

Our negative result (correlated errors, zero oracle headroom) was established
on capable models (gpt-4o-mini, gpt-3.5-turbo, claude-sonnet-4) where
single-shot accuracy is 89-94%. Small open-source models (1B-8B) score
significantly lower on BFCL (10-50% range), which means:

1. More tasks will be wrong — more room for improvement
2. Errors may be less correlated — different samples may get different
   tasks right, creating the oracle headroom SC/BoN needs
3. The ToolRM paper showed the biggest gains on small models (Qwen 0.6B
   gained +24.9pp with ToolRM re-ranking)

If small models show oracle headroom where large models don't, that
validates SC/BoN as a technique for making small models more reliable
at tool calling — a practical win for edge/cost-constrained deployments.

## Models to Test (pick 2-3)

Priority order based on BFCL scores and tool-calling support:

| Model | Params | BFCL Score | Why |
|-------|-------:|:---:|-----|
| **Qwen3-8B** | 8B | ~50-60% | Strong for size, native FC support |
| **Llama 3.1 8B Instruct** | 8B | ~45-55% | Well-supported, native FC |
| **Qwen3-4B** | 4B | ~35-45% | Tests if voting helps at lower capability |
| **SmolLM3-3B** | 3B | ~25-35% | Native tool calling in compact package |
| **Llama 3.2 3B Instruct** | 3B | ~22% | Known weak, tests maximum headroom |
| **Llama 3.2 1B Instruct** | 1B | ~11% | Floor — does voting help at all? |

Recommended pair: **Qwen3-8B** (strongest small model) + **Llama 3.2 3B**
(weak model, most headroom potential).

## Experiment Protocol

### Phase 1: Oracle Headroom Check (the critical question)

Before running any voting or BoN, check whether oracle headroom exists:

```bash
# Set these to your vLLM endpoint
export ITS_ENDPOINT=http://your-node:8000/v1
export ITS_API_KEY=your-key
export FORECAST_MODEL=openai/qwen3-8b  # litellm format for vLLM

# Run 50 tasks, N=5, temp=1.0
BFCL_FILE=BFCL_v4_multiple.json \
BUDGET=5 \
MAX_TASKS=50 \
MAX_CONCURRENT=2 \
TEMPERATURE=1.0 \
uv run python experiments/tool-call-voting/sampling_harness.py
```

Then check oracle headroom:

```bash
uv run python experiments/tool-call-voting/analyze_results.py
```

**Decision gate:** If oracle headroom > 3pp (single-shot wrong but at least
one of N samples correct), proceed to Phase 2. If oracle headroom ≈ 0 (same
as with large models), the correlated-error finding extends to small models
and we're done.

### Phase 2: SC and BoN Comparison (only if Phase 1 positive)

Run the full comparison:

```bash
# Majority voting (existing harness)
BUDGET=10 MAX_TASKS=0 TEMPERATURE=1.0 EQUIVALENCE=true \
uv run python experiments/tool-call-voting/sampling_harness.py

# Schema validation BoN (existing harness)
uv run python experiments/tool-call-voting/bon_harness.py --fuzzy

# LLM self-judge BoN (existing harness)
MAX_TASKS=50 N_SAMPLES=10 \
uv run python experiments/tool-call-voting/llmjudge_bon_harness.py

# Threshold sweep
uv run python experiments/tool-call-voting/threshold_sweep.py
```

### Phase 3: CoT-SC (if Phase 2 shows promise)

```bash
COT=true BUDGET=10 MAX_TASKS=0 TEMPERATURE=1.0 \
uv run python experiments/tool-call-voting/sampling_harness.py
```

## vLLM Setup Notes

For models with native function-calling support:

```bash
# Qwen3-8B
vllm serve Qwen/Qwen3-8B --tool-call-parser hermes

# Llama 3.1 8B
vllm serve meta-llama/Llama-3.1-8B-Instruct --tool-call-parser llama3_json

# Llama 3.2 3B
vllm serve meta-llama/Llama-3.2-3B-Instruct --tool-call-parser llama3_json
```

The harness uses litellm, which connects to vLLM via the OpenAI-compatible
endpoint. Set `FORECAST_MODEL=openai/<model-name>` and
`ITS_ENDPOINT=http://<host>:<port>/v1`.

## What Success Looks Like

Per the pre-registered criterion from the research doc:

> Voting is worth continuing if high-confidence accuracy exceeds single-shot
> by 3-5pp while covering at least 50% of cases.

For small models, we'd also accept:
- Oracle headroom > 5pp (proves the mechanism works even if voting
  doesn't fully capture it — motivates ToolRM investment)
- Any configuration where voted accuracy exceeds single-shot by > 3pp

## What Failure Looks Like

- Oracle headroom ≈ 0 on small models too — correlated errors are a
  universal property of tool calling, not a capability-dependent one
- This would be the strongest possible negative result, closing the
  entire SC/BoN direction for tool calls definitively
