---
name: inference-scaling
description: "Use when the user wants to improve LLM response quality by generating multiple candidates and selecting the best one. Applies to tasks like: scaling a prompt, running self-consistency, best-of-n selection, or comparing multiple LLM outputs."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# Inference-Time Scaling

Help the user apply inference-time scaling to get better LLM responses.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

### If not ready

- `library=missing` and `config=missing`: invoke the `setup-guide` skill.
- `library=installed` and `config=missing`: tell the user to run the `setup-guide` skill to configure.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Execute Scaling

Run the scaling script with the user's prompt and any overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh" --metadata $ARGUMENTS
```

### Algorithm Selection

If the user hasn't specified an algorithm, use the one from their config. If they mention preferences, guide them:

| User says | Algorithm | Why |
|---|---|---|
| "vote", "consensus", "most common" | self-consistency | Finds the majority answer |
| "best", "highest quality", "score", "rank" | best-of-n | Ranks by quality via LLM judge |

### Batch Detection

If the user provides a file path (e.g., "scale all prompts in data/eval.jsonl"), invoke the `batch-scaling` skill instead.

## Step 3: Present Results

Parse the JSON response and present it clearly:

1. **Selected response** — Show the winning response prominently
2. **Metadata** (if available):
   - Self-consistency: show vote counts ("Selected by majority vote — 5/8 responses agreed")
   - Best-of-N: show scores ("Selected as highest scoring — score: 0.92 out of 8 candidates")
3. **Configuration used** — algorithm, budget, model (briefly)

If the scaling failed, show the error and suggest troubleshooting steps.
