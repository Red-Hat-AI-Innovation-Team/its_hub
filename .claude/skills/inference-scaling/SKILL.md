---
name: inference-scaling
description: "Use when the user wants to run inference-time scaling on a prompt — detect environment, execute scaling, and present results. For algorithm selection, budget tuning, reward models, and troubleshooting, consult the inference-scaling-guide skill."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# Run Inference-Time Scaling

Execute scaling on a prompt. For algorithm guidance, budget tuning, and troubleshooting, consult the `inference-scaling-guide` skill.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

### If not ready

- `library=missing` or `config=missing`: invoke the `setup-guide` skill.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Execute Scaling

Run the scaling script with the user's prompt and any overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh" --metadata $ARGUMENTS
```

If the user provides a file path (e.g., "scale all prompts in data/eval.jsonl"), invoke the `batch-scaling` skill instead.

## Step 3: Present Results

1. **Selected response** — Show the winning response prominently
2. **Metadata** (if available):
   - Self-consistency: show vote counts ("Selected by majority vote — 5/8 responses agreed")
   - Best-of-N: show scores ("Selected as highest scoring — score: 0.92 out of 8 candidates")
3. **Configuration used** — algorithm, budget, model (briefly)

If the scaling failed, consult the `inference-scaling-guide` skill for troubleshooting.
