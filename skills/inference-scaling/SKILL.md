---
name: inference-scaling
description: "Use when the user wants to improve LLM response quality by generating multiple candidates and selecting the best one. Applies to tasks like: scaling a prompt, running self-consistency, best-of-n selection, or comparing multiple LLM outputs."
---

# Inference-Time Scaling

Help the user apply inference-time scaling to get better LLM responses.

## Detection

First, check the environment:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

## Routing

Based on detection results:

### Nothing available (`library=missing`, `config=missing`)
Invoke the `setup-guide` skill to walk through installation and configuration.

### Config missing but library installed (`library=installed`, `config=missing`)
Ask the user to run `/its-setup` to configure, or invoke the `setup-guide` skill.

### Server running (`server=running`, `config=found`)
This is the preferred path. Use the IaaS API:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh" --metadata "<user's prompt>"
```

### No server, library installed (`server=stopped`, `library=installed`, `config=found`)
Ask the user: "The IaaS server isn't running. I can:
1. Start it for you (recommended)
2. Run scaling directly via Python (limited to self-consistency and best-of-n with LLM judge)"

If start server:
```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh" start
```
Then proceed with IaaS path.

## Algorithm Selection

If the user hasn't specified an algorithm, use the one from their config. If they mention preferences, guide them:

| User says | Algorithm | Why |
|---|---|---|
| "vote", "consensus", "most common" | self-consistency | Finds the majority answer |
| "best", "highest quality", "score", "rank" | best-of-n | Ranks by quality |
| "step by step", "reasoning", "careful" | particle-filtering | Step-by-step with pruning |

## Batch Detection

If the user provides a file path (e.g., "scale all prompts in data/eval.jsonl"), route to `/its-scale-batch` instead.

## Presenting Results

Parse the JSON response from the scaling script:

1. **Show the selected response** prominently
2. **Show metadata** if available:
   - Self-consistency: "Selected by majority vote (5/8 responses agreed)"
   - Best-of-N: "Selected as highest scoring (score: 0.92 out of 8 candidates)"
   - Particle filtering: "Selected after 7 reasoning steps (log weight: -0.3)"
3. Keep it concise — the user wants the answer, not a wall of JSON
