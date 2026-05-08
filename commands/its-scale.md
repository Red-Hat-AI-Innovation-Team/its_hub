---
description: "Run inference-time scaling on a prompt"
argument-hint: "<prompt> [--budget N] [--algorithm ALG] [--model KEY]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh:*)"]
---

# its-hub Scale

Run inference-time scaling on a prompt to get higher quality LLM responses.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

If `config=missing`, tell the user to run `/its-setup` first.

If `server=stopped` and `library=installed`, ask: "The IaaS server isn't running. Want me to start it, or run directly via Python?"

## Step 2: Execute Scaling

Run the scaling script with the user's prompt and any overrides:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh" --metadata $ARGUMENTS
```

## Step 3: Present Results

Parse the JSON response and present it clearly:

1. **Selected response** — Show the winning response prominently
2. **Metadata** (if available):
   - Self-consistency: show vote counts and which responses matched
   - Best-of-N: show scores for each candidate
   - Particle filtering: show log weights and steps used
3. **Configuration used** — algorithm, budget, model (briefly)

If the scaling failed, show the error and suggest troubleshooting steps.
