---
name: setup-guide
description: "Use when the user wants to set up inference-time scaling for the first time, or when its_hub is not yet installed/configured in the current environment."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# its_hub Setup Guide

You are helping the user set up inference-time scaling.

## Step 1: Detect Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Explain: "its_hub is a library for inference-time scaling — it generates multiple LLM responses and selects the best one using voting or scoring algorithms."
- Ask permission: "I can install it for you. Want me to proceed?"
- If yes and `installer=uv`: run `uv pip install "its_hub[lm]"`
- If yes and `installer=pip`: run `pip install "its_hub[lm]"`
- If `installer=none`: tell the user they need Python 3.11+ and pip/uv installed first

## Step 3: Collect Configuration

Ask these questions **one at a time**:

1. **Endpoint**: "What's your model endpoint URL?" — e.g., `http://localhost:8000/v1` for vLLM, `https://api.openai.com/v1` for OpenAI
2. **API key**: "What's your API key?" (may be optional for local vLLM)
3. **Model name**: "What's the model identifier?" — e.g., `gpt-4o`, `Qwen/Qwen2.5-32B-Instruct`
4. **Algorithm**: "Which scaling algorithm do you want to use?" — consult the `inference-scaling-guide` skill for detailed algorithm selection guidance if the user is unsure.
   - **Self-consistency** — Votes on the most common answer. No extra setup needed.
   - **Best-of-N** — Scores each with an LLM judge. Requires a judge model.

## Step 4: Algorithm-Specific Config

Based on the algorithm choice:

**Self-consistency:**
- Ask: "Do you need regex patterns for answer extraction? (e.g., `\boxed{...}` for math problems). If unsure, skip — default exact-match voting works for most cases."
- If yes: collect the regex pattern(s)
- Ask: "Will your prompts involve tool/function calls?" If yes, ask which voting strategy: `tool_name`, `tool_args`, or `tool_hierarchical`

**Best-of-N:**
- Ask: "Which model should be the judge? (This can be the same model or a different one)"
- Collect: judge model name, judge endpoint (default: same as generation endpoint), judge API key (default: same as generation key)

## Step 5: Save Config

Write the config to `.its-hub/config.json`. Use this structure:

```json
{
  "models": {
    "default": {
      "endpoint": "<endpoint>",
      "api_key": "<api_key>",
      "model": "<model_name>"
    }
  },
  "algorithm": "<algorithm>",
  "budget": 8,
  "algorithm_config": { ... }
}
```

Add `.its-hub/` to `.gitignore` if not already present.

## Step 6: Verify Config

Confirm the config file was written by checking that `.its-hub/config.json` exists and contains valid JSON. If the file is missing or invalid, re-run Step 5.

Report success and remind the user they can now use the `inference-scaling` skill to run scaling.

## Adding More Models

If this skill is invoked again and a config already exists, ask: "You already have a configuration. Do you want to update it or add another model?"

If adding a model: collect endpoint, API key, and model name. Add a new entry to the `models` dict using the model name as key. Don't overwrite existing config.
