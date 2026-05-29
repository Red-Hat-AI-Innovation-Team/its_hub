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

Also check for API keys in the environment:
```!
echo "openai_key=${OPENAI_API_KEY:+found}" "anthropic_key=${ANTHROPIC_API_KEY:+found}" "endpoint=${OPENAI_BASE_URL:-none}"
```

## Step 2: Install if Needed

If `library=missing`:
- Explain: "its_hub is a library for inference-time scaling — it generates multiple LLM responses and selects the best one using voting or scoring algorithms."
- Ask permission: "I can install it for you. Want me to proceed?"
- If yes and `installer=uv`: run `uv pip install "its_hub[lm]"`
- If yes and `installer=pip`: run `pip install "its_hub[lm]"`
- If `installer=none`: tell the user they need Python 3.11+ and pip/uv installed first

## Step 3: Quick Setup or Custom

**If an API key was detected**, offer a one-question fast path:

> "I detected your OpenAI API key. I can set up with these defaults:
> - Model: `gpt-4o-mini`
> - Algorithm: `self-consistency`
> - Budget: `8`
>
> Accept these defaults, or would you like to customize?"

If the user accepts, skip to Step 5 using:
- `endpoint`: `https://api.openai.com/v1` (or `OPENAI_BASE_URL` if set)
- `model`: `gpt-4o-mini`
- `algorithm`: `self-consistency`
- `budget`: `8`

**If no API key was detected**, or the user wants to customize, proceed to Step 4.

## Step 4: Collect Configuration

Ask these questions **one at a time**:

1. **Endpoint**: "What's your model endpoint URL?" — e.g., `http://localhost:8000/v1` for vLLM, `https://api.openai.com/v1` for OpenAI
2. **Model name**: "What's the model identifier?" — e.g., `gpt-4o`, `Qwen/Qwen2.5-32B-Instruct`
3. **Algorithm**: "Which scaling algorithm do you want to use?" — consult the `inference-scaling-guide` skill for guidance if the user is unsure.
   - **Self-consistency** — Votes on the most common answer. No extra setup needed.
   - **Best-of-N** — Scores each with an LLM judge. Requires a judge model.
4. **Budget**: "Which budget do you want to use?" — consult the `inference-scaling-guide` skill for guidance if the user is unsure.

### Algorithm-Specific Config

**Self-consistency:**
- Ask: "Do you need regex patterns for answer extraction? (e.g., `\boxed{...}` for math). If unsure, skip — default voting works for most cases."
- Ask: "Will your prompts involve tool/function calls?" If yes, ask which voting strategy: `tool_name`, `tool_args`, or `tool_hierarchical`

**Best-of-N:**
- Ask: "Which model should be the judge? (This can be the same model or a different one)"
- Collect: judge model name, judge endpoint (default: same as generation model)

## Step 5: Ensure API Key

API keys are read from environment variables — **never store them in the config file**.

If no API key was detected in Step 1, tell the user to set the appropriate environment variable:

> "Set your API key as an environment variable before running scaling:
> ```bash
> export OPENAI_API_KEY="sk-..."        # OpenAI or OpenAI-compatible endpoints
> export ANTHROPIC_API_KEY="sk-ant-..."  # Anthropic endpoints
> ```
> For local vLLM endpoints that don't require authentication, no API key is needed."

## Step 6: Save Config

Write the config to `.its-hub/config.json`:

```json
{
  "models": {
    "default": {
      "endpoint": "<endpoint>",
      "model": "<model_name>"
    }
  },
  "algorithm": "<algorithm>",
  "budget": "<budget>",
  "algorithm_config": {}
}
```

Add `.its-hub/` to `.gitignore` if not already present.

Confirm the config file was written, then report success:

> "Setup complete! To run scaling, use the `inference-scaling` skill.
>
> **API keys** are read from environment variables, not the config file. Make sure the appropriate variable is set in your shell:
> ```bash
> export OPENAI_API_KEY="sk-..."        # OpenAI or OpenAI-compatible endpoints (including vLLM with auth)
> export ANTHROPIC_API_KEY="sk-ant-..."  # Anthropic endpoints
> ```
> Local endpoints (e.g., vLLM without auth) don't need an API key."

## Adding More Models

If this skill is invoked again and a config already exists, ask: "You already have a configuration. Do you want to update it or add another model?"

If adding a model: collect endpoint and model name. Add a new entry to the `models` dict using the model name as key.
