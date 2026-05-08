---
description: "Guided first-run configuration for inference-time scaling"
argument-hint: ""
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh:*)"]
---

# its-hub Setup

You are helping the user configure inference-time scaling for their LLM workflows.

## Step 1: Detect Environment

Run the detection script to understand the current state:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

## Step 2: Install if Needed

If `library=missing`:
- Ask the user: "its_hub isn't installed. I can install it for you — want me to proceed?"
- If yes and `installer=uv`: run `uv pip install its_hub`
- If yes and `installer=pip`: run `pip install its_hub`
- If `installer=none`: tell the user they need Python and pip/uv installed first

## Step 3: Collect Configuration

Ask these questions **one at a time**:

1. **Provider**: "Which LLM provider are you using?" — `openai` (default, works with vLLM) or `litellm` (multi-provider: Bedrock, Vertex, etc.)
2. **Endpoint**: "What's your model endpoint URL?" — e.g., `http://localhost:8000/v1` for vLLM, `https://api.openai.com/v1` for OpenAI
3. **API key**: "What's your API key?" (required for openai provider, optional for litellm)
4. **Model name**: "What's the model identifier?" — e.g., `gpt-4o`, `Qwen/Qwen2.5-32B-Instruct`
5. **Extra args** (only if `litellm`): "Do you need provider-specific arguments? (e.g., AWS credentials for Bedrock: `aws_access_key_id`, `aws_secret_access_key`, `aws_region_name`)" — If yes, collect as key-value pairs and store in `extra_args`.
6. **Algorithm**: "Which scaling algorithm do you want to use?"
   - **Self-consistency** — Generates N responses, votes on the most common answer. Best for: getting the agreed-upon answer. No extra setup needed.
   - **Best-of-N** — Generates N responses, scores each with a reward model. Best for: highest quality response. Requires a reward model.
   - **Particle filtering** — Step-by-step reasoning with pruning. Best for: careful reasoning tasks. Requires a process reward model + step tokens.

## Step 4: Algorithm-Specific Config

Based on the algorithm choice:

**Self-consistency:**
- Ask: "Do you need regex patterns for answer extraction? (e.g., `\\boxed{...}` for math problems). If unsure, skip — default exact-match voting works for most cases."
- If yes: collect the regex pattern(s)
- Ask: "Will your prompts involve tool/function calls?" If yes, ask which voting strategy: `tool_name`, `tool_args`, or `tool_hierarchical`

**Best-of-N:**
- Ask: "How should responses be scored?"
  - `llm-judge` — Uses another LLM to judge quality (no GPU needed)
  - Local reward model — Requires a vLLM-served reward model
- If `llm-judge`: collect judge model name, judge endpoint (or `auto`), judge API key
- If local: collect reward model name, device (e.g., `cuda:0`)

**Particle filtering:**
- Collect: step token (e.g., `"\n\n"`), stop token (optional)
- Collect: process reward model name, device, aggregation method

## Step 5: Save Config

Write the config to `.its-hub/config.json`. Use this structure:

```json
{
  "provider": "<provider>",
  "models": {
    "default": {
      "endpoint": "<endpoint>",
      "api_key": "<api_key>",
      "model": "<model_name>"
    }
  },
  "algorithm": "<algorithm>",
  "budget": 8,
  "iaas_port": 8108,
  "algorithm_config": { ... }
}
```

Add `.its-hub/` to `.gitignore` if not already present.

## Step 6: Start Server (Optional)

Ask: "Want me to start the IaaS server now?"

If yes, run:
```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh" start
```

Report success and remind the user they can now use `/its-scale` to run scaling.

## Adding More Models

If the user runs `/its-setup` again and a config already exists, ask: "You already have a configuration. Do you want to update it or add another model?"

If adding a model: collect endpoint, API key, and model name. Add a new entry to the `models` dict using the model name as key. Don't overwrite existing config.
