# its_hub Claude Code Plugin — Design Spec

## Overview

A plugin that brings inference-time scaling capabilities to coding agents (Claude Code, Cursor). Users can generate multiple LLM candidate responses and select the best one using voting, scoring, or search algorithms — directly from their coding workflow.

## Goals

- **Primary use case:** Developer workflow — a coding agent user applies inference-time scaling to their own LLM calls without leaving their editor.
- **IaaS-first:** Prefer the IaaS HTTP API when a server is running; fall back to direct Python library calls.
- **Curated defaults:** Expose self-consistency, best-of-n, and particle-filtering with sensible defaults. Advanced configuration is available but not required.
- **Guided setup:** First-time experience walks the user through installation and configuration with permission before modifying the environment.
- **Multi-agent compatible:** Claude Code and Cursor from day one; Gemini CLI, Codex, OpenCode later.

## Plugin Structure

```
its_hub/                              # repo root
├── .claude-plugin/
│   └── plugin.json                   # Claude Code manifest
├── .cursor-plugin/
│   └── plugin.json                   # Cursor manifest
├── commands/
│   ├── its-setup.md                  # guided first-run configuration
│   ├── its-scale.md                  # run inference-time scaling on a prompt
│   ├── its-scale-batch.md            # batch scaling from a file
│   └── its-server.md                 # start/stop/status for IaaS server
├── skills/
│   ├── inference-scaling/
│   │   └── SKILL.md                  # contextual: detects scaling intent
│   └── setup-guide/
│       └── SKILL.md                  # first-time setup walkthrough
├── scripts/
│   ├── its_detect.sh                 # detect server/library availability
│   ├── its_server.sh                 # manage IaaS process lifecycle
│   └── its_scale.sh                  # execute scaling (IaaS API or direct Python)
├── its_hub/                          # existing Python library (unchanged)
├── pyproject.toml                    # existing
└── ...
```

## Manifests

Both `.claude-plugin/plugin.json` and `.cursor-plugin/plugin.json` share the same content:

```json
{
  "name": "its-hub",
  "description": "Inference-time scaling for LLMs — generate multiple candidates and select the best using voting, scoring, or search",
  "author": {
    "name": "Red Hat AI Innovation Team"
  }
}
```

## Prerequisites

Before the plugin ships, the IaaS `regex_patterns` Pydantic validator for self-consistency must be relaxed. Currently (`iaas.py` line 121-128), the validator rejects `regex_patterns=None` for self-consistency, but the handler (`iaas.py` line 322-328) already gracefully falls back to the default exact-match projection function when `regex_patterns` is `None`. Relaxing the validator to match the handler allows users to skip regex pattern configuration for the common case where exact-match voting is sufficient. Until this is done, the setup flow must always collect at least one regex pattern for self-consistency on the IaaS path.

## Commands

### `/its-setup` — First-run configuration

1. Run `its_detect.sh` — check for running IaaS server and installed library.
2. If `its_hub` not installed: ask user permission, then install into existing venv or create one (`uv pip install its_hub` or `pip install its_hub`).
3. Ask for LLM provider (`openai` or `litellm`).
4. Ask for model endpoint (e.g., `http://localhost:8000/v1` for vLLM, or an OpenAI-compatible URL).
5. Ask for API key (required for openai provider; optional for litellm).
6. Ask for model name.
7. Ask for preferred algorithm (self-consistency / best-of-n / particle-filtering) with brief explanations.
8. **Algorithm-specific configuration** (branched by algorithm choice):
   - **Self-consistency:** Ask whether the user needs regex patterns for answer extraction (e.g., `\\boxed{...}` for math). At least one pattern is required for the IaaS path (see Prerequisites). Optionally ask about tool voting strategy (`tool_name`, `tool_args`, `tool_hierarchical`) if the user's use case involves tool calls.
   - **Best-of-N:** Ask for reward model source — either a local model name (requires `its_hub[vllm]`) or `llm-judge` (uses an LLM as judge). If `llm-judge`: collect judge model name, judge endpoint (or `auto`), judge API key, and optionally judge criterion (defaults to `overall_quality`).
   - **Particle filtering:** Ask for step token (e.g., `"\n\n"`), stop token, and reward model name (requires `its_hub[vllm]`). Optionally ask about reward model device and aggregation method.
9. Persist config to `.its-hub/config.json` in the project directory.
10. Add `.its-hub/` to `.gitignore` if not already present (config may contain API keys).
11. Optionally start the IaaS server.

### `/its-scale` — Single prompt scaling

Usage: `/its-scale <prompt>`

1. Check for config (run setup if missing).
2. Detect IaaS server → use API; else fall back to direct Python.
3. Send the prompt with configured algorithm and budget.
4. Return the selected response, optionally showing all candidates and scores.
5. User can override algorithm, budget, or model per-call.

### `/its-scale-batch` — Batch scaling from file

Usage: `/its-scale-batch path/to/prompts.jsonl [--output path/to/results.jsonl]`

Supported input formats:
- **JSONL** — one JSON object per line, expects a `prompt` or `messages` field.
- **CSV** — expects a `prompt` column.
- **TXT** — one prompt per line.

Flow:
1. Detect server/library (same as `/its-scale`).
2. Read and validate input file, report count.
3. Process prompts — parallel calls to IaaS (default concurrency: 8, configurable via `--concurrency N`) if server is running, sequential via Python if fallback.
4. Write results to output file (defaults to `<input_name>_scaled.jsonl`).
5. Report summary with success/failure counts.

Output format (JSONL):
```json
{
  "prompt": "original prompt text",
  "selected_response": "the winning response",
  "algorithm": "self-consistency",
  "budget": 8,
  "metadata": {"response_counts": {"answer_a": 5, "answer_b": 3}, "selected_index": 2}
}
```

Error handling: failed prompts are logged in the output row and reported at the end; processing continues.

### `/its-server` — Server lifecycle

Usage: `/its-server start|stop|status`

- `start` — Launch `its-iaas` in background, validate config completeness for the chosen algorithm before calling `/configure`, write PID to `.its-hub/server.pid`. If config is incomplete, report which fields are missing and suggest running `/its-setup`.
- `stop` — Read PID file and kill the process. If no server is running (no PID file or stale PID), print a warning and exit cleanly.
- `status` — Check PID and hit `/v1/models` to confirm alive.

## Skills

### `inference-scaling`

Trigger: *"Use when the user wants to improve LLM response quality by generating multiple candidates and selecting the best one. Applies to tasks like: scaling a prompt, running self-consistency, best-of-n selection, or comparing multiple LLM outputs."*

Behavior:
- Runs `its_detect.sh` to check availability.
- If IaaS running: construct `curl` call to `/v1/chat/completions` with prompt, algorithm, budget, and `return_response_only: false` (so the agent can display metadata like scores and vote counts).
- If no server but library installed: offer to start server or construct Python snippet using `its_hub` directly.
- If nothing available: invoke `setup-guide` skill.
- Includes algorithm decision guide: voting → self-consistency, scoring/ranking → best-of-n, step-by-step search → particle-filtering.
- When user provides a file path, routes to `/its-scale-batch`.
- Presents results clearly: selected response, and optionally the full candidate set with scores/votes.

### `setup-guide`

Trigger: *"Use when the user wants to set up inference-time scaling for the first time, or when its_hub is not yet installed/configured in the current environment."*

Behavior:
- Walks through environment detection.
- Asks permission, then installs `its_hub`.
- Collects endpoint, API key, model, algorithm preferences.
- Writes `.its-hub/config.json`.
- Offers to start IaaS server.
- Hands off to `inference-scaling` skill once setup is complete.

## Scripts

### `its_detect.sh`

Checks (in order):
1. Is an IaaS server responding at the configured URL (default `localhost:8108`)? → `server=running|stopped`
2. Is `its_hub` importable in the current Python environment? → `library=installed|missing`
3. Is `uv` or `pip` available? → `installer=uv|pip|none`
4. Is there an existing config at `.its-hub/config.json`? → `config=found|missing`

Outputs key-value pairs for skills/commands to parse.

### `its_server.sh`

- `start` — Reads `.its-hub/config.json`, starts `its-iaas` in background with `--port` from `iaas_port` (must be passed explicitly — the `its-iaas` binary defaults to port 8000, but the plugin convention is 8108). Calls `/configure` with saved settings, writes PID.
- `stop` — Reads PID file, kills process, cleans up.
- `status` — Checks PID and hits `/v1/models`.

### `its_scale.sh`

- Reads config, constructs `curl` call to IaaS `/v1/chat/completions`.
- Accepts arguments: prompt (or stdin), algorithm override, budget override, model override.
- Falls back to a Python script if no server available. The fallback is limited to **self-consistency** and **best-of-n with `llm-judge`** (neither requires a local reward model). For best-of-n with a local reward model and particle-filtering, the fallback requires a running vLLM reward model, so the script will prompt the user to start the IaaS server instead.
- Outputs selected response and optionally full metadata.

## Configuration

Stored at `.its-hub/config.json` in the project directory. Auto-generated by `/its-setup`; user never needs to edit manually. `/its-setup` adds `.its-hub/` to `.gitignore` since the config may contain API keys. Environment variables can be used in place of literal API keys (e.g., `"$ITS_API_KEY"`) — scripts expand these at runtime.

The config schema varies by algorithm. Examples for each:

**Self-consistency:**
```json
{
  "provider": "openai",
  "models": {
    "default": {
      "endpoint": "http://localhost:8000/v1",
      "api_key": "...",
      "model": "qwen-32b"
    }
  },
  "algorithm": "self-consistency",
  "budget": 8,
  "iaas_port": 8108,
  "algorithm_config": {
    "regex_patterns": ["\\\\boxed\\{([^}]+)\\}"],
    "tool_vote": null,
    "exclude_tool_args": []
  }
}
```

**Best-of-N (with LLM judge):**
```json
{
  "provider": "litellm",
  "models": {
    "default": {
      "endpoint": "http://localhost:8000/v1",
      "api_key": "...",
      "model": "qwen-32b"
    }
  },
  "algorithm": "best-of-n",
  "budget": 8,
  "iaas_port": 8108,
  "algorithm_config": {
    "rm_name": "llm-judge",
    "judge_model": "gpt-4o",
    "judge_base_url": "https://api.openai.com/v1",
    "judge_api_key": "...",
    "judge_criterion": "overall_quality",
    "judge_mode": "groupwise"
  }
}
```

**Particle filtering:**
```json
{
  "provider": "openai",
  "models": {
    "default": {
      "endpoint": "http://localhost:8000/v1",
      "api_key": "...",
      "model": "qwen-32b"
    }
  },
  "algorithm": "particle-filtering",
  "budget": 8,
  "iaas_port": 8108,
  "algorithm_config": {
    "rm_name": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
    "rm_device": "cuda:0",
    "rm_agg_method": "model",
    "step_token": "\n\n",
    "stop_token": null
  }
}
```

LiteLLM users may need provider-specific arguments (e.g., AWS credentials for Bedrock). These are stored in an optional `extra_args` field:

```json
{
  "provider": "litellm",
  "models": { ... },
  "algorithm": "best-of-n",
  "budget": 8,
  "iaas_port": 8108,
  "extra_args": {
    "aws_access_key_id": "...",
    "aws_secret_access_key": "...",
    "aws_region_name": "us-east-1"
  },
  "algorithm_config": { ... }
}
```

### Config to IaaS `/configure` field mapping

The plugin config uses a structured layout; the IaaS `/configure` endpoint uses a flat schema. Scripts translate between them:

| Plugin config path | IaaS `/configure` field |
|---|---|
| `provider` | `provider` |
| `models.<key>.endpoint` | `endpoint` |
| `models.<key>.api_key` | `api_key` |
| `models.<key>.model` | `model` |
| `algorithm` | `alg` |
| `extra_args` | `extra_args` |
| `algorithm_config.regex_patterns` | `regex_patterns` |
| `algorithm_config.tool_vote` | `tool_vote` |
| `algorithm_config.exclude_tool_args` | `exclude_tool_args` |
| `algorithm_config.rm_name` | `rm_name` |
| `algorithm_config.rm_device` | `rm_device` |
| `algorithm_config.rm_agg_method` | `rm_agg_method` |
| `algorithm_config.step_token` | `step_token` |
| `algorithm_config.stop_token` | `stop_token` |
| `algorithm_config.judge_model` | `judge_model` |
| `algorithm_config.judge_base_url` | `judge_base_url` |
| `algorithm_config.judge_api_key` | `judge_api_key` |
| `algorithm_config.judge_criterion` | `judge_criterion` |
| `algorithm_config.judge_mode` | `judge_mode` |

The `iaas_port` field is used only by `its_server.sh` (passed as `--port` to the `its-iaas` binary) and is not sent to `/configure`.

Multi-model support: users can add models by running `/its-setup` again or by saying "add model X". The agent collects endpoint, API key, and model name, then adds a new entry to the `models` dict with the model name as key. Per-request model override is supported.

Model selection priority:
1. Per-request override (user specifies in the prompt) — resolved against the `models` dict by key.
2. The `"default"` entry in the `models` dict.

If a requested model isn't configured on the IaaS server, the skill asks whether to add it (endpoint + API key), calls `/configure`, then proceeds.

## User Flows

### Flow 1: First-time user

1. User installs plugin (points coding agent at this repo).
2. User says "I want to scale my LLM responses."
3. `inference-scaling` skill fires → `its_detect.sh` → nothing found.
4. Hands off to `setup-guide` skill.
5. Agent asks permission to install, user agrees.
6. Agent installs `its_hub`, collects config, writes `.its-hub/config.json`, starts IaaS.
7. Agent runs the original scaling request and returns the result.

### Flow 2: Returning user (server running)

1. User says "run self-consistency on this prompt with budget 16."
2. `inference-scaling` skill fires → server running, config found.
3. Agent constructs `curl` to IaaS, returns selected response.

### Flow 3: Returning user (no server, library installed)

1. User says "scale this with best-of-n."
2. `inference-scaling` skill fires → no server, library found.
3. Agent asks: start server or run directly via Python?
4. Proceeds accordingly.

### Flow 4: Batch processing

1. User says "scale all prompts in data/eval.jsonl."
2. `inference-scaling` skill detects file path, routes to `/its-scale-batch`.
3. Reads file, processes in parallel against IaaS, writes results.
4. Reports summary.

## Algorithms

### Scope

The plugin exposes the three algorithms supported by the IaaS server: **self-consistency**, **best-of-n**, and **particle-filtering**. Beam search, entropic particle filtering, and other library-only algorithms are not exposed through the plugin — they can be used via the Python library directly but are not part of the plugin's curated surface.

### User-facing descriptions

| Algorithm | When to use | What it does | Requires |
|---|---|---|---|
| Self-consistency | You want the most common/agreed-upon answer | Generates N responses, votes on the most frequent answer | Nothing extra (optionally regex patterns for answer extraction) |
| Best-of-N | You want the highest-quality response | Generates N responses, scores each with a reward model, picks the best | Reward model (local vLLM model or LLM-as-judge) |
| Particle filtering | You want careful step-by-step reasoning | Explores reasoning paths step by step, pruning weak paths as it goes | Process reward model (local vLLM) + step token config |

## Future Work

- Support for additional coding agents (Gemini CLI, Codex, OpenCode) — tracked in a separate issue.
- Particle filtering may warrant its own skill if configuration complexity grows.
- Streaming support for long-running batch jobs.
- Relaxing the IaaS validator for `regex_patterns` in self-consistency — the handler already supports `None` (default exact-match voting), but the Pydantic validator currently rejects it. This would simplify setup for users who don't need regex-based answer extraction.
