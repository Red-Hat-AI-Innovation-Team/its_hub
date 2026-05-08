# its_hub Coding Agent Plugin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plugin for Claude Code and Cursor that lets users apply inference-time scaling (self-consistency, best-of-n, particle filtering) to their LLM workflows via commands, skills, and shell scripts.

**Architecture:** The plugin is a set of markdown files (commands, skills) and shell scripts that live at the repo root alongside the existing `its_hub` Python library. Commands are slash-command entry points, skills fire contextually, and scripts handle environment detection, server lifecycle, and scaling execution. The IaaS HTTP API is the primary backend; direct Python is the fallback.

**Tech Stack:** Bash (scripts), Markdown (commands/skills), JSON (config/manifests), Python (IaaS prerequisite fix), `curl`/`jq` (API calls)

**Spec:** `docs/superpowers/specs/2026-05-08-claude-code-plugin-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `.claude-plugin/plugin.json` | Claude Code manifest |
| `.cursor-plugin/plugin.json` | Cursor manifest |
| `scripts/its_detect.sh` | Environment detection: server status, library, installer, config |
| `scripts/its_server.sh` | IaaS server lifecycle: start, stop, status |
| `scripts/its_scale.sh` | Execute scaling requests via IaaS API or Python fallback |
| `commands/its-setup.md` | `/its-setup` — guided first-run configuration |
| `commands/its-server.md` | `/its-server` — server lifecycle command |
| `commands/its-scale.md` | `/its-scale` — single prompt scaling |
| `commands/its-scale-batch.md` | `/its-scale-batch` — batch scaling from file |
| `skills/inference-scaling/SKILL.md` | Contextual skill: detects scaling intent, routes to commands |
| `skills/setup-guide/SKILL.md` | Contextual skill: first-time setup walkthrough |

### Modified files

| File | Change |
|---|---|
| `its_hub/integration/iaas.py` | Add `ParticleFilteringResult` metadata extraction |
| `tests/test_iaas.py` | Add test for particle filtering metadata |
| `.gitignore` | Add `.its-hub/` entry |

---

## Task 1: Plugin Manifests

**Files:**
- Create: `.claude-plugin/plugin.json`
- Create: `.cursor-plugin/plugin.json`

- [ ] **Step 1: Create Claude Code manifest**

Create `.claude-plugin/plugin.json`:
```json
{
  "name": "its-hub",
  "description": "Inference-time scaling for LLMs — generate multiple candidates and select the best using voting, scoring, or search",
  "author": {
    "name": "Red Hat AI Innovation Team"
  }
}
```

- [ ] **Step 2: Create Cursor manifest**

Create `.cursor-plugin/plugin.json` with identical content.

- [ ] **Step 3: Add `.its-hub/` to `.gitignore`**

Append to the existing `.gitignore`:
```
# Plugin user config (may contain API keys)
.its-hub/
```

- [ ] **Step 4: Commit**

```bash
git add .claude-plugin/plugin.json .cursor-plugin/plugin.json .gitignore
git commit -s -m "Add plugin manifests for Claude Code and Cursor"
```

---

## Task 2: ParticleFilteringResult Metadata (IaaS Prerequisite)

**Files:**
- Modify: `its_hub/integration/iaas.py` (function `_extract_algorithm_metadata`, around line 410)
- Modify: `tests/test_iaas.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_iaas.py`, add to `TestPydanticModels` or create a new class:

```python
class TestAlgorithmMetadata:
    """Test algorithm metadata extraction."""

    def test_particle_filtering_metadata_extraction(self):
        """Test that ParticleFilteringResult produces metadata."""
        from its_hub.algorithms.particle_gibbs import ParticleFilteringResult
        from its_hub.integration.iaas import _extract_algorithm_metadata

        result = ParticleFilteringResult(
            responses=[
                {"role": "assistant", "content": "response1"},
                {"role": "assistant", "content": "response2"},
                {"role": "assistant", "content": "response3"},
            ],
            log_weights_lst=[-0.5, -1.2, -0.3],
            selected_index=2,
            steps_used_lst=[5, 3, 7],
        )

        metadata = _extract_algorithm_metadata(result)

        assert metadata is not None
        assert metadata["algorithm"] == "particle-filtering"
        assert metadata["log_weights_lst"] == [-0.5, -1.2, -0.3]
        assert metadata["selected_index"] == 2
        assert metadata["steps_used_lst"] == [5, 3, 7]
        assert len(metadata["responses"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_iaas.py::TestAlgorithmMetadata::test_particle_filtering_metadata_extraction -v`
Expected: FAIL — metadata is `None` because no branch exists for `ParticleFilteringResult`

- [ ] **Step 3: Implement metadata extraction**

In `its_hub/integration/iaas.py`, in `_extract_algorithm_metadata`, add a branch after the `BestOfNResult` block:

```python
from its_hub.algorithms.particle_gibbs import ParticleFilteringResult

# ... existing branches ...

elif isinstance(algorithm_result, ParticleFilteringResult):
    return {
        "algorithm": "particle-filtering",
        "responses": algorithm_result.responses,
        "log_weights_lst": algorithm_result.log_weights_lst,
        "selected_index": algorithm_result.selected_index,
        "steps_used_lst": algorithm_result.steps_used_lst,
    }
```

Also add the import at the top of `_extract_algorithm_metadata`:
```python
from its_hub.algorithms.particle_gibbs import ParticleFilteringResult
```

Remove the dead TODO comment block below the `BestOfNResult` branch (the commented-out stubs for `BestOfNResult` and `BeamSearchResult` that are no longer needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_iaas.py::TestAlgorithmMetadata -v`
Expected: PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check its_hub/integration/iaas.py && ruff format --check its_hub/integration/iaas.py`

- [ ] **Step 6: Commit**

```bash
git add its_hub/integration/iaas.py tests/test_iaas.py
git commit -s -m "Add ParticleFilteringResult metadata extraction to IaaS"
```

---

## Task 3: Detection Script (`its_detect.sh`)

**Files:**
- Create: `scripts/its_detect.sh`

This is the foundation — all commands and skills depend on it.

- [ ] **Step 1: Create the detection script**

```bash
mkdir -p scripts
```

Create `scripts/its_detect.sh`:

```bash
#!/usr/bin/env bash
# Detect its_hub environment: server, library, installer, config
set -euo pipefail

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"
IAAS_PORT=8108

# Read port from config if available
if [ -f "$CONFIG_PATH" ]; then
    CONFIGURED_PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('iaas_port', 8108))" 2>/dev/null || echo 8108)
    IAAS_PORT="$CONFIGURED_PORT"
fi

# Check IaaS server
if curl -s --connect-timeout 2 "http://localhost:${IAAS_PORT}/v1/models" > /dev/null 2>&1; then
    echo "server=running"
else
    echo "server=stopped"
fi

# Check library
if python3 -c "import its_hub" 2>/dev/null; then
    echo "library=installed"
else
    echo "library=missing"
fi

# Check installer
if command -v uv > /dev/null 2>&1; then
    echo "installer=uv"
elif command -v pip > /dev/null 2>&1; then
    echo "installer=pip"
else
    echo "installer=none"
fi

# Check config
if [ -f "$CONFIG_PATH" ]; then
    echo "config=found"
else
    echo "config=missing"
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/its_detect.sh
```

- [ ] **Step 3: Test manually**

```bash
./scripts/its_detect.sh
```

Expected: four `key=value` lines (values depend on environment)

- [ ] **Step 4: Commit**

```bash
git add scripts/its_detect.sh
git commit -s -m "Add environment detection script for plugin"
```

---

## Task 4: Server Management Script (`its_server.sh`)

**Files:**
- Create: `scripts/its_server.sh`

- [ ] **Step 1: Create the server script**

Create `scripts/its_server.sh`:

```bash
#!/usr/bin/env bash
# Manage its_hub IaaS server lifecycle
set -euo pipefail

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"
PID_FILE=".its-hub/server.pid"
ACTION="${1:-status}"

die() { echo "ERROR: $1" >&2; exit 1; }

read_config() {
    [ -f "$CONFIG_PATH" ] || die "Config not found at $CONFIG_PATH. Run /its-setup first."
    python3 -c "import json,sys; c=json.load(open('$CONFIG_PATH')); print(c.get('$1', '${2:-}'))"
}

build_configure_payload() {
    CONFIG_PATH="$CONFIG_PATH" MODEL_KEY="${1:-default}" python3 -c "
import json, sys, os

config = json.load(open(os.environ['CONFIG_PATH']))
model_key = os.environ['MODEL_KEY']
model_cfg = config.get('models', {}).get(model_key, {})
alg_cfg = config.get('algorithm_config', {})

payload = {
    'provider': config.get('provider', 'openai'),
    'endpoint': model_cfg.get('endpoint', ''),
    'api_key': model_cfg.get('api_key'),
    'model': model_cfg.get('model', ''),
    'alg': config.get('algorithm', ''),
}

# Add extra_args if present
extra = config.get('extra_args')
if extra:
    payload['extra_args'] = extra

# Map algorithm_config fields to flat /configure fields
field_map = {
    'regex_patterns': 'regex_patterns',
    'tool_vote': 'tool_vote',
    'exclude_tool_args': 'exclude_tool_args',
    'rm_name': 'rm_name',
    'rm_device': 'rm_device',
    'rm_agg_method': 'rm_agg_method',
    'step_token': 'step_token',
    'stop_token': 'stop_token',
    'judge_model': 'judge_model',
    'judge_base_url': 'judge_base_url',
    'judge_api_key': 'judge_api_key',
    'judge_criterion': 'judge_criterion',
    'judge_mode': 'judge_mode',
    'judge_top_n': 'judge_top_n',
    'judge_temperature': 'judge_temperature',
    'judge_max_tokens': 'judge_max_tokens',
    'enable_judge_logging': 'enable_judge_logging',
}

for src, dst in field_map.items():
    val = alg_cfg.get(src)
    if val is not None:
        payload[dst] = val

# Remove None values
payload = {k: v for k, v in payload.items() if v is not None}

json.dump(payload, sys.stdout)
"
}

case "$ACTION" in
    start)
        IAAS_PORT=$(read_config iaas_port 8108)

        # Check if already running
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Server already running (PID $(cat "$PID_FILE")) on port $IAAS_PORT"
            exit 0
        fi

        mkdir -p .its-hub

        echo "Starting IaaS server on port $IAAS_PORT..."
        nohup python3 -m its_hub.integration.iaas --host 0.0.0.0 --port "$IAAS_PORT" \
            > .its-hub/server.log 2>&1 &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PID_FILE"

        # Wait for server to be ready
        for i in $(seq 1 30); do
            if curl -s --connect-timeout 1 "http://localhost:${IAAS_PORT}/v1/models" > /dev/null 2>&1; then
                echo "Server started (PID $SERVER_PID)"

                # Configure the server
                PAYLOAD=$(build_configure_payload)
                RESPONSE=$(curl -s -X POST "http://localhost:${IAAS_PORT}/configure" \
                    -H "Content-Type: application/json" \
                    -d "$PAYLOAD")

                if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='success'" 2>/dev/null; then
                    echo "Server configured: $(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('message',''))")"
                else
                    echo "WARNING: Server started but configuration failed: $RESPONSE"
                fi
                exit 0
            fi
            sleep 1
        done

        echo "ERROR: Server failed to start within 30 seconds. Check .its-hub/server.log"
        kill "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
        ;;

    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "No server PID file found. Server may not be running."
            exit 0
        fi

        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Server stopped (PID $PID)"
        else
            echo "Server process $PID not found (stale PID file). Cleaning up."
        fi
        rm -f "$PID_FILE"
        ;;

    status)
        IAAS_PORT=$(read_config iaas_port 8108 2>/dev/null || echo 8108)

        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            PID=$(cat "$PID_FILE")
            MODELS=$(curl -s --connect-timeout 2 "http://localhost:${IAAS_PORT}/v1/models" 2>/dev/null || echo '{"data":[]}')
            echo "server=running pid=$PID port=$IAAS_PORT"
            echo "models=$MODELS"
        else
            echo "server=stopped"
            [ -f "$PID_FILE" ] && echo "(stale PID file found — cleaning up)" && rm -f "$PID_FILE"
        fi
        ;;

    *)
        echo "Usage: its_server.sh {start|stop|status}"
        exit 1
        ;;
esac
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/its_server.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/its_server.sh
git commit -s -m "Add IaaS server lifecycle management script"
```

---

## Task 5: Scaling Script (`its_scale.sh`)

**Files:**
- Create: `scripts/its_scale.sh`

- [ ] **Step 1: Create the scaling script**

Create `scripts/its_scale.sh`:

```bash
#!/usr/bin/env bash
# Execute inference-time scaling via IaaS API or Python fallback
set -euo pipefail

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"

usage() {
    echo "Usage: its_scale.sh [OPTIONS] <prompt>"
    echo ""
    echo "Options:"
    echo "  --algorithm ALG    Override algorithm (self-consistency|best-of-n|particle-filtering)"
    echo "  --budget N         Override budget (default: from config)"
    echo "  --model KEY        Override model (key from config models dict)"
    echo "  --metadata         Include full algorithm metadata in output"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

# Parse arguments
ALGORITHM=""
BUDGET=""
MODEL_KEY="default"
SHOW_METADATA=false
PROMPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --model) MODEL_KEY="$2"; shift 2 ;;
        --metadata) SHOW_METADATA=true; shift ;;
        --help) usage ;;
        *) PROMPT="$1"; shift ;;
    esac
done

[ -z "$PROMPT" ] && die "No prompt provided. Usage: its_scale.sh <prompt>"
[ -f "$CONFIG_PATH" ] || die "Config not found at $CONFIG_PATH. Run /its-setup first."

# Read config
IAAS_PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('iaas_port', 8108))")
[ -z "$ALGORITHM" ] && ALGORITHM=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('algorithm', 'self-consistency'))")
[ -z "$BUDGET" ] && BUDGET=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('budget', 8))")
MODEL_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('models', {}).get('$MODEL_KEY', {}).get('model', ''))")

[ -z "$MODEL_NAME" ] && die "Model '$MODEL_KEY' not found in config. Run /its-setup to add it."

# Determine return_response_only based on metadata flag
RETURN_RESPONSE_ONLY=true
$SHOW_METADATA && RETURN_RESPONSE_ONLY=false

# Try IaaS server first
if curl -s --connect-timeout 2 "http://localhost:${IAAS_PORT}/v1/models" > /dev/null 2>&1; then
    # IaaS path — use env vars to avoid shell injection
    REQUEST=$(ITS_MODEL="$MODEL_NAME" ITS_PROMPT="$PROMPT" ITS_BUDGET="$BUDGET" ITS_RRO="$RETURN_RESPONSE_ONLY" python3 -c "
import json, sys, os
req = {
    'model': os.environ['ITS_MODEL'],
    'messages': [{'role': 'user', 'content': os.environ['ITS_PROMPT']}],
    'budget': int(os.environ['ITS_BUDGET']),
    'return_response_only': os.environ['ITS_RRO'] == 'true'
}
json.dump(req, sys.stdout)
")

    RESPONSE=$(curl -s -X POST "http://localhost:${IAAS_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$REQUEST")

    # Check for errors
    if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'error' not in d.get('detail','')" 2>/dev/null; then
        echo "$RESPONSE"
    else
        echo "ERROR: $RESPONSE" >&2
        exit 1
    fi
else
    # Python fallback — limited to self-consistency and best-of-n with llm-judge
    if [ "$ALGORITHM" = "particle-filtering" ]; then
        die "Particle filtering requires a running IaaS server. Run: scripts/its_server.sh start"
    fi

    RM_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('algorithm_config', {}).get('rm_name', ''))" 2>/dev/null || echo "")
    if [ "$ALGORITHM" = "best-of-n" ] && [ "$RM_NAME" != "llm-judge" ] && [ -n "$RM_NAME" ]; then
        die "Best-of-N with local reward model requires a running IaaS server. Run: scripts/its_server.sh start"
    fi

    ITS_CONFIG="$CONFIG_PATH" ITS_MODEL_KEY="$MODEL_KEY" ITS_ALGORITHM="$ALGORITHM" \
    ITS_PROMPT="$PROMPT" ITS_BUDGET="$BUDGET" ITS_METADATA="$SHOW_METADATA" \
    python3 -c "
import json, sys, os

config = json.load(open(os.environ['ITS_CONFIG']))
model_key = os.environ['ITS_MODEL_KEY']
model_cfg = config.get('models', {}).get(model_key, {})
alg_cfg = config.get('algorithm_config', {})
prompt = os.environ['ITS_PROMPT']
budget = int(os.environ['ITS_BUDGET'])
show_metadata = os.environ['ITS_METADATA'] == 'true'

from its_hub.lms import OpenAICompatibleLanguageModel, LiteLLMLanguageModel

provider = config.get('provider', 'openai')
if provider == 'litellm':
    lm = LiteLLMLanguageModel(
        model_name=model_cfg['model'],
        api_key=model_cfg.get('api_key'),
        api_base=model_cfg.get('endpoint'),
    )
else:
    lm = OpenAICompatibleLanguageModel(
        endpoint=model_cfg['endpoint'],
        api_key=model_cfg.get('api_key', ''),
        model_name=model_cfg['model'],
    )

algorithm = os.environ['ITS_ALGORITHM']
if algorithm == 'self-consistency':
    from its_hub.algorithms import SelfConsistency
    from its_hub.algorithms.self_consistency import create_regex_projection_function
    patterns = alg_cfg.get('regex_patterns')
    proj = create_regex_projection_function(patterns) if patterns else None
    alg = SelfConsistency(proj, tool_vote=alg_cfg.get('tool_vote'), exclude_args=alg_cfg.get('exclude_tool_args'))
elif algorithm == 'best-of-n':
    from its_hub.algorithms import BestOfN
    from its_hub.integration.reward_hub import LLMJudgeRewardModel
    judge = LLMJudgeRewardModel(
        model=alg_cfg['judge_model'],
        criterion=alg_cfg.get('judge_criterion', 'overall_quality'),
        judge_type=alg_cfg.get('judge_mode', 'groupwise'),
        api_key=alg_cfg.get('judge_api_key'),
        base_url=alg_cfg.get('judge_base_url'),
    )
    alg = BestOfN(judge)

result = alg.infer(lm, prompt, budget=budget, return_response_only=not show_metadata)

if show_metadata and hasattr(result, 'the_one'):
    output = {'selected': result.the_one, 'type': type(result).__name__}
    print(json.dumps(output, default=str))
else:
    if isinstance(result, dict):
        print(json.dumps(result))
    else:
        print(result)
"
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/its_scale.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/its_scale.sh
git commit -s -m "Add scaling execution script with IaaS and Python fallback"
```

---

## Task 6: Setup Command (`/its-setup`)

**Files:**
- Create: `commands/its-setup.md`

- [ ] **Step 1: Create the setup command**

```bash
mkdir -p commands
```

Create `commands/its-setup.md`:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add commands/its-setup.md
git commit -s -m "Add /its-setup command for guided plugin configuration"
```

---

## Task 7: Server Command (`/its-server`)

**Files:**
- Create: `commands/its-server.md`

- [ ] **Step 1: Create the server command**

Create `commands/its-server.md`:

````markdown
---
description: "Manage the IaaS inference-time scaling server"
argument-hint: "start|stop|status"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# its-hub Server Management

Manage the its_hub Inference-as-a-Service (IaaS) server.

## Usage

Run the server management script with the requested action:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh" $ARGUMENTS
```

Report the result to the user:
- **start**: Confirm the server is running, show the port and configured model/algorithm
- **stop**: Confirm the server has been stopped
- **status**: Show whether the server is running, its PID, port, and configured models

If no arguments are provided, default to `status`.

If the config is missing, suggest running `/its-setup` first.
````

- [ ] **Step 2: Commit**

```bash
git add commands/its-server.md
git commit -s -m "Add /its-server command for IaaS lifecycle management"
```

---

## Task 8: Scale Command (`/its-scale`)

**Files:**
- Create: `commands/its-scale.md`

- [ ] **Step 1: Create the scale command**

Create `commands/its-scale.md`:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add commands/its-scale.md
git commit -s -m "Add /its-scale command for single prompt scaling"
```

---

## Task 9: Batch Scale Command (`/its-scale-batch`)

**Files:**
- Create: `commands/its-scale-batch.md`

- [ ] **Step 1: Create the batch command**

Create `commands/its-scale-batch.md`:

````markdown
---
description: "Run inference-time scaling on a batch of prompts from a file"
argument-hint: "<file> [--output <file>] [--concurrency N]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_scale.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh:*)"]
---

# its-hub Batch Scale

Run inference-time scaling on multiple prompts from a file.

## Supported Input Formats

- **JSONL** — one JSON object per line with a `prompt` or `messages` field
- **CSV** — must have a `prompt` column
- **TXT** — one prompt per line

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

If `config=missing`, tell the user to run `/its-setup` first.
If `server=stopped`, recommend starting it: "Batch processing works best with the IaaS server. Want me to start it?"

## Step 2: Parse Arguments

Extract from `$ARGUMENTS`:
- `file` — the input file path (required)
- `--output` — output file path (default: `<input_name>_scaled.jsonl`)
- `--concurrency` — number of parallel requests (default: 8, only used with IaaS)

Validate the input file exists and detect its format from the extension.

## Step 3: Read and Validate Input

Read the file and extract prompts based on format:
- JSONL: parse each line, extract `prompt` or `messages` field
- CSV: read with Python csv module, extract `prompt` column
- TXT: each line is a prompt

Report: "Found N prompts in <filename>"

## Step 4: Process Prompts

For each prompt, call the scaling script. Use `xargs -P` or a Python asyncio loop for parallelism when using IaaS.

Write each result to the output file as a JSONL line:
```json
{"prompt": "...", "selected_response": "...", "algorithm": "...", "budget": N, "metadata": {...}}
```

If a prompt fails, write an error entry and continue:
```json
{"prompt": "...", "error": "error message", "algorithm": "...", "budget": N}
```

## Step 5: Report Summary

Report: "N/M prompts completed successfully. K failed. Results written to <output_file>"

If there were failures, list the line numbers and error messages.
````

- [ ] **Step 2: Commit**

```bash
git add commands/its-scale-batch.md
git commit -s -m "Add /its-scale-batch command for batch processing"
```

---

## Task 10: Setup Guide Skill

**Files:**
- Create: `skills/setup-guide/SKILL.md`

- [ ] **Step 1: Create the skill**

```bash
mkdir -p skills/setup-guide skills/inference-scaling
```

Create `skills/setup-guide/SKILL.md`:

````markdown
---
name: setup-guide
description: "Use when the user wants to set up inference-time scaling for the first time, or when its_hub is not yet installed/configured in the current environment."
---

# its_hub Setup Guide

You are helping the user set up inference-time scaling for the first time.

## Detection

First, detect the environment by running:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

## If Nothing is Installed

1. Explain what its_hub does: "its_hub is a library for inference-time scaling — it generates multiple LLM responses and selects the best one using voting, scoring, or search algorithms."
2. Ask permission: "I can install it for you. This will add the `its_hub` Python package to your environment. Want me to proceed?"
3. If yes: install using the detected installer (`uv pip install its_hub` or `pip install its_hub`)
4. Proceed to configuration

## Configuration

Invoke the `/its-setup` command to walk through configuration:
- Provider, endpoint, API key, model
- Algorithm choice with explanations
- Algorithm-specific settings

## After Setup

Once configured, hand off to the `inference-scaling` skill if the user had an original scaling request. Otherwise, tell the user:
- "You're all set! You can now use `/its-scale <prompt>` to run inference-time scaling."
- Mention `/its-server` for server management and `/its-scale-batch` for batch processing.
````

- [ ] **Step 2: Commit**

```bash
git add skills/setup-guide/SKILL.md
git commit -s -m "Add setup-guide skill for first-time configuration"
```

---

## Task 11: Inference Scaling Skill

**Files:**
- Create: `skills/inference-scaling/SKILL.md`

- [ ] **Step 1: Create the skill**

Create `skills/inference-scaling/SKILL.md`:

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add skills/inference-scaling/SKILL.md
git commit -s -m "Add inference-scaling skill for contextual scaling"
```

---

## Task 12: Final Integration Verification

- [ ] **Step 1: Verify file structure**

```bash
find .claude-plugin .cursor-plugin commands skills scripts/its_*.sh -type f | sort
```

Expected:
```
.claude-plugin/plugin.json
.cursor-plugin/plugin.json
commands/its-scale-batch.md
commands/its-scale.md
commands/its-server.md
commands/its-setup.md
scripts/its_detect.sh
scripts/its_scale.sh
scripts/its_server.sh
skills/inference-scaling/SKILL.md
skills/setup-guide/SKILL.md
```

- [ ] **Step 2: Verify scripts are executable**

```bash
ls -la scripts/its_*.sh
```

All should have `+x` permission.

- [ ] **Step 3: Run detection script**

```bash
./scripts/its_detect.sh
```

Should output four `key=value` lines without errors.

- [ ] **Step 4: Run ruff on modified Python files**

```bash
ruff check its_hub/integration/iaas.py
ruff format --check its_hub/integration/iaas.py
```

- [ ] **Step 5: Run test suite**

```bash
pytest tests/test_iaas.py -v
```

- [ ] **Step 6: Verify .gitignore**

```bash
grep -q '.its-hub/' .gitignore && echo "OK" || echo "MISSING"
```

- [ ] **Step 7: Final commit if any loose changes**

```bash
git status
# If clean: no action needed
# If changes: commit with descriptive message
```
