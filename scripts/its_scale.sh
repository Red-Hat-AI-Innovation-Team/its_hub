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

# Read config values via env vars to avoid shell injection
IAAS_PORT=$(CONFIG_PATH="$CONFIG_PATH" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('iaas_port', 8108))")
[ -z "$ALGORITHM" ] && ALGORITHM=$(CONFIG_PATH="$CONFIG_PATH" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('algorithm', 'self-consistency'))")
[ -z "$BUDGET" ] && BUDGET=$(CONFIG_PATH="$CONFIG_PATH" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('budget', 8))")
MODEL_NAME=$(CONFIG_PATH="$CONFIG_PATH" MODEL_KEY="$MODEL_KEY" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('models', {}).get(os.environ['MODEL_KEY'], {}).get('model', ''))")

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
    if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'choices' in d else 1)" 2>/dev/null; then
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

    RM_NAME=$(CONFIG_PATH="$CONFIG_PATH" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('algorithm_config', {}).get('rm_name', ''))" 2>/dev/null || echo "")
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
