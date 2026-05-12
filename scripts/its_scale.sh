#!/usr/bin/env bash
# Execute inference-time scaling via Python library
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"

usage() {
    echo "Usage: its_scale.sh [OPTIONS] <prompt>"
    echo ""
    echo "Options:"
    echo "  --algorithm ALG    Override algorithm (self-consistency|best-of-n)"
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
[ -z "$ALGORITHM" ] && ALGORITHM=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('algorithm', 'self-consistency'))")
[ -z "$BUDGET" ] && BUDGET=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('budget', 8))")
MODEL_NAME=$(CONFIG_PATH="$CONFIG_PATH" MODEL_KEY="$MODEL_KEY" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('models', {}).get(os.environ['MODEL_KEY'], {}).get('model', ''))")

[ -z "$MODEL_NAME" ] && die "Model '$MODEL_KEY' not found in config. Run /its-setup to add it."

# Only self-consistency and best-of-n are supported in v1 via this script
if [ "$ALGORITHM" = "particle-filtering" ] || [ "$ALGORITHM" = "beam-search" ]; then
    die "$ALGORITHM requires process reward models and is experimental in v1. Use the Python API directly for advanced algorithms."
fi

ITS_CONFIG="$CONFIG_PATH" ITS_MODEL_KEY="$MODEL_KEY" ITS_ALGORITHM="$ALGORITHM" \
ITS_PROMPT="$PROMPT" ITS_BUDGET="$BUDGET" ITS_METADATA="$SHOW_METADATA" \
$PYTHON -c "
import json, sys, os

config = json.load(open(os.environ['ITS_CONFIG']))
model_key = os.environ['ITS_MODEL_KEY']
model_cfg = config.get('models', {}).get(model_key, {})
alg_cfg = config.get('algorithm_config', {})
prompt = os.environ['ITS_PROMPT']
budget = int(os.environ['ITS_BUDGET'])
show_metadata = os.environ['ITS_METADATA'] == 'true'

from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint=model_cfg['endpoint'],
    api_key=model_cfg.get('api_key', ''),
    model_name=model_cfg['model'],
)

algorithm = os.environ['ITS_ALGORITHM']
if algorithm == 'self-consistency':
    from its_hub.core.algorithms.self_consistency import SelfConsistency, create_regex_projection_function
    patterns = alg_cfg.get('regex_patterns')
    proj = create_regex_projection_function(patterns) if patterns else None
    alg = SelfConsistency(proj, tool_vote=alg_cfg.get('tool_vote'), exclude_args=alg_cfg.get('exclude_tool_args'))
elif algorithm == 'best-of-n':
    from its_hub.core.algorithms.bon import BestOfN
    from its_hub.core.reward_models.llm_judge import LLMJudge

    judge_cfg = {k: v for k, v in alg_cfg.items() if k.startswith('judge_')}
    judge_endpoint = judge_cfg.get('judge_endpoint', model_cfg['endpoint'])
    judge_api_key = judge_cfg.get('judge_api_key', model_cfg.get('api_key', ''))
    judge_model = judge_cfg.get('judge_model', model_cfg['model'])

    judge_lm = OpenAICompatibleLanguageModel(
        endpoint=judge_endpoint,
        api_key=judge_api_key,
        model_name=judge_model,
    )
    judge = LLMJudge(lm=judge_lm)
    alg = BestOfN(judge)
else:
    print(json.dumps({'error': f'Unsupported algorithm: {algorithm}'}))
    sys.exit(1)

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
