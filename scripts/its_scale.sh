#!/usr/bin/env bash
# Execute inference-time scaling via Python library.
#
# Usage: its_scale.sh [OPTIONS] <prompt>
# Options:
#   --algorithm ALG   Override algorithm (self-consistency|best-of-n)
#   --budget N        Override budget (default: from config)
#   --model KEY       Override model (key from config models dict)
#   --metadata        Include full algorithm metadata in output
# Output: JSON with selected response and optional metadata.
#
# Reads config from .its-hub/config.json (override via ITS_HUB_CONFIG).
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

ITS_CONFIG="$CONFIG_PATH" ITS_MODEL_KEY="$MODEL_KEY" ITS_ALGORITHM="$ALGORITHM" \
ITS_PROMPT="$PROMPT" ITS_BUDGET="$BUDGET" ITS_METADATA="$SHOW_METADATA" \
$PYTHON "$_SCRIPTS_DIR/_its_scale_runner.py"
