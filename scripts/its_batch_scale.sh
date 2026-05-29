#!/usr/bin/env bash
# Execute inference-time scaling on a batch of prompts from a file.
#
# Usage: its_batch_scale.sh [OPTIONS] <input-file>
# Options:
#   --algorithm ALG   Override algorithm (self-consistency|best-of-n)
#   --budget N        Override budget (default: from config)
#   --model KEY       Override model (key from config models dict)
#   --output FILE     Output file path (default: results/<input>_scaled.jsonl)
# Output: JSON summary with counts and output file path.
#
# Reads config from .its-hub/config.json (override via ITS_HUB_CONFIG).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"

usage() {
    echo "Usage: its_batch_scale.sh [OPTIONS] <input-file>"
    echo ""
    echo "Options:"
    echo "  --algorithm ALG    Override algorithm (self-consistency|best-of-n)"
    echo "  --budget N         Override budget (default: from config)"
    echo "  --model KEY        Override model (key from config models dict)"
    echo "  --output FILE      Output file path"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

# Parse arguments
ALGORITHM=""
BUDGET=""
MODEL_KEY="default"
OUTPUT_FILE=""
INPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algorithm) ALGORITHM="$2"; shift 2 ;;
        --budget) BUDGET="$2"; shift 2 ;;
        --model) MODEL_KEY="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --help) usage ;;
        -*) die "Unknown option: $1" ;;
        *) INPUT_FILE="$1"; shift ;;
    esac
done

[ -z "$INPUT_FILE" ] && die "No input file provided. Usage: its_batch_scale.sh <input-file>"
[ -f "$INPUT_FILE" ] || die "Input file not found: $INPUT_FILE"
[ -f "$CONFIG_PATH" ] || die "Config not found at $CONFIG_PATH. Run /its-setup first."

# Default output path: results/<input_name>_scaled.jsonl
if [ -z "$OUTPUT_FILE" ]; then
    BASENAME="$(basename "${INPUT_FILE%.*}")"
    OUTPUT_FILE="results/${BASENAME}_scaled.jsonl"
fi

ITS_CONFIG="$CONFIG_PATH" ITS_MODEL_KEY="$MODEL_KEY" ITS_ALGORITHM="$ALGORITHM" \
ITS_BUDGET="$BUDGET" ITS_INPUT="$INPUT_FILE" ITS_OUTPUT="$OUTPUT_FILE" \
$PYTHON "$_SCRIPTS_DIR/_its_batch_runner.py"
