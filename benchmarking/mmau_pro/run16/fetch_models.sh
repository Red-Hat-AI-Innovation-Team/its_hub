#!/bin/bash
# Download the three model checkpoints at the PINNED revisions and verify them.
# Idempotent — re-running only fills gaps.
set -euo pipefail
RUN16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN16_DIR/config.sh"
source "$RUN16_DIR/lib.sh"

for run in run11 run12 run15; do
  run_cfg "$run"
  echo "== $run: $MODEL_ID @ $REV"
  ensure_model "$MODEL_ID" "$REV"
done
echo "ALL MODELS OK (pinned revisions present in $HF_HOME)"
