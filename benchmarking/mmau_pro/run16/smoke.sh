#!/bin/bash
# Sanity smoke test. Phase A is offline and exact on any machine; Phase B serves
# each model once (1 replica, GPU 0), asserts both phase0 gates, and runs a
# 4-item budget-64 probe cell end-to-end. ~30-45 min total.
# SMOKE_PHASE=A runs only the offline half (no GPU needed); default runs both.
set -euo pipefail
: "${SMOKE_PHASE:=AB}"
RUN16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN16_DIR/config.sh"
source "$RUN16_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/smoke" "$OUT_ROOT/servers"

echo "=== SMOKE PHASE A (offline: tests, data, scoring, checkpoint identity) ==="
"$EPF_PY" -m pytest tests/ -q --ignore=tests/e2e     # expect: 104 passed
"$EPF_PY" - "$DATA_TESTMINI" "$DATA_AUDIO" <<'PYEOF'
import sys
from benchmarking.mmau_pro.loader import load_mmau_mcq
from benchmarking.mmau_pro.scoring import extract_letter, match_answer_index, predicted_index
tm, au = sys.argv[1], sys.argv[2]
t = load_mmau_mcq(tm, subset="test", audio_root=au)
assert (len(t), sum(1 for r in t if r.answer_index is None)) == (5090, 24)
assert len(load_mmau_mcq(tm, subset="full")) == 957
assert len(load_mmau_mcq(tm, subset="test_le30s", audio_root=au)) == 2190
assert extract_letter('The answer is clear.\n\nAnswer: B', 4) == 1
assert extract_letter('Step 1: ...\nFinal Answer: \\boxed{C}', 5) == 2
assert predicted_index('I think it is the dog barking', ['cat', 'dog barking', 'dog']) == 1
assert match_answer_index('The Dog barking!', ['cat', 'dog barking', 'dog']) == 1
print("PHASE A data + scoring: OK (exact SETUP_GUIDE §12 E1 values)")
PYEOF
for run in $RUN_MODELS; do
  run_cfg "$run"
  check_snapshot "$MODEL_ID" "$REV"
done
echo "=== PHASE A PASS ==="
[ "$SMOKE_PHASE" = "A" ] && { echo "(SMOKE_PHASE=A — skipping the served Phase B)"; exit 0; }

echo "=== SMOKE PHASE B (serve, gate, tiny b64 cell — per model) ==="
for run in $RUN_MODELS; do
  run_cfg "$run"
  echo "-- $run: $MODEL_ID as '$NAME'"
  serve_one "$MODEL_ID" "$REV" "$NAME" "$MAXLEN" 0
  wait_healthy "$BASE_PORT" 1800
  gate_endpoint "$BASE_PORT" "$NAME"
  SM="$OUT_ROOT/smoke/${run}_smoke"
  rm -f "$SM.jsonl"
  "$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "http://localhost:$BASE_PORT/v1" --model-name "$NAME" \
    --data-root "$DATA_TESTMINI" --subset "$SUBSET" --audio-root "$DATA_AUDIO" \
    --prompts "${PROMPTS%%,*}" --signals mean_logprob --budgets 64 \
    --select all --limit 4 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight "$MAX_INFLIGHT" \
    --jsonl "$SM.jsonl" --csv "$SM.csv" --log "$SM.log"
  "$EPF_PY" "$RUN16_DIR/summarize_errors.py" "$SM.jsonl"
  if [ "$run" = "run11" ]; then
    # E3 soft check: two fixed testmini items, greedy. Letters are STACK-SPECIFIC —
    # a difference on other GPUs is a WARN, not a failure (SETUP_GUIDE §12 E3).
    "$EPF_PY" -m benchmarking.mmau_pro.cot_compare \
      --endpoint "http://localhost:$BASE_PORT/v1" --model-name "$NAME" \
      --data-root "$DATA_TESTMINI" --subset full \
      --ids 2035bce6-a746-4a82-82c1-d61da27cb533,69c911db-5532-4677-b28e-77eb231e6d24 \
      --audio-mode local-path --csv "$OUT_ROOT/smoke/e3_soft.csv" \
      || echo "WARN: E3 soft check errored (non-fatal)"
    echo "NOTE: compare letters against SETUP_GUIDE §12 E3 (reference: methods 2-9 -> C/D, method 1 -> D/D)."
  fi
  kill_servers "$NAME"
done
echo "=== SMOKE PASS — ready for: nohup bash $RUN16_DIR/run_all.sh > run16.log 2>&1 & ==="
