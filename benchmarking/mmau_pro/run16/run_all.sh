#!/bin/bash
# THE single command: extend Runs 11/12/15 to budgets 64 & 128 and build the
# combined bootstrap HTML. Fully resumable — on crash/interrupt, re-run this
# same script; completed cells are skipped via the probe's JSONL resume.
#
#   nohup bash benchmarking/mmau_pro/run16/run_all.sh > run16.log 2>&1 &
#
# Output: $OUT_ROOT/epf_full5090_bootstrap_b128.html (a NEW file — the original
# b1-32 report is never touched). DRY_RUN=1 prints the plan without executing.
set -euo pipefail
RUN16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN16_DIR/config.sh"
source "$RUN16_DIR/lib.sh"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT/servers"
SUMMARY="$OUT_ROOT/summary.txt"
echo "=== RUN16 start $(date '+%F %T') | GPUs=$NUM_GPUS | budgets=$BUDGETS | runs=$RUN_MODELS ===" | tee -a "$SUMMARY"

probe_stage() {  # probe_stage <budget> — uses the run_cfg vars; flags replicate Runs 11/12/15
  "$EPF_PY" -m benchmarking.mmau_pro.diversity_probe \
    --endpoints "$(endpoints_csv)" --model-name "$NAME" \
    --data-root "$DATA_TESTMINI" --subset "$SUBSET" --audio-root "$DATA_AUDIO" \
    --prompts "$PROMPTS" --signals mean_logprob,entropy --budgets "$1" \
    --select all --limit "$PROBE_LIMIT" --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight "$MAX_INFLIGHT" \
    --jsonl "$RUN_DIR/$STEM.jsonl" --csv "$RUN_DIR/$STEM.csv" --log "$RUN_DIR/$STEM.log"
}

for run in $RUN_MODELS; do
  run_cfg "$run"
  mkdir -p "$RUN_DIR"
  unpack_seed "$RUN16_DIR/seeds/$STEM.seed.jsonl.gz" "$RUN_DIR/$STEM.jsonl"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY: $run -> model=$MODEL_ID@${REV:0:12} name=$NAME subset=$SUBSET prompts=$PROMPTS" \
         "budgets=[$BUDGETS] endpoints=$(endpoints_csv) out=$RUN_DIR/$STEM.*"
    continue
  fi

  ensure_model "$MODEL_ID" "$REV"

  for i in $(seq 0 $((NUM_GPUS - 1))); do serve_one "$MODEL_ID" "$REV" "$NAME" "$MAXLEN" "$i"; done
  for i in $(seq 0 $((NUM_GPUS - 1))); do wait_healthy $((BASE_PORT + i)) 1800; done
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    gate_endpoint $((BASE_PORT + i)) "$NAME" || { kill_servers "$NAME"; exit 1; }
  done

  for B in $BUDGETS; do
    t0=$(date +%s)
    probe_stage "$B"
    if ! "$EPF_PY" "$RUN16_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B"; then
      echo "$run b$B: errors detected — one automatic resume retry" | tee -a "$SUMMARY"
      probe_stage "$B"
      "$EPF_PY" "$RUN16_DIR/summarize_errors.py" "$RUN_DIR/$STEM.jsonl" --min-budget "$B" \
        || echo "WARNING: $run b$B still has errors after retry — check $RUN_DIR/$STEM.log and server logs" | tee -a "$SUMMARY"
    fi
    echo "$run b$B stage done in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$SUMMARY"
  done
  kill_servers "$NAME"
done

if [ "$DRY_RUN" = "1" ]; then echo "DRY RUN complete — no commands executed"; exit 0; fi

# Combined report: one section per run whose CSV exists (all three on a full run;
# fewer during rehearsals with a restricted RUN_MODELS).
IN_ARGS=()
run_cfg run11; [ -e "$RUN_DIR/$STEM.csv" ] && IN_ARGS+=(--in "Qwen2.5-Omni-7B — Runs 11+16=$RUN_DIR/$STEM.csv")
run_cfg run12; [ -e "$RUN_DIR/$STEM.csv" ] && IN_ARGS+=(--in "Qwen2.5-Omni-3B — Runs 12+16=$RUN_DIR/$STEM.csv")
run_cfg run15; [ -e "$RUN_DIR/$STEM.csv" ] && IN_ARGS+=(--in "Qwen2-Audio-7B-Instruct — Runs 15+16 (le30s)=$RUN_DIR/$STEM.csv")
[ ${#IN_ARGS[@]} -gt 0 ] || { echo "FATAL: no CSVs found for the report" >&2; exit 1; }
"$EPF_PY" -m benchmarking.mmau_pro.epf_bootstrap --n "$N_BOOT" "${IN_ARGS[@]}" \
  --out "$OUT_ROOT/epf_full5090_bootstrap_b128.html"

echo "=== RUN16 COMPLETE $(date '+%F %T') -> $OUT_ROOT/epf_full5090_bootstrap_b128.html ===" | tee -a "$SUMMARY"
