# Run 16 configuration. Every knob is overridable from the environment, e.g.
#   NUM_GPUS=8 bash run_all.sh
# or via a config.local.sh (gitignored) next to this file — it is sourced last and wins.
# Source this file, then lib.sh, from every run16 script.

RUN16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RUN16_DIR/../../.." && pwd)"

# local overrides (plain VAR=value lines) — sourced FIRST so the := defaults
# below respect them, including everything derived from EPF_DATA_ROOT
[ -f "$RUN16_DIR/config.local.sh" ] && source "$RUN16_DIR/config.local.sh"

# --- machine / environment ----------------------------------------------------
: "${EPF_ENV_NAME:=epf}"
: "${CONDA_BASE:=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
: "${EPF_PY:=$CONDA_BASE/envs/$EPF_ENV_NAME/bin/python}"  # ALWAYS absolute — `conda activate` may lose the PATH race (SETUP_GUIDE §2)
: "${EPF_DATA_ROOT:=$HOME/epf_data}"       # will hold mmau_pro_testmini/ + mmau_pro_audio/ (~75 GB)
: "${HF_HOME:=$EPF_DATA_ROOT/hf_cache}"; export HF_HOME   # models land here (~49 GB)
: "${NUM_GPUS:=$(nvidia-smi -L 2>/dev/null | wc -l)}"
: "${BASE_PORT:=8100}"                     # one vLLM replica per GPU on BASE_PORT+i
: "${GPU_MEM_UTIL:=0.85}"                  # NOT 0.9 — audio-encoder attention spikes OOM'd an engine at 0.9 (SETUP_GUIDE §10)

# --- experiment ----------------------------------------------------------------
: "${BUDGETS:=64 128}"                     # staged in this order; b1–32 come from the committed seeds
: "${MAX_INFLIGHT:=64}"                    # per-endpoint item concurrency = max(1, MAX_INFLIGHT // budget)
: "${PROBE_LIMIT:=6000}"                   # >= 5,090 covers every item; set 4 to rehearse the driver
: "${N_BOOT:=10000}"                       # bootstrap resamples for the final HTML
: "${PROMPTS_RUN11:=5}"                    # default P5 only; 4,5,7,9 = full grid (~170 vs ~80 GPU-pair-h total)
: "${PROMPTS_RUN12:=4,5}"
: "${PROMPTS_RUN15:=4,9}"
: "${RUN_MODELS:=run11 run12 run15}"       # subset to rehearse a single run
: "${OUT_ROOT:=$REPO_ROOT/benchmarking/mmau_pro/results/run16_hibudget}"
: "${DRY_RUN:=0}"                          # 1 = print the per-run plan instead of executing

# --- per-run tables: model id, PINNED revision, served name, context, subset, file stem ---
# Served names replicate the original runs (cosmetic — not part of the resume key).
RUN11_MODEL_ID="Qwen/Qwen2.5-Omni-7B"
RUN11_REV="ae9e1690543ffd5c0221dc27f79834d0294cba00"
RUN11_NAME="qwen-omni";    RUN11_MAXLEN=32768; RUN11_SUBSET="test";       RUN11_STEM="epf_full5090"

RUN12_MODEL_ID="Qwen/Qwen2.5-Omni-3B"
RUN12_REV="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
RUN12_NAME="qwen-omni-3b"; RUN12_MAXLEN=32768; RUN12_SUBSET="test";       RUN12_STEM="epf3b_5090"

RUN15_MODEL_ID="Qwen/Qwen2-Audio-7B-Instruct"
RUN15_REV="0a095220c30b7b31434169c3086508ef3ea5bf0a"
RUN15_NAME="qwen2-audio";  RUN15_MAXLEN=8192;  RUN15_SUBSET="test_le30s"; RUN15_STEM="epf_q2a_le30s"

# --- dataset pins ----------------------------------------------------------------
PARQUET_DATASET="macabdul9/MMAU_Pro_Testmini"                     # ships testmini AND full-test parquets
PARQUET_REV="81eb01fb86bfc183dcb88b376ab1ca149a9d9c4b"
AUDIO_DATASET="gamma-lab-umd/MMAU-Pro"                            # data.zip: 44 GiB -> 5,787 files / 53 GB
AUDIO_ZIP_SHA256="8fab3e820b27bf7f239ae74a45a1085f1364b6971a7b5ac53f220d091b9b111c"

: "${DATA_TESTMINI:=$EPF_DATA_ROOT/mmau_pro_testmini}"
: "${DATA_AUDIO:=$EPF_DATA_ROOT/mmau_pro_audio}"
