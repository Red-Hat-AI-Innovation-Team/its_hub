# Run 16 — extend the MMAU-Pro EPF grids to budgets 64 & 128

This package lets you reproduce-and-extend three completed experiments (Runs 11/12/15 in
`benchmarking/mmau_pro/RESULTS.md`) with two higher particle budgets, on your own GPUs, with
**one command**. The b1–32 results ship in this repo as gzipped seeds — you only compute the
new b64/b128 cells, and the final HTML shows the full b1→128 curves.

| run | model (pinned revision) | items | grid extended |
|---|---|---|---|
| 11 | Qwen/Qwen2.5-Omni-7B `ae9e169…` | 5,090 MCQ (full MMAU-Pro test) | default **P5** × 2 signals (knob: `PROMPTS_RUN11=4,5,7,9` for the full grid) |
| 12 | Qwen/Qwen2.5-Omni-3B `f75b40e…` | 5,090 MCQ | P4,P5 × 2 signals |
| 15 | Qwen/Qwen2-Audio-7B-Instruct `0a09522…` | 2,190 MCQ (≤30 s clip subset) | P4,P9 × 2 signals |

## Requirements

- Linux box with NVIDIA GPUs (≥48 GB each recommended; the reference box used 2× 96 GB),
  CUDA driver for the cu130 wheel stack (or your own torch/vllm wheels, see below), conda,
  and a HuggingFace account (`hf auth login`) for the dataset download.
- Disk: ~110 GB data (44 GiB zip transient + 53 GB audio + parquets) + ~49 GB models + env.
- GPU time at defaults (`PROMPTS_RUN11=5`): ≈ **80 GPU-pair-hours** on 2× RTX PRO 6000
  (Blackwell) — scale by your GPU count/speed. Full Run-11 grid (`PROMPTS_RUN11=4,5,7,9`)
  ≈ 170. Budget-128 stages dominate.

## The whole flow

```bash
git clone <repo-url> && cd entropic-particle-filter
git checkout self-log-probs

export EPF_DATA_ROOT=/big/volume/epf_data   # optional — defaults to ~/epf_data; models go to $EPF_DATA_ROOT/hf_cache
export NUM_GPUS=8                           # optional — defaults to nvidia-smi count

bash benchmarking/mmau_pro/run16/setup_env.sh     # ONE conda env: 'epf'
bash benchmarking/mmau_pro/run16/fetch_data.sh    # parquets + audio (sha256-verified), exact-count checks
bash benchmarking/mmau_pro/run16/fetch_models.sh  # 3 checkpoints at pinned revisions
bash benchmarking/mmau_pro/run16/smoke.sh         # ~30-45 min: tests, data, gates, tiny b64 cell per model

nohup bash benchmarking/mmau_pro/run16/run_all.sh > run16.log 2>&1 &
tail -f run16.log
```

Result: `benchmarking/mmau_pro/results/run16_hibudget/epf_full5090_bootstrap_b128.html`
(three model sections, tables + interactive accuracy-vs-budget plots, bootstrap error bars).

**Crashed / interrupted / machine rebooted?** Re-run the same `nohup … run_all.sh` line.
Every completed (item × prompt × signal × budget) cell is skipped via the append-only JSONL
resume; servers are restarted as needed. Nothing is ever recomputed or lost.

## Knobs (env vars or `config.local.sh` next to `config.sh`)

| knob | default | meaning |
|---|---|---|
| `EPF_DATA_ROOT` | `~/epf_data` | where datasets + HF model cache live |
| `NUM_GPUS` | auto | one vLLM replica per GPU, ports 8100+i |
| `PROMPTS_RUN11` | `5` | `4,5,7,9` = full Run-11 grid (~2× total GPU time) |
| `BUDGETS` | `64 128` | staged in order |
| `MAX_INFLIGHT` | `64` | per-endpoint item concurrency = `max(1, MAX_INFLIGHT // budget)` |
| `RUN_MODELS` | `run11 run12 run15` | run a subset, e.g. rehearse with `run15` only |
| `GPU_MEM_UTIL` | `0.85` | do NOT raise to 0.9 — audio-encoder spikes OOM'd an engine |
| `DRY_RUN` | `0` | `1` prints the plan without executing |

## Things worth knowing

- **One env is enough.** All three models are served by the same vLLM 0.22.1 / torch 2.11.0
  stack; only serve-time flags differ (Qwen2-Audio runs at `--max-model-len 8192`, its full
  context). `requirements-epf.txt` pins the exact reference environment (CUDA-13.0 wheels) —
  on a different CUDA stack, install matching torch/vllm wheels first, then the rest of the
  pins. `uv` is only for offline unit tests here (it does not know about vllm/torch).
- **"Same checkpoint" is enforced three ways**: models are downloaded `--revision <sha>`,
  `smoke.sh` asserts the pinned snapshot exists and every weight shard in the safetensors
  index is present, and the servers start with `--revision <sha>` so they cannot load
  anything else.
- **Concurrency at high budgets**: at b≥64 only one item runs per endpoint at a time, but
  each item fans out up to `budget` concurrent particle requests per step (the client is
  unthrottled by design). vLLM queues this fine; raise `MAX_INFLIGHT` beyond 64 only if your
  servers stay healthy.
- **Expected smoke numbers**: 104 unit tests pass; loader counts test=5090 (24 ungradeable),
  full=957, test_le30s=2190; both phase0 gates PASS per endpoint; the 4-item b64 cell ends
  with 0 errors. The two-item E3 letter check is *stack-specific* — a mismatch on different
  GPUs is a WARN, not a failure.
- **A block of hundreds of identical connection errors** in a stage = a vLLM engine died
  (check `results/run16_hibudget/servers/*.log`), not a data problem. `run_all.sh` already
  retries once; if errors persist, just re-run it — resume sweeps up errored cells.

## What to send back

Tarball of `benchmarking/mmau_pro/results/run16_hibudget/`:
the HTML, the three `.csv`/`.log`, `summary.txt`, and (gzipped) the three extended `.jsonl`.

```bash
cd benchmarking/mmau_pro/results
gzip -k run16_hibudget/*/**.jsonl 2>/dev/null || true
tar czf run16_results.tgz run16_hibudget --exclude='*.jsonl'
```

## Provenance notes

- Parquets: HF dataset `macabdul9/MMAU_Pro_Testmini` @ `81eb01fb…` (has both testmini and the
  full-test split). Audio: HF dataset `gamma-lab-umd/MMAU-Pro` `data.zip`, sha256-verified.
  The 1,099 testmini clips are a strict subset of `data.zip` — `fetch_data.sh` copies them
  into place (`prep_testmini_audio.py --check` verifies).
- The two `_le30s` parquets in `run16/data/` are derived artifacts (all clips ≤30 s —
  Qwen2-Audio's encoder window); `build_le30s_subset.py` documents and can regenerate them.
- Original experiment write-ups: `benchmarking/mmau_pro/RESULTS.md` §§13–16; operational
  lessons: `SETUP_GUIDE.md`.
