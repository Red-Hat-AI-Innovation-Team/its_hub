# SETUP_GUIDE — Running PF/EPF audio-benchmark experiments with this codebase

Audience: a fresh Claude/engineer session on a new machine (or new GPUs) that has this repo and
needs to run **inference-time-scaling experiments (PF/EPF with self-certainty weights) on audio
LMs** — new models (Audio Flamingo, other Omni variants, …) and/or new benchmarks (MMAR, full
MMAU-Pro, …). Everything below was verified in practice on the original 2× RTX PRO 6000 Blackwell
box (July 2026). Read `benchmarking/mmau_pro/RESULTS.md` for the science; this file is the *how*.

---

## 1. What this repo is (30 seconds)

- `its_hub/` — a minimal library implementing **Particle Filtering (PF)** and **Entropic Particle
  Filtering (EPF)** for inference-time scaling. Particle weights come from the *generator's own
  token logprobs* ("self-certainty": `mean_logprob` or `entropy` signal) — **no reward model**.
  `budget` = number of particles. Deep-dive docs in `documentation/` (chapters 07/08 = the math).
- `benchmarking/mmau_pro/` — a complete audio-MCQ harness (loader, prompts, scoring, runners,
  probes, reports) built for MMAU-Pro × Qwen2.5-Omni-7B. It is the template for any new
  (model × benchmark) pair.
- Findings so far (Runs 1–15, `RESULTS.md`): PF/EPF@4 gives a borderline +4 pp on the best prompt;
  selected accuracy saturates by budget ~8–16 while oracle coverage keeps climbing; resampling
  actively culls correct minorities; no self-generated signal fixes selection. New experiments
  should keep this framing in mind (the interesting axes are *selection* and *coverage*, not raw N).

## 2. Environment setup

```bash
# The canonical env is conda `epf` (Python 3.11). Verified stack: torch 2.11.0 (cu130), vLLM 0.22.1.
# Exact pinned package set: ./requirements-epf.txt (pip freeze of the reference env, 2026-07-08)
conda create -n epf python=3.11 -y
<env>/bin/pip install -r requirements-epf.txt        # pinned deps (torch/vllm are CUDA-13.0 builds)
<env>/bin/pip install -e ".[dev,benchmark]"          # its_hub itself, from the repo root

# Offline dev/tests can instead use uv:
uv sync --extra dev
uv run pytest tests/ -q --ignore=tests/e2e     # 104 tests, ~4 s, no GPU needed
```

**Packaged handoff:** for the scripted end-to-end path (env + pinned data/models + smoke +
single-command driver, built for the Run-16 b64/128 extension) see
`benchmarking/mmau_pro/run16/README.md`. ONE env serves all models there — the env is about
the vLLM/torch stack, not the checkpoint.

**PATH GOTCHA (will bite you):** `conda activate epf` does NOT win the PATH on the original box (a
uv-managed CPython shadows it). Always call the absolute interpreter
`/home/exx/miniconda3/envs/epf/bin/python` or use `conda run -n epf …`. On a new machine, verify
`which python` actually resolves to the env you installed into before debugging anything else.

- `tests/e2e/` needs the optional `math_verify` dep — exclude it unless doing math e2e.
- Server-side audio decode deps (install into the *serving* env): `librosa soundfile av resampy`
  (PyAV is what vLLM falls back to; without it you get "install vllm[audio]" / "Invalid audio file").
- Keep `HF_HOME` on a large volume (models + audio caches are tens of GB).

## 3. Serving a model with vLLM (the template)

```bash
HF_HOME=<big-volume>/hf_cache CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <env>/bin/vllm serve <HF-model-id> \
  --served-model-name <short-name> --port 810<gpu> --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.85 \
  --allowed-local-media-path <parent-dir-of-ALL-audio-roots> \
  --limit-mm-per-prompt '{"audio":3}'
```

Flag-by-flag, with the lessons attached:

| flag / env | why |
|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | **Mandatory on Blackwell (sm_120)** — flashinfer JIT sampler fails to build. Harmless elsewhere. |
| `--gpu-memory-utilization 0.85` | **NOT 0.9.** The audio encoder's attention is O(n²) in clip length and allocates ~4–5 GB transiently per long clip *outside* vLLM's budgeted memory. At 0.9 we OOM-killed an engine mid-run (see §10). 0.85 on a 96 GB card leaves ~14 GB for spikes. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces fragmentation for those spikes |
| `--allowed-local-media-path` | must cover every audio root you'll reference via `file://` URLs — use the common parent (e.g. `/home/exx/inference-time-scaling`), not one dataset dir |
| `--limit-mm-per-prompt '{"audio":3}'` | MMAU-Pro has up to 3 audios/item; raise for other benchmarks if needed (caps clip COUNT, not duration) |
| `--max-model-len 32768` | fits the worst MMAU-Pro item (600 s clip ≈ 15k audio tokens) with room; re-audit per benchmark (§5) |
| one server per GPU, ports 8100/8101/… | probes round-robin items client-side across `--endpoints`; there is no tensor-parallel setup here |

Health: `curl -s http://localhost:8100/v1/models`. Load takes O(10 s–minutes). Before serving,
check `nvidia-smi` for orphaned allocations squatting on the card (we had a 2 GB ghost with no
process attached) — they reduce your headroom and you cannot reclaim them without a reboot.

**Run servers detached** (`nohup setsid … > server.log 2>&1 &`) so they survive the controlling
session, and keep a watchdog polling `/v1/models` — when an engine dies (OOM), the API server
returns 500s briefly and then the port goes connection-refused; every in-flight item errors.

## 4. Model gates — run these BEFORE any experiment on a new model

The PF/EPF machinery has two hard server-side requirements. `phase0_gate.py` tests both:

```bash
<env>/bin/python -m benchmarking.mmau_pro.phase0_gate \
  --endpoint http://localhost:8100/v1 --model-name <name> --data-root <mmau_pro_testmini>
```

1. **Gate 1 — token logprobs WITH audio input.** Self-certainty weights need per-token `logprobs`
   (and `top_logprobs` for the `entropy` signal) in chat completions even when the prompt contains
   audio. Some endpoints return none with multimodal input → PF/EPF weights silently degrade to
   neutral (the library warns once and weighs 0.0).
2. **Gate 2 — `continue_final_message` WITH audio.** Step-wise generation appends the partial
   assistant turn and asks the model to continue it (vLLM `extra_body` flags
   `continue_final_message=true, add_generation_prompt=false`). OpenAI-typed endpoints do not
   support this — the LM client detects endpoint type by substring `"openai" in endpoint`
   (`its_hub/core/lms/openai_lm.py`), so serve via vLLM.

Additional per-model checks before burning GPU-hours:

3. **A/B causality** (`ab_causality.py`): greedy accuracy with audio vs audio-stripped on ~15
   items. If accuracy/answers don't change, your audio isn't reaching the model (wrong media path,
   silent decode failure).
4. **Chunkability screen** (`cot_compare.py` on 20–40 items): PF/EPF resample at `\n\n`
   boundaries. If a model answers tersely (avg_chunks ≈ 1), there are no steps to resample and
   PF degenerates to best-of-N-ish. Check `avg_chunks ≥ ~3` for at least one prompt, and that
   answers parse (the scorer reads `Answer: <letter>`, `\boxed{LETTER}`, bare uppercase letters,
   then fuzzy choice-text). Prompt note from Run 5: the best *greedy* prompt and the best
   *PF-feeding* (most chunkable) prompt are usually different.
5. **Audio-length capacity.** Qwen2.5-Omni encodes ~25 tokens/sec of audio with no hard clip cap
   (long clips chunk-processed) → 32k context ≈ 20 min total audio. **Other models differ
   sharply**: Qwen2-Audio truncates clips at 30 s; Audio Flamingo variants have their own window
   limits. For any new model, (a) find its audio-token rate and clip cap, (b) run the benchmark's
   N LONGEST items through a 1-token generation preflight and check `usage.prompt_tokens` +
   absence of HTTP 400s, (c) confirm `prompt_tokens + max_steps×max_tokens_per_step ≤ max-model-len`.
   Do NOT trust a smoke test on the first-N items — parquet order is biased toward short audio.

## 5. Data layout and loader contract

Current datasets on the original box (adapt paths on a new machine):

| what | where |
|---|---|
| MMAU-Pro parquets | `/home/exx/inference-time-scaling/mmau_pro_testmini/data/{testmini,testmini_le30s,test}-00000-of-00001.parquet` |
| testmini audio (957 items) | `/home/exx/inference-time-scaling/mmau_pro_testmini/data/` — 1,099 audio files (1,088 wav + 11 mp3; do NOT copy with a `*.wav` glob), same dir as the parquets |
| full-test audio (5,787 files, 53 GB) | `/home/exx/inference-time-scaling/mmau_pro_audio/data/` |

Loader: `benchmarking/mmau_pro/loader.py` → `load_mmau_mcq(data_root, subset, audio_root=None)`.
`subset` ∈ `full` (testmini, 957 MCQ) / `le30s` (411) / `test` (**the full set: 5,305 rows →
5,090 MCQ**). `audio_root` resolves relative audio paths when they live outside the parquet root
(the `test` subset needs `--audio-root …/mmau_pro_audio`).

**Dataset quirks you must know (MMAU-Pro test set):**
- **24/5,090 ungradeable**: gold `answer` text doesn't fuzzy-match any choice (threshold 0.85) →
  `answer_index=None` → excluded from accuracy denominators (rows still run, `correct=null`).
- **497 single-choice items** (~10%): one "choice" that echoes the answer — every method gets them
  right. Report accuracy both ways (all-gradeable and excl-1) as Run 5 did; the 957-run headline
  numbers include their 106 equivalents.
- `length_type` has THREE spellings of long-tail labels: `ultra-long`, `ultra_long`, and NaN.
  Normalize before any by-length analysis (old JSONLs contain `"nan"` strings).
- testmini's 957 ids ⊂ test's 5,090 ids → paired old-vs-new comparisons are a JSONL filter away.
- Max 11 choices (scoring `LETTERS = "ABCDEFGHIJK"` covers exactly 11); max 3 audios/item;
  durations p50 = 50 s, max = 600 s.

**Adding a new benchmark (e.g. MMAR ~1,000 single-choice, 4 options):** create
`benchmarking/<bench>/` mirroring `mmau_pro/`. The only real contract is producing `MCQRecord`s
(`unique_id, question, choices, answer, audio_paths (absolute), category, length_type,
answer_index`). If the benchmark is lettered-MCQ, reuse `scoring.py` unchanged; reuse `prompt.py`
builders (they're benchmark-agnostic: audio parts + "Question/Options" text). Precompute
`answer_index` with `match_answer_index` at load time.

## 6. Algorithms & canonical experiment configuration

Both algorithms step-generate with `StepGeneration(step_token="\n\n", stop_token="Answer:",
max_steps=6)` and `max_tokens_per_step=300`, sampling temp **0.8** (StepGeneration's default).
Weights: signal `mean_logprob` (mean token logprob) or `entropy` (−mean top-20 entropy), style
`logit` (log-odds transform). Resampling every step; final answer = argmax final weight.

**⚠ Two "EPF" configs exist in this repo — do not mix them silently:**

| context | resampling | temperature gate | signal |
|---|---|---|---|
| `run_mmau.py` epf arm (Run 4 headline) | systematic | library defaults `ess_threshold=0.5, early_phase=0.5` | `entropy` |
| all probes (Runs 6–10 + the full-set grid) | systematic | probe defaults `--ess-threshold 0.6 --early-phase 0.7` | both, explicit |

When replicating or comparing, state which config you used in your results doc. The canonical
**grid config** (what `epf_div_bootstrap` reflects) is the probe one: temp 0.8, 0.6/0.7,
systematic, style logit, prompts {4,5,7,9}, signals {mean_logprob, entropy}, budgets {1,8,16,32}.

## 7. Runners — which script for which experiment

All are `python -m benchmarking.mmau_pro.<script> --help`-documented. GPU scripts are resumable
via append-only JSONL; analysis scripts are offline.

| script | what it measures | multi-GPU |
|---|---|---|
| `run_mmau.py` | accuracy: prompts × {baseline@1, PF@b, EPF@b} | single `--endpoint` only |
| `cot_compare.py` | greedy (t=0) prompt bake-off + chunkability | `--concurrency` on one endpoint |
| `diversity_probe.py` | **the EPF grid**: prompts × signals × budgets with SMC metrics (selected/oracle/majority acc, distinct ratio, consensus, final ESS) | `--endpoints` comma list, items round-robined |
| `divsource_probe.py` | EPF vs INDEP (resampling off) + per-step ESS traces | `--endpoints` |
| `rerank_probe.py` | terminal answer-confidence re-rank of finished swarms | `--endpoints` |
| `epf_bootstrap.py` | bootstrap error bars (std_subsample + SE_full) from a grid CSV → HTML | offline |
| `epf_report.py`, `make_report.py`, `plot_*.py` | HTML/plot reports from CSVs | offline |

Reference invocation — the full-MMAU-Pro EPF grid exactly as run in July 2026 (2 GPUs):

```bash
<env>/bin/python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --subset test --audio-root /home/exx/inference-time-scaling/mmau_pro_audio \
  --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets <B> \
  --select all --limit 6000 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
  --max-inflight <24 if B==1 else 64> \
  --jsonl benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.jsonl \
  --csv   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv \
  --log   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.log
```

**Concurrency rule (learned the hard way):** the probe's per-endpoint item concurrency is
`max(1, max_inflight // budget)`. At budget 1 that means `max_inflight` DISTINCT items encoding audio
simultaneously per GPU — this is what OOM'd a replica at 64. Cap `--max-inflight 24` for budget-1
stages; 64 is safe for budgets ≥ 8 (≤ 8 items/endpoint; particles of one item share the cached
audio prefix). At budgets ≥ 64 only one item runs per endpoint, but the per-item particle fan-out
is unthrottled (`max_concurrency=-1` in the LM client) — up to `budget` concurrent requests per
step; vLLM queues this fine.

**Trajectory-segmentation flags (added July 2026 — Runs 13/14):** `diversity_probe` accepts
`--step-token` (per-prompt resampling boundary; e.g. `$'\n'` for single-newline `Step N:` formats
like P9 — raise `--max-steps` accordingly), `--stop-regex` (stop only on letter-final answers, not
any `Answer:` substring — P5's sub-answers false-trigger the default), and `--stop-on-repeat`
(kills digit-incremented repeat-step loops). All default-off. **Empirical warning from Runs 13/14:
enabling real multi-step resampling consistently LOWERED oracle coverage 8–12 pp with no selection
gain** — resampling frequency is inversely related to coverage on this task family.

**Run big grids budget-staged** (all cells at b=1, then 8, 16, 32) in a detached script:
early stages give a complete low-budget picture in ~2 h and calibrate the cost model before the
expensive tail. Chain the stages with a `for B in 1 8 16 32` loop over the same `--jsonl`.

## 8. Output conventions — where results go and how they're structured

**Everything lands in `benchmarking/<bench>/results/`** (for MMAU-Pro:
`benchmarking/mmau_pro/results/`), organized **one folder per run** — see
`results/README.md` for the index. Never scatter outputs elsewhere. Naming scheme: one
experiment = one `runNN_<slug>/` folder + one basename, three files —

```
results/runNN_<slug>/<experiment>.jsonl  # append-only raw rows — THE source of truth, resumable
results/runNN_<slug>/<experiment>.csv    # per-item metrics table, regenerated from the deduped JSONL each run
results/runNN_<slug>/<experiment>.log    # human-readable trend/score report, regenerated each run
results/runNN_<slug>/<experiment>*.html  # one-file reports (bootstrap, heatmaps) built from the CSV
results/plots/                           # cross-run comparison figures only
```

Existing folders: `run01_cot_screen` … `run11_epf_full5090` (+ `plots/`, `smoke/`), holding
basenames `mmau_957_results`, `mmau_150_sweep`, `mmau_ablation`, `mmau_smoke`, `cot957`,
`epf_div`, `run6_full`, `rerank`, `divsource`, `divsource_full`, `divsource_t1`, `epf_full5090`.
**When starting a new experiment, create the next `runNN_<slug>/` folder** and pass explicit
`--jsonl/--csv/--log` paths into it (script defaults point at the historical folders).

**Resumability model (identical across runners):** each JSONL row carries its cell key —
`(unique_id, method, signal, budget)` for probes, `(unique_id, method, arm, budget)` for
`run_mmau`. On start, rows without `error` are marked done and skipped; errored rows are retried;
final CSV/report dedupe keeps the latest row per key. Two consequences you should exploit:
1. **Crash recovery is free** — just rerun the same command.
2. **Seeding**: if a new run's config is bit-identical to an old one on a subset of items (e.g.
   testmini ⊂ full test set), `cp old.jsonl new.jsonl` before starting — the overlap is skipped.
   Note the provenance in RESULTS.md when you do this (see Run 10's precedent).

**Documentation duty:** every run gets a numbered section in `benchmarking/<bench>/RESULTS.md`
following the existing format — *why → config (exact CLI) → n/error-count → results table →
takeaways/verdict → artifact filenames*, plus a row in the file table (§18 there) and the exact
reproduce command (§19 there). A run that isn't written up there effectively doesn't exist.

**Bootstrap reporting** (what the stakeholder wants to see): run `epf_bootstrap.py --in
results/<grid>.csv --out results/<grid>_bootstrap.html`. It reports, per cell × metric,
**point ± std100 (± SE_full)** — std100 = std of accuracy over 100-question subsamples without
replacement ("how noisy is a small eval"), SE_full = bootstrap-with-replacement SE of the reported
number (→ `sqrt(p(1−p)/n)`), 10k resamples, seed 0, with a closed-form cross-check printed. Labels
auto-adapt to n. Don't overwrite an old report HTML if the old one is still referenced — write a
new file alongside (e.g. keep `epf_div_bootstrap.html` = the 957 run; the full-set grid gets its
own `epf_full5090_bootstrap.html` once that run finishes).

## 9. Cost model & run management

Anchors from the original box (Qwen2.5-Omni-7B bf16, enforce-eager, 2× 96 GB Blackwell,
all-audio MMAU-Pro, temp 0.8, 6×300-token steps) — **cell-level s/item across 2 GPUs**:

| budget | s/item (2 GPUs) | one 4,133-item cell |
|---|---|---|
| 1 (conc 24/endpoint) | 0.22–0.28 | ~16 min |
| 8 (conc 8/endpoint) | 0.5–1.0 | 35–70 min |
| 16 / 32 | scales ≈ linearly in budget from b8 | hours |

Full 8-cell × {1,8,16,32} grid on 5,090 items ≈ **40–50 GPU-pair-hours**. Do NOT use the per-row
`latency_s` in JSONLs for cost modeling — it includes queue wait (timer starts before the
semaphore); use the "done in Xs (Y s/item)" cell lines from the runner logs.

Operational pattern that works:
- stage script detached: `nohup setsid bash run_stages.sh > stages.log 2>&1 &`
- watch `stages.log` for `done in .* errors` lines and stage boundaries; alert on `FAILED|Traceback`
- a 60 s watchdog curl on every endpoint (engine deaths look like: 500s, then connection refused)
- after each cell, error count should be ~0. A sudden block of hundreds of identical
  `ClientConnectorError` rows = an endpoint died, not a data problem: restart the server, rerun
  the same command, resume sweeps up the errored keys.
- known batching behavior (`its_hub/core/lms/step_generation.py`): one slow particle
  head-of-line-blocks its batched step — long-audio items make whole batches slow, worse at high
  budgets. (Related but distinct TODO at the top of that file: a *dead* particle — e.g. max-tokens
  — can stop the whole generation.)

## 10. Anomaly log (things that actually happened — check these first when debugging)

1. **Audio-encoder OOM (2026-07-06).** GPU1 engine died mid-run: SDPA in
   `modeling_qwen2_5_omni.py` audio tower tried to allocate 4.29 GiB with 3.56 GiB free
   (`gpu-memory-utilization 0.9` + 2 GB orphaned allocation + 64 concurrent distinct items at
   budget 1). Every request to that port errored (~20% of the cell) until restart. Fix that
   worked: 0.85 utilization + `expandable_segments:True` + `--max-inflight 24` at b1. Zero errors
   across 40k+ rows since.
2. **Orphaned GPU memory**: ~2 GB shown used on a GPU with no compute process. Survives process
   kills; plan headroom around it.
3. **`pkill -f` footgun**: `pkill -f "vllm serve"` matches your own shell's command line and kills
   it. Kill by PID (`pgrep -af '[v]llm serve'` first), or use bracketed patterns.
4. **Smoke-test bias**: first-N parquet items are short-audio; a clean 16-item smoke proved
   nothing about long-audio behavior. Always also preflight the longest items.
5. **RESULTS.md drift**: some stale bits exist (an old "Last updated" line; Run 4's category
   table omits small categories and slightly disagrees with loader counts). The numbers in the
   run tables themselves were re-verified against raw artifacts (July 2026) and are correct.
6. **Endpoint-type detection is a substring check** (`"openai" in endpoint`) — a URL containing
   "openai" anywhere gets OpenAI semantics (no continue_final_message). Avoid such hostnames.
7. **Entropy signal needs `top_logprobs`** (auto-set to 20). It's a top-k-truncated entropy —
   fine for ranking, not an absolute entropy.
8. **Missing logprobs are silently neutral**: if an endpoint stops returning logprobs, particles
   get weight 0.0 and PF becomes uniform resampling. One warning is logged — grep for
   "no token logprobs" in long runs.

## 11. End-to-end checklist for a NEW (model × benchmark) experiment

1. Env: absolute-path python, `pip install -e ".[dev,benchmark]"`, unit tests pass.
2. Data: parquet + audio on disk; write/adapt loader → verify record count, ungradeable count,
   `all(os.path.exists)` on audio, category mix.
3. Serve: template from §3 (0.85 util!), media path = common parent, one server per GPU, detached.
4. Gates: `phase0_gate` per endpoint (logprobs+audio, continue+audio) → must PASS both.
5. Causality + chunkability: `ab_causality` (~15 items), `cot_compare` (~40 items) → pick prompts;
   check answer parsing against `scoring.py`.
6. Context audit: longest items × (audio tokens/sec) + generation budget ≤ max-model-len;
   1-token preflight on the longest 30–60 items.
7. Smoke: one cheap cell (~16 items) into a THROWAWAY jsonl → 0 errors, sane metrics.
8. Cost estimate from §9 anchors → decide grid size; get sign-off if > ~10 GPU-hours.
9. Run budget-staged, detached, monitored (b1 with low inflight!); check error counts per cell.
10. Analysis: CSV → `epf_bootstrap` HTML (+ plots); write the numbered RESULTS.md section with
    exact config, artifacts, and the reproduce command.

## 12. Sanity examples — reproduce these before trusting a new setup

Three tiers, cheapest first. Run them in order; each tier's expected outputs were produced and
re-verified on the reference stack (2026-07-08).

### E1 — offline, exact on ANY machine (no GPU, no data server)

```bash
uv run pytest tests/ -q --ignore=tests/e2e
# expected: 104 passed in ~4 s

<env>/bin/python -c "
from benchmarking.mmau_pro.loader import load_mmau_mcq
recs = load_mmau_mcq('/home/exx/inference-time-scaling/mmau_pro_testmini', subset='test',
                     audio_root='/home/exx/inference-time-scaling/mmau_pro_audio')
print(len(recs), sum(1 for r in recs if r.answer_index is None))"
# expected: 5090 24        (full test set; 957 for subset='full', 411 for 'le30s')

<env>/bin/python -c "
from benchmarking.mmau_pro.scoring import extract_letter, predicted_index, match_answer_index
print(extract_letter('The answer is clear.\n\nAnswer: B', 4))                       # 1  (letter B)
print(extract_letter('Step 1: ...\nFinal Answer: \\\\boxed{C}', 5))                  # 2  (letter C)
print(predicted_index('I think it is the dog barking', ['cat','dog barking','dog'])) # 1  (longest match wins)
print(match_answer_index('The Dog barking!', ['cat','dog barking','dog']))           # 1  (normalized gold match)"
# expected: 1 2 1 1 — these are pure functions; any deviation means a modified scoring.py
```

### E2 — served model responds correctly (per endpoint)

```bash
<env>/bin/python -m benchmarking.mmau_pro.phase0_gate \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini
# expected on ANY correctly-served audio model:
#   GATE 1 (logprobs WITH audio): PASS
#   GATE 2 (continue_final_message WITH audio): PASS
#   RESULT: {"gate1_logprobs": true, "gate2_continue": true}
# On the reference stack (Qwen2.5-Omni-7B bf16, vLLM 0.22.1, greedy) the Gate-1 line is exactly
#   first token: 'Lon' logprob=-10.634275436401367 top_logprobs=20
# (identical across both GPUs/replicas here; the exact value is stack-specific — a different
#  vLLM/torch/GPU can shift it, PASS/FAIL is what must hold everywhere.)
```

### E3 — end-to-end: 2 fixed items, greedy, must match these letters

Runs all 9 prompts on two specific testmini items (t=0, one short single-audio each, ~1 min):

```bash
<env>/bin/python -m benchmarking.mmau_pro.cot_compare \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --subset full \
  --ids 2035bce6-a746-4a82-82c1-d61da27cb533,69c911db-5532-4677-b28e-77eb231e6d24 \
  --audio-mode local-path --csv /tmp/sanity.csv
```

Expected predicted letters on the reference stack (gold = **C** for both items):

| prompt method | `2035bce6…` (expect ✓) | `69c911db…` (expect ✗) |
|---|---|---|
| 1 assistant-prefill | **D** (wrong) | D |
| 2–9 (all others) | **C** (correct) | D |

i.e. the per-method score table prints `acc=0.500` for methods 2–9 and `0.000` for method 1.
Verified bitwise-identical across two back-to-back runs, and method 4's letters match the Run 5
CSV recorded three weeks (and several server restarts) earlier. Greedy decoding on the SAME
vLLM/torch/GPU stack is reproducible; a different stack can legitimately flip a near-tie token —
if letters differ, first re-check E1/E2 and your serve flags before suspecting the harness.

---
*Maintained alongside the July-2026 full-MMAU-Pro grid run. When something here disagrees with
the code, trust the code and fix this file.*
