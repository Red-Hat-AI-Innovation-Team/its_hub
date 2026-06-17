# MMAU‑Pro × Inference‑Time Scaling (PF/EPF with generator self‑certainty) — Run Log & Results

**Question.** On a *fixed* Qwen2.5‑Omni‑7B, can Particle Filtering (PF) / Entropic PF (EPF) —
with particle weights taken from the **generator's own token log‑probabilities / entropy
("self‑certainty"), no external reward model** — improve MMAU‑Pro multiple‑choice accuracy
(the RL‑free alternative to GRPO)?

**TL;DR (full 957‑MCQ result).** With a good chain‑of‑thought prompt (#4 plan‑and‑solve),
PF@4 / EPF@4 reach **0.576 / 0.575 vs a 0.534 baseline — about +4 pp, borderline‑significant
(z≈1.8, p≈0.07)**. With the other prompt (#2) ITS is neutral. The gain is **small,
prompt‑dependent, concentrated in hard perceptual categories** (spatial audio, sound, speech),
and **does not scale with budget** — consistent with self‑certainty being a weak (fluency)
reward signal. There is **no GRPO result on MMAU‑Pro** to compare against (see caveats).

Last updated: 2026‑06‑10.

---

## 1. Setup

- **Model / serving:** Qwen2.5‑Omni‑7B (thinker, text‑out) via vLLM 0.22.1, OpenAI‑compatible, port 8100.
  - Required env/flags discovered: **`VLLM_USE_FLASHINFER_SAMPLER=0`** (flashinfer JIT sampler fails to build on Blackwell sm_120); installed server‑side audio decoders **`librosa soundfile av resampy`** (`av`/PyAV is what vLLM falls back to; without it → "install vllm[audio]" / "Invalid audio file"); `--allowed-local-media-path <data>` + `--limit-mm-per-prompt '{"audio":3}'`; `HF_HOME` on the 3.4 TB volume.
  - Audio input: both `input_audio` (base64) and `audio_url` (`file://`, "local‑path") work; **local‑path used for the 957 run** (avoids re‑sending ~26 MB base64 each PF step).
- **Particle weight:** generator self‑certainty (now the only weight source in its_hub — the former `weight_source=` kwarg is gone). PF uses signal `mean_logprob`, style `logit`; EPF uses signal `entropy`. Reasoning is chunked into PF/EPF steps on `\n\n` (`StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=6)`), `max_tokens_per_step=300`.
- **Data:** `mmau_pro_testmini` — 957 MCQ items (non‑empty `choices`; `answer` is the choice *text*). `le30s` split = 411 MCQ (≤30 s audio) used for dev. Scoring: lettered choices A–K, parse `Answer: <letter>`, normalized/fuzzy match to gold (5 items ungradeable → excluded).
- **Arms:** `baseline` = budget 1 (single trajectory, no resampling); `pf` / `epf` = budgets as noted.
- **Harness:** `benchmarking/mmau_pro/` (`run_mmau.py`, `prompt.py`, `scoring.py`, `loader.py`, `audio.py`, `cot_compare.py`, `phase0_gate.py`, `ab_causality.py`). Raw outputs in `benchmarking/mmau_pro/results/`. Client‑side deps: `pip install its_hub[benchmark]` (click, pandas, pyarrow on top of the `[lm]` extra).

## 2. Validation (does the pipeline actually work on audio?)

- **Phase‑0 gate 1 — logprobs WITH audio input: PASS.** vLLM returns generated‑token `logprobs`/`top_logprobs` even with audio in the prompt → self‑certainty is viable, no fallback needed.
- **Phase‑0 gate 2 — `continue_final_message` WITH an audio user turn: PASS.** The model continues a prefilled partial assistant turn → the step‑by‑step audio carry works.
- **A/B causality (does the model *hear* it?): PASS.** 15 items, greedy, audio‑present vs audio‑removed: **0.533 vs 0.400**, and **6/15 answers changed** — multi‑audio "which clip contains X" items are correct only with audio. The audio reaches the model.
- **Pipeline smoke (n=8):** baseline 0.500, PF@4 0.625, EPF@4 0.375 — 0 errors. (Also surfaced that the *original* terse prompt made the model answer in one letter; fixed by the CoT prompts below.)

## 3. Run 1 — 8‑prompt CoT comparison (which prompt makes the model reason, chunkably?)

20 single‑audio le30s items, one greedy generation each. `reasoned` = fraction with ≥15 words of reasoning; `chunks` = mean `\n\n`‑segments (PF step granularity).

| # | prompt | acc | reasoned | avg_words | avg_chunks |
|---|--------|----:|---------:|----------:|-----------:|
| 1 | assistant‑prefill CoT | 0.350 | 1.00 | 112 | 4.1 |
| 2 | zero‑shot CoT (user trigger) | 0.400 | 0.95 | 110 | 4.0 |
| 3 | few‑shot CoT | 0.300 | 0.90 | 28 | **1.0** |
| **4** | **plan‑and‑solve** | **0.500** | 1.00 | **138** | **5.2** |
| 5 | least‑to‑most | 0.450 | 1.00 | 42 | 2.1 |
| 6 | describe‑then‑reason (audio) | 0.350 | 1.00 | 93 | 3.0 |
| 7 | format‑forcing (## Step) | 0.400 | 1.00 | 104 | 3.6 |
| 8 | anti‑shortcut (≥3 steps) | 0.250 | 1.00 | 65 | 2.1 |

**Takeaways:** 7/8 prompts produce real, chunkable reasoning (so PF/EPF have steps to resample). Few‑shot (#3) collapsed to terse answers (1 chunk) → dropped. Carried forward: **#4 (best), #2, #7**.

### Run 1b — n=40 stratified re‑run + new prompt #9

Re‑run of the screen on **40 single‑audio le30s items stratified across `category`** (even round‑robin coverage — this is a *fresh* sample, **not** the smallest‑40 above, so it is not a direct extension of the n=20) plus a **9th prompt**: evidence‑grounded steps ending in `Final Answer: \boxed{LETTER}` (supplied verbatim; mapped as a system‑prompt CoT). Same metric defs, greedy, one generation/item. (Raw CSV/log removed as redundant after the full 957 bake-off — see Run 5.)

Category mix (40): `sound 5, voice_chat 5, spatial_audio 5, speech 5, open 5, music 5, sound_speech 5, sound_music 4, music_speech 1`.

| # | prompt | acc | reasoned | avg_words | avg_chunks |
|---|--------|----:|---------:|----------:|-----------:|
| 1 | assistant‑prefill CoT | 0.550 | 1.00 | 78.5 | 2.9 |
| 2 | zero‑shot CoT (user trigger) | 0.525 | 1.00 | 97.4 | 3.4 |
| 3 | few‑shot CoT | 0.475 | 0.88 | 26.3 | **1.0** |
| **4** | **plan‑and‑solve** | 0.575 | 1.00 | 101.6 | **4.3** |
| 5 | least‑to‑most | **0.625** | 0.90 | 31.9 | 2.0 |
| 6 | describe‑then‑reason (audio) | 0.600 | 1.00 | 70.5 | 2.5 |
| 7 | format‑forcing (## Step) | 0.550 | 1.00 | 92.7 | 3.6 |
| 8 | anti‑shortcut (≥3 steps) | 0.550 | 1.00 | 60.6 | 2.0 |
| 9 | evidence‑grounded (boxed) | 0.500 | 1.00 | 96.2 | **1.9** |

**Takeaways:**
- **Accuracy is within noise.** With ~40 gradeable items/cell the 95% CI is ≈±15pp, so the 0.475–0.625 spread is *not* significant — this is a chunkability screen, not a ranking. The auto‑"BEST" (least‑to‑most, 0.625) is statistically indistinguishable from #4/#6.
- **#4 plan‑and‑solve stays the most chunkable** (4.3 `\n\n` chunks, 100% reasoned) — still the right pick for PF/EPF (which need many resamplable steps), consistent with the downstream runs below using it.
- **#3 few‑shot** again collapses to 1 chunk (terse) — confirmed unsuitable.
- **New #9 (boxed)** reasons plenty (96 words, 100% reasoned) but chunks **lowest (1.9)** — exactly as flagged: its single‑newline `Step N:` format doesn't split on `\n\n`, so *as written* it is the least PF‑friendly prompt despite heavy reasoning. Its `\boxed{}` answers parsed 40/40 (the scorer was extended with a top‑priority `\boxed{LETTER}` rule for this).

## 4. Run 2 — CoT × ITS ablation (n=30, budget 4)

30 smallest single‑audio le30s items; `{2,4,7} × {baseline, PF@4, EPF@4}`; 270 runs, 0 errors.

| prompt | baseline | PF@4 | EPF@4 |
|--------|---------:|-----:|------:|
| #2 zero‑shot | 0.367 | 0.400 | **0.500** |
| #4 plan‑and‑solve | **0.567** | 0.533 | 0.500 |
| #7 format‑forcing | 0.500 | 0.333 | 0.367 |

**Takeaways:** mixed/noisy at n=30. #7 ITS *hurts* → dropped. Carried forward **#2 and #4** to a budget sweep. (Note: #4's 0.567 baseline here did **not** hold at larger n — see below.)

## 5. Run 3 — 150‑item budget sweep (the scaling test)

150 single‑audio le30s items; `{2,4} × {baseline@1, PF/EPF @4,8,16}`; 2100 runs, 0 errors; n≈148/cell (95% CI ≈ ±8 pp).

| prompt | base | PF@4 | PF@8 | PF@16 | EPF@4 | EPF@8 | EPF@16 |
|--------|----:|----:|----:|-----:|-----:|-----:|------:|
| #2 zero‑shot | 0.426 | 0.432 | 0.459 | 0.439 | **0.480** | 0.459 | 0.432 |
| #4 plan‑and‑solve | 0.412 | **0.527** | 0.439 | 0.453 | 0.419 | 0.439 | 0.419 |

**Takeaways:** **no clean budget scaling** — accuracy does not rise 4→8→16 (often peaks at 4, then drops). Apparent best cells (#4 PF@4 +11.5 pp, #2 EPF@4 +5.4 pp) are at/near the ±8 pp noise band and don't replicate at higher budget. This motivated the full‑957 run to resolve signal vs noise.

## 6. Run 4 — FULL 957 MCQ (the decisive result)

All 957 MCQ; `{2,4} × {baseline@1, PF@4, EPF@4}`; **5742 runs, 0 errors**; n=952/cell (5 ungradeable); 95% CI ≈ ±3.3 pp; `local-path` audio; item‑concurrency 12.

| prompt | base@1 | PF@4 | EPF@4 | PF Δ | EPF Δ |
|--------|-------:|-----:|------:|-----:|------:|
| #2 zero‑shot | 0.561 | 0.548 | 0.558 | −1.3 | −0.3 |
| **#4 plan‑and‑solve** | 0.534 | **0.576** | **0.575** | **+4.2** | **+4.1** |

Significance (2‑proportion z vs baseline): **#4 PF z=+1.84, EPF z=+1.80 (p≈0.07, borderline)**; #2 PF z=−0.55, EPF z=−0.14 (n.s.).

**Regression to the mean:** the big n=148 effects shrank at n=952 — #4 PF@4 **+11.5 → +4.2 pp**, #2 EPF@4 **+5.4 → −0.3 pp** — confirming the small‑n bumps were largely noise. A real but modest ~+4 pp residual remains on #4.

### #4 plan‑and‑solve — accuracy by category (base / PF@4 / EPF@4)

| category | n | base | PF@4 | EPF@4 |
|----------|--:|----:|----:|-----:|
| spatial_audio (hardest) | 69 | 0.203 | **0.319** | 0.246 |
| sound | 199 | 0.412 | 0.447 | **0.487** |
| speech | 171 | 0.556 | **0.632** | 0.602 |
| voice_chat | 64 | 0.484 | **0.562** | 0.531 |
| open | 105 | 0.886 | 0.933 | **0.943** |
| multi | 89 | 0.438 | 0.393 | 0.472 |
| music (already strong) | 220 | 0.655 | 0.659 | 0.641 |
| sound_speech | 18 | 0.111 | 0.389 | 0.389 |

By `length_type` (#4): EPF helps the **long / ultra‑long** clips (long 0.500→0.586, ultra‑long 0.500→0.625); PF helps short/medium more.

**Interpretation:** ITS helps where the model is *uncertain about perception* (spatial audio, environmental sound, speech) and adds nothing where it's already strong (music). This is the most interesting, interpretable part of the result.

## 7. Run 5 — full 957, 9-prompt greedy CoT bake-off (pick the best base prompt)

**Goal:** with prompts #1–#9 fixed, run **every prompt over all 957 MCQ, one greedy generation each (t=0)** — no PF/EPF — to choose the best base prompt. (Different decoding setup from Run 4: a single full greedy completion per item, *not* PF's step-chunked `base@1`, so absolute numbers differ slightly.) All-audio incl. **89 multi-audio**, `local-path`, concurrency 24 → **8,613 generations, 0 errors, ~58 min**. n=952 gradeable/prompt (5 ungradeable); **846** after also excluding the **106 single-choice** trivial items — `acc (excl-1)` is the fair comparator (single-choice items are correct for every prompt).

| # | prompt | acc (all, 952) | **acc (excl-1, 846)** | avg_words | avg_chunks |
|---|--------|----:|----:|----:|----:|
| 1 | assistant‑prefill CoT | 0.593 | 0.563 | 87 | 3.0 |
| 2 | zero‑shot CoT | 0.550 | 0.517 | 85 | 2.8 |
| 3 | few‑shot CoT | 0.538 | 0.502 | 29 | 1.0 |
| 4 | plan‑and‑solve | 0.559 | 0.513 | 128 | 5.0 |
| **5** | **least‑to‑most** | **0.613** | **0.571** | 42 | 2.3 |
| 6 | describe‑then‑reason | 0.582 | 0.537 | 96 | 2.7 |
| 7 | format‑forcing (## Step) | 0.597 | 0.564 | 92 | 3.5 |
| 8 | anti‑shortcut (≥3 steps) | 0.589 | 0.544 | 69 | 2.1 |
| 9 | evidence‑grounded (boxed) | 0.550 | 0.505 | 106 | 2.0 |

**Best prompt — a 3-way top tier (paired McNemar on the 846 common items):**
- **#5 least‑to‑most (0.571)** leads but is **statistically tied** with **#7 format‑forcing (0.564, p=0.67)** and **#1 assistant‑prefill (0.563, p=0.63)**; #8 anti‑shortcut (0.544) borderline (p=0.10).
- #5 is **significantly better** than the verbose/boxed/few‑shot prompts: vs #4 plan‑and‑solve p=0.001, vs #9 boxed p<0.001, vs #2 p=0.001, vs #3 p<0.001, vs #6 p=0.03.

**Key finding — chunkability ≠ accuracy.** The prompts that were *best for PF chunking* (#4 plan‑and‑solve 5.0 chunks, #9 boxed) are the **worst for greedy accuracy** (0.513, 0.505). The most accurate prompt, **#5 least‑to‑most, is concise** (42 words, 2.3 chunks). A longer, more‑segmented trace did not buy correctness here — it slightly hurt it. (So the prompt that's best to *run greedily* and the prompt that's best to *feed PF* may differ — #4 still chunks best for ITS; #5/#7/#1 score best as one-shot prompts.)

**By category (acc_all):** least‑to‑most wins on sound (0.55) & music (0.70); assistant‑prefill/format‑forcing win on speech (0.66–0.67); `open` ≈ 0.9 across the board; **spatial_audio is hard for all (0.20–0.32)**. Full matrix in `cot957_html/index.html`.

**Artifacts:** `results/cot957.log` (score tables), `results/cot957.csv` (8,613 per-response rows), `results/cot957.jsonl` (resumable raw), `results/cot957_html/index.html` (summary + accuracy matrix + per-category side-by-side pages).

## 8. Honest verdict

- **Best ITS number: 0.576 (#4 PF@4), ≈ +4 pp over our own baseline, no RL used** — borderline‑significant (p≈0.07), prompt‑dependent, category‑localized, and **non‑scaling** with budget.
- It is **not** a "ITS matches GRPO for free" result. The gain is small and self‑certainty behaves like a weak, fluency‑based reward — it can't reliably steer resampling toward *correct* trajectories.
- Leaderboard context (reported numbers, full MMAU‑Pro test set, NOT our measurement): base Qwen2.5‑Omni ≈ 52%, AF3 51.7%, Gemini‑2.5‑Flash 59.2%, human 77.9%. Our testmini‑MCQ base (0.53–0.56) is in range; best ITS (0.576) is a few points above base.

### Caveats / what NOT to claim
- **No GRPO/RL result exists on MMAU‑Pro.** Published GRPO numbers (R1‑AQA, SARI, **Omni‑R1: 65.9 → 71.3**) are on the **original MMAU**, a *different and easier* benchmark — not comparable to these MMAU‑Pro numbers. The "+4 pp" here is **over our own no‑ITS baseline**, *not* over any RL result.
- n=952 → ±3.3 pp CI; the #4 effect is borderline (p≈0.07), not conclusively significant.
- `baseline` = PF at budget 1 (single self‑certainty trajectory), i.e. the same CoT prompt with no resampling.

## 9. Files (`benchmarking/mmau_pro/results/`)

| file | what |
|------|------|
| `mmau_957_results.jsonl` | full 957 run, 5742 rows (the headline result) |
| `mmau_957.log` | 957 run log |
| `mmau_150_sweep.jsonl` | 150‑item budget sweep, 2100 rows |
| `mmau_ablation.jsonl` | n=30 CoT×ITS ablation, 270 rows |
| `cot_compare.log` | 8‑prompt CoT comparison output (n=20, smallest) |
| `cot957.jsonl` | **Run 5**: resumable raw rows, full 957 × 9 greedy bake-off |
| `cot957.csv` | **Run 5**: 8,613 per-(prompt,item) responses + parsed fields |
| `cot957.log` | **Run 5**: overall + excl-trivial + by-category score tables |
| `cot957_html/` | **Run 5**: `index.html` (summary + matrix) + per-category side-by-side pages |
| `cot957_all.html` | **Run 5**: single-file side-by-side of all 957 questions × 9 prompts (~6 MB; open in a browser) |
| `mmau_smoke.jsonl` | initial 8‑item pipeline smoke |

Each row: `{unique_id, method, arm, budget, category, length_type, correct, latency_s, error, content}`.

## 10. Reproduce

```bash
# serve (Blackwell)
HF_HOME=$BV/hf_cache CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/exx/miniconda3/envs/epf/bin/vllm serve Qwen/Qwen2.5-Omni-7B \
  --served-model-name qwen-omni --port 8100 --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.9 \
  --allowed-local-media-path /home/exx/inference-time-scaling/mmau_pro_testmini \
  --limit-mm-per-prompt '{"audio":3}'

# full 957 (resumable, concurrent)
conda run -n epf python -m benchmarking.mmau_pro.run_mmau \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --subset full \
  --prompt-methods 2,4 --arms baseline,pf,epf --budgets 4 \
  --audio-mode local-path --item-concurrency 12 \
  --output benchmarking/mmau_pro/results/mmau_957_results.jsonl

# Run 5: full 957 x 9 greedy bake-off (resumable) + paginated HTML
uv run python -m benchmarking.mmau_pro.cot_compare \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --subset full --select all --audio-mode local-path --concurrency 24 \
  --jsonl benchmarking/mmau_pro/results/cot957.jsonl \
  --csv   benchmarking/mmau_pro/results/cot957.csv \
  --log   benchmarking/mmau_pro/results/cot957.log
uv run python -m benchmarking.mmau_pro.make_report \
  --in benchmarking/mmau_pro/results/cot957.csv \
  --out-dir benchmarking/mmau_pro/results/cot957_html --paginate category
```

## 11. Next lever (not yet run)

The category pattern (ITS helps where perception is uncertain) motivates the **answer‑choice‑confidence reward**: weight particles by `P(correct choice | audio, question, reasoning‑so‑far)` — a *correctness* proxy rather than fluency — which is the signal most likely to be significant and to actually scale with budget. The harness/`weight_source` is already pluggable for this.
