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

## 7. Honest verdict

- **Best ITS number: 0.576 (#4 PF@4), ≈ +4 pp over our own baseline, no RL used** — borderline‑significant (p≈0.07), prompt‑dependent, category‑localized, and **non‑scaling** with budget.
- It is **not** a "ITS matches GRPO for free" result. The gain is small and self‑certainty behaves like a weak, fluency‑based reward — it can't reliably steer resampling toward *correct* trajectories.
- Leaderboard context (reported numbers, full MMAU‑Pro test set, NOT our measurement): base Qwen2.5‑Omni ≈ 52%, AF3 51.7%, Gemini‑2.5‑Flash 59.2%, human 77.9%. Our testmini‑MCQ base (0.53–0.56) is in range; best ITS (0.576) is a few points above base.

### Caveats / what NOT to claim
- **No GRPO/RL result exists on MMAU‑Pro.** Published GRPO numbers (R1‑AQA, SARI, **Omni‑R1: 65.9 → 71.3**) are on the **original MMAU**, a *different and easier* benchmark — not comparable to these MMAU‑Pro numbers. The "+4 pp" here is **over our own no‑ITS baseline**, *not* over any RL result.
- n=952 → ±3.3 pp CI; the #4 effect is borderline (p≈0.07), not conclusively significant.
- `baseline` = PF at budget 1 (single self‑certainty trajectory), i.e. the same CoT prompt with no resampling.

## 8. Files (`benchmarking/mmau_pro/results/`)

| file | what |
|------|------|
| `mmau_957_results.jsonl` | full 957 run, 5742 rows (the headline result) |
| `mmau_957.log` | 957 run log |
| `mmau_150_sweep.jsonl` | 150‑item budget sweep, 2100 rows |
| `mmau_ablation.jsonl` | n=30 CoT×ITS ablation, 270 rows |
| `cot_compare.log` | 8‑prompt CoT comparison output |
| `mmau_smoke.jsonl` | initial 8‑item pipeline smoke |

Each row: `{unique_id, method, arm, budget, category, length_type, correct, latency_s, error, content}`.

## 9. Reproduce

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
```

## 10. Next lever (not yet run)

The category pattern (ITS helps where perception is uncertain) motivates the **answer‑choice‑confidence reward**: weight particles by `P(correct choice | audio, question, reasoning‑so‑far)` — a *correctness* proxy rather than fluency — which is the signal most likely to be significant and to actually scale with budget. The harness/`weight_source` is already pluggable for this.
