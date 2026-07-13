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

**TL;DR 2 (the July-2026 full-scale sequel, Runs 11–14).** At full MMAU-Pro scale (5,090 MCQ,
SE ±0.7 pp) and across two model sizes (Omni-7B & 3B): **selected accuracy saturates by ~8
particles (~0.59 on 7B / ~0.53 on 3B); oracle coverage climbs to 0.83–0.91 and is size-invariant;
the oracle−selected gap is the whole story and it *widens* as models shrink** (calibration degrades
with size, exploration doesn't). Three independent manipulations of resampling frequency (off →
Run 8; more via stop-rule fix → Run 14; more via matched delimiter → Run 13) all agree:
**every added resampling round lowers oracle 8–12 pp and never improves selection.** Best-of-N +
an external selector dominates PF/EPF on this task family at any step granularity (§§13–15).

**TL;DR 3 (cross-family check, Run 15).** Same grid on **Qwen2-Audio-7B-Instruct** — a different
model family (Whisper-style encoder, hard 30 s clip window → run on the 2,190-item ≤30 s subset of
the full test set; P4+P9, the only prompts that survive its format screen). The shape replicates:
selected saturates by b8 (~0.44–0.47, +2 pp over b1), oracle climbs to 0.64–0.67, gap 0.18–0.23 at
b32, signals indistinguishable on selection. **New findings:** the oracle ceiling is
*family*-dependent (0.64–0.67 vs 0.83–0.91 on Omni's matching cells — coverage was size-invariant
within the Omni family but tracks
base competence across families), and **plain majority vote beats self-certainty selection on P9 at
every budget ≥ 8** — the selector-is-the-wall verdict extends to a third model (§16).

Last updated: 2026‑07‑10.

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

**Artifacts:** `results/run05_cot957/cot957.log` (score tables), `results/run05_cot957/cot957.csv` (8,613 per-response rows), `results/run05_cot957/cot957.jsonl` (resumable raw), `results/run05_cot957/cot957_html/index.html` (summary + accuracy matrix + per-category side-by-side pages).

## 8. Run 6 — EPF diversity sweep (is scaling worth it, or is the weight the bottleneck?)

**Why:** Run 5 ranked prompts by *greedy* accuracy, but EPF needs **exploration**. We ran **Entropic
Particle Filtering** on 100 stratified single-audio items × 4 prompts {**#4** plan-and-solve, **#5**
least-to-most, **#7** format-forcing, **#9** evidence-grounded/boxed} × 2 self-certainty weights
{`mean_logprob`, `entropy`} × budgets {1, 8, 16, 32}, instrumented with new SMC metrics
([diversity_probe.py](benchmarking/mmau_pro/diversity_probe.py)). Config: temp **0.8**, **systematic**
resampling, **ess_threshold 0.6**, **early_phase 0.7**. **3,200 EPF runs, 0 errors, ~35 min on both GPUs**
(GPU0:8100 + GPU1:8101, items round-robined). n=99 gradeable (1 ungradeable).

**The result is the same in all 8 (prompt × signal) cells.** Representative numbers (`mean_logprob`):

*Selected accuracy (what EPF returns) — barely scales with N:*

| prompt | b1 | b8 | b16 | b32 |
|--------|---:|---:|---:|----:|
| #4 plan-and-solve | 0.374 | 0.505 | **0.586** | 0.525 |
| #5 least-to-most | 0.556 | 0.586 | 0.566 | 0.566 |
| #7 format-forcing | 0.455 | 0.505 | 0.556 | 0.576 |
| #9 evidence-grounded | 0.475 | 0.475 | 0.535 | 0.576 |

*Oracle accuracy (is the correct answer in ANY particle?) — climbs strongly with N:*

| prompt | b1 | b8 | b16 | b32 | gap (oracle−selected) @b32 |
|--------|---:|---:|---:|----:|----:|
| #4 plan-and-solve | 0.374 | 0.626 | 0.717 | 0.848 | **+0.32** |
| #5 least-to-most | 0.556 | 0.747 | 0.747 | 0.828 | +0.26 |
| #7 format-forcing | 0.455 | 0.667 | 0.717 | 0.798 | +0.22 |
| #9 evidence-grounded | 0.475 | 0.778 | 0.838 | **0.929** | **+0.35** |

**Same run, `entropy` signal** — selected accuracy (b1→b32):

| prompt | b1 | b8 | b16 | b32 |
|--------|---:|---:|---:|----:|
| #4 plan-and-solve | 0.434 | 0.525 | **0.556** | 0.434 |
| #5 least-to-most | 0.495 | 0.566 | 0.525 | 0.535 |
| #7 format-forcing | 0.475 | 0.535 | 0.576 | **0.616** |
| #9 evidence-grounded | 0.455 | 0.545 | 0.556 | 0.576 |

`entropy` oracle accuracy (b1→b32) + gap:

| prompt | b1 | b8 | b16 | b32 | gap @b32 |
|--------|---:|---:|---:|----:|----:|
| #4 plan-and-solve | 0.434 | 0.677 | 0.758 | 0.798 | +0.36 |
| #5 least-to-most | 0.495 | 0.828 | 0.778 | 0.828 | +0.29 |
| #7 format-forcing | 0.475 | 0.677 | 0.717 | 0.869 | +0.25 |
| #9 evidence-grounded | 0.455 | 0.788 | 0.899 | **0.949** | **+0.37** |

**Both signals behave identically** — oracle climbs to 0.80–0.95, selected stalls at ~0.55, gap +0.22 to
+0.37. `entropy` edges `mean_logprob` on a couple of cells (#7 selected 0.616 @b32; #9 oracle 0.949) but
not meaningfully — **neither weight converts oracle coverage into selected accuracy.**

**Findings:**
1. **Oracle coverage climbs hard with budget** (≈0.45 → 0.85–0.95) — more particles *do* surface the
   correct answer. Exploration is real and scales.
2. **Selected accuracy does NOT scale** — it plateaus at ~0.52–0.62 from b8→b32 (sometimes *drops*, e.g.
   #4 peaks at b16 then falls). **Plurality-vote (majority) is no better** (~0.50–0.60).
3. **Huge oracle − selected gap (+0.22 to +0.37 @b32):** the right answer is in the swarm 80–95% of the
   time but EPF returns the wrong particle. On **~40%** of items the correct answer is even a swarm
   *minority* the self-certainty weight votes against (P9 entropy b32: 37/100 present-but-not-selected).
4. **Diversity is real but the distinct-answer *ratio* falls with N** (0.97→0.06) — partly mechanical
   (ratio = #unique/N), since the absolute #distinct answers actually *grows* (≈1.4 → 2.5 among 32). The
   swarm converges its plurality onto a confident, often-wrong answer while the tail increasingly holds
   the correct one.

**Verdict → the self-certainty WEIGHT is the bottleneck, not the particle count.** This is exactly the
"weak fluency reward, doesn't scale" prediction. Scaling N raises *oracle* but not *selected* because
self-certainty (both `mean_logprob` and `entropy`) cannot identify the correct particle — it favors a
confident-but-wrong plurality. **No amount of extra particles fixes selected accuracy; only a better
weight does** → the answer-choice-confidence reward (tested in §9).

**Higher temperature?** *Not the priority.* At temp 0.8 the right answer is already present 80–95% of the
time (oracle ≫ selected); a higher temp would mostly push oracle *higher* while selected stays stuck,
widening the gap — reinforcing (not fixing) the weight diagnosis. Worth a small probe later, but the
decisive lever is the weight, not exploration.

**Prompt note (vindicates not picking by greedy acc):** **#9 evidence-grounded** — the *worst* greedy
prompt (Run 5: 0.505) — has the **richest swarm** (highest oracle, 0.93–0.95 @b32; most distinct answers).
With a competent weight, #9 + EPF has the most headroom (~0.95 oracle ceiling). #4 plan-and-solve is
similar. So the best EPF base prompt ≠ the best greedy prompt.

## 9. Run 7 — terminal answer-confidence re-rank (can a better selector recover the oracle gap?)

**Why:** Run 6's gap is a *selection* failure (right answer in the swarm, EPF picks wrong). This tests the
cheapest fix — re-rank the finished EPF swarm by **answer confidence** instead of self-certainty. Three
scorers: **L-audio** (letter read-out, chat, *with* audio), **L-text** (letter read-out, text-only),
**O-text** (option likelihood via `/v1/completions` echo, text-only, length-normalized) — each with two
selection rules (**argmax-particle**, **conf-vote**), vs baselines **epf** (status quo) / **majority** /
**oracle**. 100 stratified single-audio items × prompts {4,5,7,9} × budgets {8,16,32}; EPF mean_logprob
swarm. **1,200 runs, 0 errors.** n=99/cell → 95% CI ≈ ±10 pp.
(Server constraint: chat returns no `prompt_logprobs` here, so option-likelihood must use the text-only
`/completions` echo path — hence the L-text control, to compare letter-vs-option fairly.)

**Headline @budget 32 — nothing recovers the gap:**

| prompt | oracle | epf (status quo) | majority | best re-rank |
|--------|---:|---:|---:|---:|
| #4 plan-and-solve | 0.828 | 0.505 | 0.566 | 0.556 (L-audio vote) |
| #5 least-to-most | 0.818 | 0.586 | 0.616 | 0.616 (L-audio vote) |
| #7 format-forcing | 0.798 | 0.566 | 0.576 | 0.576 (L-text vote) |
| #9 evidence-grounded | 0.889 | 0.535 | 0.525 | 0.535 (L-audio vote) |

Every selector — self-certainty, majority, and all six confidence re-ranks — clusters at **0.50–0.62**,
while oracle is **0.80–0.89**. The unrecovered gap (+0.23 to +0.35) is essentially untouched; all deltas
vs epf are within the ±10 pp noise band.

**Why it fails — it reshuffles, it doesn't improve.** Decomposing the most-aggressive rule at b32
(*recovered* = oracle-right-but-EPF-wrong items it rescues; *broken* = EPF-right items it newly loses):

| prompt | rule | recovered | broken | net acc vs epf |
|--------|------|---:|---:|---:|
| #9 evidence-grounded | L-audio argmax | 14/35 | 16 | −0.020 |
| #4 plan-and-solve | O-text argmax | 13/32 | 12 | +0.010 |

The re-rank *does* find ~10–14 correct minority particles — but discards an equal number EPF had right.
**One-for-one trade → net zero.** Answer-confidence is no better calibrated than self-certainty at telling
the model's own right answers from its wrong ones.

**Cross-cuts:**
- **L-audio ≈ L-text** — re-attending the audio in the probe gives no consistent edge → the audio-less
  constraint on option-likelihood was *not* the limiting factor (the comparison is fair).
- **O-text ≤ L-text**, and **`otext_vote` is the worst rule** (net negative, e.g. P5/P9 −0.05 to −0.10) →
  full-option-text likelihood did not beat single-letter; the surface-form de-bias hypothesis didn't pan out.
- **majority** is quietly the best baseline (small positive, ≈ best re-rank) — reaffirming Run 6.

**Verdict → the ceiling is calibration, and NO self-generated signal fixes it.** Fluency self-certainty
(Runs 4/6), answer-letter confidence, and option-text likelihood all fail identically: the model is
confidently wrong on the items it gets wrong, so any self-derived weight reshuffles rather than improves.
Closing the oracle gap requires an **independent/external verifier** (a different judge model, or a trained
verifier/PRM) — not a reweighting of the generator's own confidence. This is a stronger, more decisive
negative than Run 6 (it rules out the "just pick a better self-signal" hope).

## 10. Run 8 — where does the (lack of) diversity come from: generation or resampling?

**Why:** even at budget 32 the EPF swarm reaches ~80% consensus (~4/5 particles agree) — is that because
the particles were *born similar* (low generation diversity) or because resampling *collapsed* an initially
diverse swarm? We ran two arms on the same 100 items, holding generation fixed (temp 0.8, mean_logprob) and
toggling only resampling: **EPF** (systematic resampling ON, as deployed) vs **INDEP** (resampling OFF → N
independent step-chunked trajectories = the generator's intrinsic diversity). Prompts {4,5,7,9} × budgets
{8,16,32}, **2,400 runs, 0 errors**, plus a per-step ESS curve (logged free inside the loop).

**Result @budget 32 (mean across the 4 prompts):**

| arm | distinct | consensus | **oracle** | selected |
|-----|---:|---:|---:|---:|
| EPF (resample **ON**) | 0.065 | 0.813 | **0.838** | 0.558 |
| INDEP (resample **OFF**) | 0.089 | 0.697 | **0.957** | 0.558 |

Three findings, in order of importance:

1. **Resampling actively *culls the correct answer* (the big one).** Turning resampling off lifts **oracle
   +0.12 (0.838 → 0.957)** — independent N=32 sampling contains the right answer **~96%** of the time, but
   EPF's resampling concentrates on high-self-certainty particles and discards the (lower-fluency, often
   correct) minority *before the end*, dropping oracle to ~0.84. The penalty is **worst at low budget**
   (P4: +0.22 @b8, +0.19 @b16, +0.14 @b32) — fewer particles, more gets culled.
2. **Selection is still the sole binding bottleneck.** `selected` is **0.558 in *both* arms** despite INDEP
   oracle 0.957 — preserving diversity doesn't help if the selector can't pick the right particle (Run 7).
3. **"Born similar" is real but not the limiter.** Even INDEP is low-diversity in absolute terms (~0.089
   distinct ≈ **~3 distinct answers among 32**, ~0.70 consensus) — the model is fairly deterministic at temp
   0.8. But that's still *enough to contain the correct answer 96% of the time*, so low distinct-count is not
   what's capping accuracy. (Raising temperature would add diversity, but coverage isn't the problem.)

**Per-step ESS explains the mechanism.** EPF's weights peak hard mid-trajectory — ESS dips to ~0.25–0.35
around step 2 (e.g. P7 b32: `0.83 → 0.24 → 0.33 → 0.40 → 0.46`) — i.e. strong concentration pressure that
culls the minority — then *recovers* near the end as duplicates equalize, so the **final** ESS looks benign
(~0.7–0.9, matching Run 6). INDEP shows the same early dip but **stays low** (`0.84 → 0.19 → 0.14 → 0.30 …`),
revealing the underlying self-certainty weights are genuinely peaked; resampling is what acts on that peak to
remove particles. So Run 6's healthy-looking final ESS was masking a sharp mid-trajectory cull.

**Verdict:** the swarm is *born* low-diversity but still covers the answer (~96% under independent sampling);
**EPF's resampling is net-harmful here** — it lowers the oracle ceiling by ~12 pp without improving selected
accuracy. Within the RL-free framing, the implied pipeline is **plain best-of-N sampling (no resampling) +
an external selector**, not particle filtering. The binding constraint is unchanged and reconfirmed: the
**selector**, not particle count, diversity, or temperature.

### Full‑957 replication (all 957 MCQ, temp 0.8)

Re‑ran the identical EPF‑vs‑INDEP comparison on the **entire 957‑MCQ set** (incl. 89 multi‑audio),
**22,968 runs, 0 errors**, n=952/cell (5 ungradeable), 95% CI ≈ ±3 pp — so this is the decisive‑scale
version of the n=100 result above. Mean across the 4 prompts:

| budget | arm | distinct | consensus | oracle | selected |
|---|---|---:|---:|---:|---:|
| 8 | EPF | 0.184 | 0.872 | 0.731 | 0.581 |
| 8 | INDEP | 0.265 | 0.726 | **0.880** | 0.592 |
| 16 | EPF | 0.107 | 0.854 | 0.803 | 0.597 |
| 16 | INDEP | 0.155 | 0.715 | **0.936** | 0.587 |
| 32 | EPF | 0.061 | 0.833 | 0.850 | 0.595 |
| 32 | INDEP | 0.087 | 0.705 | **0.966** | 0.589 |

**Both findings hold at full scale:** (1) turning resampling **off lifts oracle +0.149 @b8 / +0.133 @b16 /
+0.116 @b32** — EPF resampling culls the correct minority, worst at low budget; (2) **selected accuracy is
unchanged (~0.59 in both arms, every budget)** despite INDEP oracle reaching **0.966** — the selector is the
sole bottleneck. The ±3 pp CIs make the oracle gap unambiguous (it's ~12–15 pp). Raw: `divsource_full.*`.

## 11. Run 9 — does a higher temperature (1.0) add diversity?

**Why:** Run 8 showed the swarm is *born* low-diversity (even independent N=32 → ~3 distinct answers). Is
the lever simply **temperature**? Same divsource setup (EPF vs INDEP arms, per-step ESS), 100 stratified
items, prompts {4,5,7,9} × budgets {8,16,32}, **only the sampling temperature changed: 0.8 → 1.0** (all else
identical). **2,400 runs, 0 errors.**

**Result — temperature barely moves anything (mean across 4 prompts, `0.8 → 1.0`):**

| budget | arm | distinct | consensus | oracle | selected |
|---|---|---|---|---|---|
| 32 | EPF | 0.065 → 0.065 | 0.81 → 0.82 | 0.838 → 0.813 | 0.558 → 0.540 |
| 32 | INDEP | 0.089 → 0.096 | 0.70 → 0.66 | 0.957 → 0.980 | 0.558 → 0.525 |
| 8 | EPF | 0.192 → 0.191 | 0.86 → 0.85 | 0.692 → 0.702 | 0.563 → 0.503 |
| 8 | INDEP | 0.273 → 0.293 | 0.73 → 0.69 | 0.848 → 0.866 | 0.556 → 0.558 |

(Budget 16 is the same story.) Raising the temperature **does not add meaningful diversity** — distinct/
consensus move by ~0.01–0.04, within noise — and **selected accuracy is unchanged** (~0.55 at both temps).
INDEP oracle nudges up ~+0.02 (slightly more spread surfaces the answer a touch more often), but the swarm
was already covering the answer ~96%, so it's moot.

**Verdict:** the model is **intrinsically low-entropy on MCQ answers** — "born similar" persists even at temp
1.0, so temperature is **not** the lever. Combined with Runs 7–8, the bottleneck is now triply confirmed to
be the **selector**, not particle count, not diversity, and not temperature. (Higher temps than 1.0 would
trade coherence for spread; not worth it given oracle is already ~0.96 and selection is the wall.)

## 12. Run 10 — Run 6's EPF budget sweep on the FULL 957 (4 prompts × 2 signals × {1,8,16,32})

**Why:** Run 6 (the EPF diversity/scaling sweep) was only n=100. This re-runs the **entire grid on all 957
MCQ**: prompts {4,5,7,9} × signals {mean_logprob, entropy} × budgets {1,8,16,32}, EPF only. To avoid
recompute, the **mean_logprob {8,16,32}** cells were carried over from Run 8's full‑957 EPF arm (identical
config, same `compute_metrics`); only **mean_logprob{1} + entropy{1,8,16,32}** were newly run. **30,624 rows,
0 errors**, n=952 gradeable/cell (±3 pp CI).

**Mean across the 4 prompts:**

| signal | budget | **selected** | oracle | majority | distinct | consensus |
|--------|---:|---:|---:|---:|---:|---:|
| mean_logprob | 1 | 0.557 | 0.557 | 0.557 | (1 particle) | — |
| mean_logprob | 8 | 0.581 | 0.731 | 0.589 | 0.184 | 0.872 |
| mean_logprob | 16 | **0.597** | 0.803 | 0.614 | 0.107 | 0.854 |
| mean_logprob | 32 | 0.595 | 0.850 | 0.614 | 0.061 | 0.833 |
| entropy | 1 | 0.567 | 0.567 | 0.567 | (1 particle) | — |
| entropy | 8 | 0.593 | 0.756 | 0.605 | 0.195 | 0.853 |
| entropy | 16 | 0.582 | 0.814 | 0.605 | 0.113 | 0.835 |
| entropy | 32 | 0.586 | 0.866 | 0.607 | 0.064 | 0.823 |

**Findings (now at decisive scale):**
1. **Selected accuracy barely scales with budget.** From the single‑trajectory anchor (b1 ≈ 0.557/0.567) it
   gains only **~+0.03–0.04** and **saturates by budget 8–16** (mean_logprob peaks 0.597 @b16; entropy 0.593
   @b8), then is flat/dips at 32. With ±3 pp CIs the ITS lift over b1 is marginal‑to‑borderline — **no real
   scaling.** (Matches Run 6's noisy n=100 read, now tight.)
2. **Oracle climbs steeply** (0.56 → 0.85–0.87 @b32) — exploration keeps surfacing the answer — but selected
   doesn't follow → the oracle−selected gap *widens* with budget. **majority ≈ selected** (~0.59–0.61), both
   far below oracle.
3. **The two weight signals are equivalent.** entropy is a hair better on a couple cells (selected@b8,
   oracle@b32) but within noise; neither escapes the plateau. So Run 6's "weak fluency reward" conclusion is
   not specific to mean_logprob — entropy behaves the same.

**Verdict:** confirms Run 6 at full scale — **EPF buys ~+0.03–0.04 selected accuracy over a single sample and
plateaus by ~8 particles; neither self‑certainty signal scales; the oracle gap only grows.** Consistent with
Runs 7–9: the selector is the wall. Raw: `run6_full.*` (mean_logprob{8,16,32} provenance = Run 8).

## 13. Run 11 — the Run 6/10 EPF grid on the FULL MMAU-Pro test set (5,090 MCQ)

**Why:** every prior run used the 957-MCQ testmini. This scales the canonical EPF grid — 4 prompts
{4,5,7,9} × 2 signals {mean_logprob, entropy} × budgets {1,8,16,32} — to the **full MMAU-Pro test
set**: `test` parquet, 5,305 rows → **5,090 MCQ** (incl. 497 single-choice trivial items, same
convention as the 957 which had 106; 24 ungradeable → n=5,066/cell, **95% CI ≈ ±1.4 pp, bootstrap
SE ≈ ±0.7 pp**). Audio from `mmau_pro_audio/` (5,787 files, max clip 600 s; 430 two-audio + 26
three-audio items). Config identical to Run 10 (temp 0.8, systematic, ess 0.6/0.7, style logit).
The 957 overlap was **seeded from `run6_full.jsonl`** (testmini ⊂ test; same mechanism as Run 10's
Run-8 seeding) — the 957-slice of this run reproduces Run 10's table exactly, by construction.
**132,256 fresh EPF runs (162,880 rows total), 0 errors, ~50 h on both GPUs**, budget-staged
(b1 → b8 → b16 → b32, resumable JSONL).

**Serving notes (required at this scale):** `--gpu-memory-utilization 0.85` (not 0.9) +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `--max-inflight 24` for the budget-1 stage
— at b1 the probe otherwise encodes 64 distinct audios per GPU simultaneously and the O(n²) audio
encoder OOM-killed an engine (one cell had to be re-swept; final data has 0 errors). Details in
`SETUP_GUIDE.md` §3/§10.

**Mean across the 4 prompts** (bootstrap in `epf_full5090_bootstrap.html`; SE ≈ ±0.007/cell):

| signal | budget | **selected** | oracle | majority |
|--------|---:|---:|---:|---:|
| mean_logprob | 1 | 0.557 | 0.557 | 0.557 |
| mean_logprob | 8 | 0.585 | 0.732 | 0.596 |
| mean_logprob | 16 | **0.590** | 0.792 | 0.605 |
| mean_logprob | 32 | **0.590** | 0.845 | 0.613 |
| entropy | 1 | 0.558 | 0.558 | 0.558 |
| entropy | 8 | 0.587 | 0.754 | 0.601 |
| entropy | 16 | 0.583 | 0.818 | 0.607 |
| entropy | 32 | 0.586 | 0.865 | 0.608 |

**Findings (the Run 6/10 story at decisive scale, now with sub-pp error bars):**
1. **Selected accuracy saturates by budget 8.** +2.8–3.3 pp over b1, then flat: every b8→b16→b32
   step is within ±1 pp per cell. The apparent b8→b16 bumps in Run 10 (e.g. P7 mean_logprob
   +4.7 pp) collapse to ≈0 at full scale (−0.1 pp) — they were noise, as suspected.
2. **Oracle keeps climbing** — 0.56 → 0.845/0.865 @b32 (mean); **P9 evidence-grounded reaches
   0.914 with BOTH signals** — the right answer is in the swarm 91% of the time while P9's
   selected accuracy is the *lowest* (0.565–0.577). The oracle−selected gap at b32 is +0.26 on
   average and **+0.34 for P9**.
3. **majority ≈ selected + ~0.02** at every budget — plurality voting doesn't close the gap either.
4. **The two signals are equivalent** (differences within ~±0.5 pp mean; entropy slightly higher
   oracle, mean_logprob slightly higher selected — both within noise).

**Bootstrap headline** (`epf_full5090_bootstrap.html`): a 100-question eval wobbles ±0.049 on these
cells; the reported full-set numbers are precise to ±0.0069 (~7× tighter) — and 2.3× tighter than
the 957-run's ±0.016.

**Verdict:** unchanged and now essentially beyond statistical doubt at this benchmark's full scale —
**EPF's selected accuracy does not scale with particle count past ~8; exploration (oracle) scales
beautifully; no self-certainty signal converts it; the selector is the wall.** This is the
strongest-n version of the honest verdict below; the next lever remains an external/independent
selector (§20).

## 14. Run 12 — model-size ablation: Qwen2.5-Omni-**3B**, full 5,090 MCQ (P4, P5)

**Why:** first non-7B model — does the EPF story (saturation, oracle climb, selection wall) depend
on model size? Same serving stack (both GPUs, gates PASS ×2), same probe config as Runs 10/11.

**Prompt screen first (important):** a 40-item chunkability screen showed **P7 and P9 mechanically
break on 3B** — P7 gets one-word answers (3% multi-chunk; the 3B ignores the `## Step` format) and
P9 emits its single-newline `Step N:` template with zero blank lines (1.0 chunks) → nothing for EPF
to resample. Grid therefore ran **P4 + P5** (the two chunk-compliant prompts; both have full 7B
reference grids). A 100-item follow-up probe (`run12_omni3b_5090/chunkprobe_3b_greedy100.csv`, same
items scored on both models via `run05_cot957/cot957.csv`) quantified it: prompt-format compliance
degrades with model size (P7 97%→3% multi-step; P9 both models 99% single-newline steps — 7B's 2.0
chunks came from one incidental blank line before "Final Answer"). NOT a max-tokens issue (0–4%
of responses near the cap). The probe also exposed a 3B-specific **repetition pathology in P5**:
degenerate sub-question loops that re-ask the same 3–4 questions with incremented counters
("Sub-question 8/12/16: <identical text>", up to 42 chunks/560 words before the token cap) —
motivating the loop guard used in Run 14.

**Grid:** P4,P5 × {mean_logprob, entropy} × budgets {1,8,16,32} × 5,090 MCQ, temp 0.8, ess 0.6/0.7,
systematic/logit — **81,440 rows, 0 errors, ~21 h on both GPUs** (no seeding; new model).
n=5,066 gradeable/cell, SE ≈ ±0.7 pp. Raw: `run12_omni3b_5090/epf3b_5090.*`.

| cell | selected | oracle | majority |  | 7B ref (sel/orc @same cell) |
|---|---:|---:|---:|---|---|
| P4 ml b1 / b8 / b16 / b32 | .524 / .534 / .546 / .538 | .524 / .698 / .753 / .831 | — / .549 / .557 / .564 | | .549/.549 → .584/.826 |
| P4 ent b1 / b8 / b16 / b32 | .512 / .523 / .531 / .540 | .512 / .720 / .786 / .836 | — / .555 / .555 / .559 | | .551/.551 → .575/.835 |
| P5 ml b1 / b8 / b16 / b32 | .448 / .528 / .529 / .537 | .448 / .759 / .812 / .845 | — / .555 / .563 / .562 | | .568/.568 → .603/.815 |
| P5 ent b1 / b8 / b16 / b32 | .449 / .486 / .478 / .478 | .449 / .791 / .847 / .875 | — / .549 / .548 / .551 | | .567/.567 → .588/.866 |

**Findings:**
1. **The trends replicate**: selected saturates by b8 (then ≤1 pp per doubling); oracle climbs
   steeply and keeps climbing; majority ≈ selected + ~0.02.
2. **Oracle is size-invariant, selection is not.** 3B oracle ≈ 7B oracle at every budget (±1–3 pp;
   0.83–0.88 @b32) while 3B selected sits 2–12 pp lower → the **oracle−selected gap widens to
   +0.29…+0.40** (7B: +0.18…+0.28). Exploration doesn't shrink with model size; calibration does.
   The selection wall is worse for smaller models.
3. **First signal non-equivalence:** P5×entropy underperforms P5×mean_logprob by 5–6 pp
   (0.478 vs 0.537 @b32) — plausibly the entropy weight favoring the 3B's low-entropy
   repetitive/terse steps. P4 signals stay equivalent.
4. **Stop-token caveat (affects P5 both models, discovered here):** the harness stop rule
   (`"Answer:" in step`) fires on P5's *sub-answers* — live-verified: P5 particles stop after 1–2
   steps of 6. On 7B this is benign (4% early, 97% of reasoning kept — sub-answers rarely say
   "Answer:"); on **3B it cuts 75% of trajectories at ~chunk 1 (42% of reasoning kept)**. So
   Run 11/12 P5 cells measure a short-horizon EPF variant. Runs 13/14 (below) test the matched
   fixes: per-prompt step tokens and a letter-final-answer stop rule + repeat-step loop guard.

**Verdict:** the selection bottleneck is not a 7B quirk — it's *worse* at 3B with identical
exploration, strengthening the calibration-ceiling interpretation. Combined report (both models,
toggleable plots + bootstrap tables): `run11_epf_full5090/epf_full5090_bootstrap.html`.

## 15. Runs 13–14 — step-boundary ablations (does giving EPF *real* steps help?)

Run 12's format findings left two "bugs" in how trajectories are segmented: P9's steps are
invisible to the `\n\n` delimiter (≈1 resample total), and the `"Answer:"` substring stop cuts P5
short (1–2 steps). Both accidentally made EPF behave like best-of-N. These runs **fix each one and
measure what true step-wise EPF is worth.** New `StepGeneration` options (default-off, unit-tested):
`stop_regex` (letter-final-answer stop), `stop_on_repeat` (digit/case-insensitive repeat-step loop
guard), plus the probe's `--step-token`.

### Run 13 — P9 with the matched delimiter (7B, 957 items)

`step_token='\n'` (one `Step N:` line per chunk, ~8–14 real resampling rounds), `max_steps=14`;
control = Run 10's P9 cells (`\n\n`, ~1 terminal resample), same items/config. 7,656 rows, all
errors swept. Raw: `run13_p9delim/p9delim_957.*`.

| cell (mean_logprob) | selected: ctrl→matched | oracle: ctrl→matched |
|---|---|---|
| b8 | 0.553 → 0.575 | 0.780 → **0.688** |
| b16 | 0.565 → 0.572 | 0.873 → **0.758** |
| b32 | 0.577 → 0.565 | 0.911 → **0.825** |

(entropy signal: same shape; selected ±2 pp noise, oracle −8…−12 pp; distinct-ratio −25%.)

**Real stepping does not improve selection (all deltas within ±3 pp CI) and destroys coverage —
oracle drops 8–12 pp at every budget.** P9's famous 0.91 oracle was largely an artifact of its
*accidental* 1-resample regime: each extra resampling round culls more of the correct minority.

### Run 14 — P5 with the fixed stop rule + loop guard (3B, 957 items)

`--stop-regex 'Answer:\s*(\\boxed\{)?\(?[A-K]\b' --stop-on-repeat --max-steps 10`; control =
Run 12's P5 cells on the same 957 (early-stop, 1–2 steps). 7,656 rows, 0 errors. Raw:
`run14_p5stopfix/p5stopfix_957.*`.

| cell (mean_logprob) | selected: ctrl→fixed | oracle: ctrl→fixed | parse: ctrl→fixed |
|---|---|---|---|
| b1 | 0.446 → **0.503** | — | 0.84 → 0.96 |
| b8 | 0.511 → 0.534 | 0.747 → 0.739 | 0.91 → 0.97 |
| b16 | 0.533 → 0.522 | 0.827 → **0.777** | 0.93 → 0.96 |
| b32 | 0.541 → **0.503** | 0.843 → 0.824 | 0.94 → 0.96 |

**The fix genuinely improves trajectories** (b1 +5.7 pp, parse 0.84→0.96, loops guarded) — **and
the longer trajectories' extra resampling rounds still lower oracle and erase the gains by b32.**
(entropy signal: same shape — b1 +5.0 pp, selected flat elsewhere, oracle −1…−2 pp; it stays the
weaker P5 signal on 3B at every budget, consistent with Run 12 finding 3.)
(Known residual: letter-shaped sub-answers still stop ~49% of trajectories early — the fix halves
premature stopping, not eliminates it. Also documented here for completeness: the substring stop
rule made ALL prior P5 EPF cells a 1–2-step regime — live-verified — and P9's `\n\n` cells a
≤2-step regime, i.e. Runs 6–12's P5/P9 numbers describe near-best-of-N variants, which the
verdict below shows is the *stronger* configuration anyway.)

### Verdict (both ablations, both models)

A third and fourth independent manipulation of resampling frequency, same answer as Run 8:
**more resampling rounds → lower oracle, unchanged selection.** The step-wise machinery isn't
merely not-earning-its-keep — every added resampling opportunity actively costs coverage. Within
the RL-free framing this closes the case: **best-of-N sampling + a better (external) selector
dominates particle filtering on this task family, at any step granularity.**

## 16. Run 15 — cross-family swap: Qwen2-Audio-7B-Instruct on the ≤30 s test subset (P4, P9)

**Why:** Runs 11–14 established the story within one model family (Qwen2.5-Omni 7B/3B). Run 15
swaps to a **different family** — Qwen2-Audio-7B-Instruct (Whisper-large-v3-style audio encoder,
Qwen2-7B text backbone, 8,192-token context) — to test which findings are architecture-general.

**Audio-length constraint → new subset.** Qwen2-Audio's encoder has a hard **30 s window per
clip** (longer audio is truncated, not chunk-processed like Omni; ~25 audio tokens/s, ≤750
tokens/clip). Per the phase decision, only items whose clips *all* fit are used:
`test_le30s-00000-of-00001.parquet` = the full 5,090-MCQ test parquet filtered by clip duration
(soundfile) → **2,190 MCQ (2,176 gradeable, 182 single-choice)**, loader `subset="test_le30s"`.
Category mix stays broad (sound 570, speech 303, music 262, spatial 262, voice_chat 250, multi 243).
Gates on both replicas: logprobs-with-audio PASS, continue_final_message PASS; A/B causality: 9/15
answers flip without audio; context audit on the 30 longest items: max 1,578 prompt tokens, 0 errors.

**Prompt screens (the model breaks the old prompt set).** 9-prompt greedy screen (n=40) then a
P4/5/7/9 screen at n=300 (stratified; SE ≈ ±0.028), both in
`run15_qwen2audio_le30s/{chunkscreen_q2a_greedy40,greedy300_p4579}.*`:

| P (n=300, greedy) | acc | acc excl-1 | reasoned | avg words | `\n\n` chunks / %≥2 | no-prediction |
|---|---|---|---|---|---|---|
| P4 plan-and-solve | 0.390 | 0.333 | 74% | 46 | **1.80 / 46%** | **6%** |
| P5 least-to-most | 0.227 | 0.208 | 21% | 12 | 1.42 / 12% | **43%** |
| P7 format-forcing | 0.383 | 0.371 | 61% | 31 | 1.32 / 9% | 18% |
| P9 evidence-grounded | **0.407** | **0.383** | **79%** | 45 | 1.33 / 17% | 14% |

Qwen2-Audio is the tersest model in the campaign — *no* prompt exceeds 2.1 avg chunks (Omni-7B:
2.0–5.0). **P5 is unusable** (echoes a sub-question and stops; 43% of items yield no parseable
answer) and **P7 is dominated by P9 on every metric** → grid ran **P4 + P9** (P4 = the only real
chunker, P9 = best accuracy). Third consecutive model where format compliance reshuffles the
usable prompt set — screen per model, always.

**Config:** canonical probe grid — P4,P9 × {mean_logprob, entropy} × budgets {1,8,16,32},
temp 0.8, ess 0.6 / early 0.7, systematic, style logit, `step_token='\n\n'`, `stop="Answer:"`,
max_steps 6; budget-staged, b1 at `--max-inflight 24`. **35,040 rows, 0 errors**, ~4.8 h wall on
2 GPUs (fast: short clips + terse generations). Raw: `run15_qwen2audio_le30s/epf_q2a_le30s.*`.

**Results (n=2,176 gradeable; SE_full ≈ ±0.011).** selected / oracle / majority:

| cell | b1 | b8 | b16 | b32 |
|---|---|---|---|---|
| P4 mean_logprob | .424 / .424 / .424 | .446 / .580 / .454 | .439 / .627 / .453 | .452 / .668 / .460 |
| P4 entropy | .429 / .429 / .429 | .453 / .604 / .458 | .450 / .652 / .458 | .443 / .675 / .449 |
| P9 mean_logprob | .447 / .447 / .447 | .466 / .584 / .480 | .468 / .613 / .486 | .461 / .645 / .487 |
| P9 entropy | .442 / .442 / .442 | .466 / .591 / .485 | .466 / .632 / .491 | .460 / .659 / .484 |

(distinct-ratio 0.17–0.19 → 0.05–0.06 from b8→b32; final ESS 0.80–0.93; parse 0.96–0.98 at b≥8,
0.95 at P9 b1.)

**Takeaways.**
1. **The shape replicates on a third model / second family:** selected saturates by b8
   (best cell, P9 mean_logprob b16, +2.1 pp over b1; largest gain anywhere +2.8 pp ≈ 1.8×SE —
   none significant), oracle keeps climbing, the oracle−selected gap widens with budget to
   0.18–0.23 at b32; the two signals are indistinguishable on selection (≤1.1 pp; entropy holds
   a small +2–2.5 pp oracle edge on P4).
2. **Oracle ceiling is family-dependent, not universal:** 0.64–0.67 @32 vs 0.83–0.91 on Omni-7B's
   matching P4/P9 cells. Within the Omni family coverage was size-invariant (7B ≈ 3B); across
   families it tracks base competence (b1 0.42–0.45 here vs 0.54–0.57 on Omni-7B's Run-11 b1
   cells). Exploration can only surface what the model can produce at all.
3. **Majority vote beats the self-certainty selection on P9 at every budget ≥ 8** (e.g. b16:
   0.491 vs 0.466 entropy; margins +1.4–2.5 pp), and edges it on P4 as well (+0.6–1.3 pp, within
   noise) — the first model where plain voting strictly dominates the EPF argmax. Self-certainty calibration is worse here than on either Omni size;
   the selector remains the wall, and on this model even the *cheapest* external-ish signal
   (consensus) already clears it.
4. **Near-best-of-N regime caveat** (same as §15): with 1.3–1.8 avg chunks these cells run ~1–2
   resampling rounds, i.e. close to best-of-N — which §15 showed is the *stronger* configuration
   anyway. No step-boundary ablation was run here; the model's terseness leaves little room for one.

### Run 16 (handoff — high-budget extension, external GPUs)

Runs 11/12/15 extended to **budgets 64 and 128**, packaged for execution on a collaborator's
machine: `benchmarking/mmau_pro/run16/` (scripted env / pinned data+model fetch / smoke /
single-command driver; the b1–32 rows ship as gzipped seeds so only the new cells run). Scope:
Run 11 default **P5 only** (`PROMPTS_RUN11=4,5,7,9` opts into the full grid), Runs 12/15 full
grids. Output: `results/run16_hibudget/epf_full5090_bootstrap_b128.html` (a NEW file; the b1–32
report stays untouched). Results to be appended here when the run lands.

## 17. Honest verdict

*(Written after Run 4/testmini; §§13–16 confirm and sharpen every point at full scale, across two
model sizes and a second model family, and under step-boundary ablations. The July-2026 summary is
TL;DR 2/3 at the top.)*

- **Best ITS number: 0.576 (#4 PF@4), ≈ +4 pp over our own baseline, no RL used** — borderline‑significant (p≈0.07), prompt‑dependent, category‑localized, and **non‑scaling** with budget.
- It is **not** a "ITS matches GRPO for free" result. The gain is small and self‑certainty behaves like a weak, fluency‑based reward — it can't reliably steer resampling toward *correct* trajectories.
- Leaderboard context (reported numbers, full MMAU‑Pro test set, NOT our measurement): base Qwen2.5‑Omni ≈ 52%, AF3 51.7%, Gemini‑2.5‑Flash 59.2%, human 77.9%. Our testmini‑MCQ base (0.53–0.56) is in range; best ITS (0.576) is a few points above base.

### Caveats / what NOT to claim
- **No GRPO/RL result exists on MMAU‑Pro.** Published GRPO numbers (R1‑AQA, SARI, **Omni‑R1: 65.9 → 71.3**) are on the **original MMAU**, a *different and easier* benchmark — not comparable to these MMAU‑Pro numbers. The "+4 pp" here is **over our own no‑ITS baseline**, *not* over any RL result.
- n=952 → ±3.3 pp CI; the #4 effect is borderline (p≈0.07), not conclusively significant.
- `baseline` = PF at budget 1 (single self‑certainty trajectory), i.e. the same CoT prompt with no resampling.

## 18. Files (`benchmarking/mmau_pro/results/`)

Organized **one folder per run** (see `results/README.md` for the full index); cross-run figures in
`results/plots/`. Each run folder holds `<experiment>.jsonl` (raw, resumable), `.csv`, `.log`, plus
one-file HTML reports.

| folder / file | what |
|------|------|
| `run01_cot_screen/cot_compare.log` | **Run 1**: 8‑prompt CoT comparison (n=20, smallest) |
| `run02_ablation/mmau_ablation.jsonl` | **Run 2**: n=30 CoT×ITS ablation, 270 rows |
| `run03_sweep150/mmau_150_sweep.jsonl` | **Run 3**: 150‑item budget sweep, 2100 rows |
| `run04_full957/mmau_957_results.jsonl` (+`.log`) | **Run 4**: full 957 run, 5742 rows (the headline result) |
| `run05_cot957/cot957.{jsonl,csv,log}` | **Run 5**: full 957 × 9 greedy bake-off (8,613 responses) |
| `run05_cot957/cot957_html/`, `cot957_all.html` | **Run 5**: summary+matrix pages; single-file side-by-side (~6 MB) |
| `run06_epf_div/epf_div.{jsonl,csv,log}` | **Run 6**: EPF diversity sweep n=100 (per-item SMC metrics) |
| `run07_rerank/rerank.{jsonl,csv,log}` | **Run 7**: terminal answer-confidence re-rank, 1,200 rows |
| `run08_divsource/divsource.*` | **Run 8**: EPF vs INDEP + per-step ESS (n=100, temp 0.8) |
| `run08_divsource/divsource_full.*` | **Run 8 (full)**: same on all 957 MCQ — appended to §10 |
| `run09_divsource_t1/divsource_t1.*` | **Run 9**: EPF vs INDEP at temp 1.0 (n=100) |
| `run10_run6_full/run6_full.*` | **Run 10**: Run 6 grid on full 957; 30,624 rows. mean_logprob{8,16,32} seeded from Run 8 |
| `run10_run6_full/epf_div.html` | one-file report (trend heatmaps + appendix), **generated from `run6_full.csv`** |
| `run10_run6_full/epf_div_bootstrap.html` | **Run 10**: bootstrap error bars (std100 + SE957) |
| `run11_epf_full5090/epf_full5090.{jsonl,csv,log}` | **Run 11**: the grid on the FULL 5,090-MCQ test set; 162,880 rows (957-slice seeded from `run6_full.jsonl`) |
| `run11_epf_full5090/epf_full5090_bootstrap.html` | **Runs 11+12+15 combined report**: per-model sections (Omni-7B + Omni-3B + Qwen2-Audio), bootstrap error bars + interactive acc-vs-budget plots |
| `run12_omni3b_5090/epf3b_5090.{jsonl,csv,log}` | **Run 12**: Omni-**3B** grid, P4+P5 × 2 signals × {1,8,16,32} × 5,090; 81,440 rows |
| `run12_omni3b_5090/chunkprobe_3b_greedy100.csv` (+`chunkprobe_ids.txt`) | **Run 12 side-probe**: 100 seeded items × 9 prompts, greedy 3B responses (chunkability/format-compliance analysis; 7B counterpart = `run05_cot957/cot957.csv`) |
| `run13_p9delim/p9delim_957.{jsonl,csv,log}` | **Run 13**: P9 × 7B × 957 with matched `\n` delimiter (control = Run 10 P9 cells) |
| `run14_p5stopfix/p5stopfix_957.{jsonl,csv,log}` | **Run 14**: P5 × 3B × 957 with letter-answer stop regex + repeat guard (control = Run 12 P5 cells) |
| `run15_qwen2audio_le30s/epf_q2a_le30s.{jsonl,csv,log}` | **Run 15**: Qwen2-Audio-7B-Instruct grid, P4+P9 × 2 signals × {1,8,16,32} × 2,190 (≤30 s subset); 35,040 rows, 0 errors |
| `run15_qwen2audio_le30s/chunkscreen_q2a_greedy40.*`, `greedy300_p4579.*` | **Run 15 screens**: 9-prompt × 40 + P4/5/7/9 × 300 greedy (prompt pick; P5/P7 fail on this model) |
| `run16_hibudget/` (created by `run16/run_all.sh`) | **Run 16**: b64/b128 extension of Runs 11/12/15 + `epf_full5090_bootstrap_b128.html`; package/seeds in `benchmarking/mmau_pro/run16/` |
| `plots/` | cross-run figures: `acc_vs_budget.{png,html}` (Run 6), `epf_acc_vs_budget.{png,html}` (Run 10), `acc_vs_budget_combined.html`, `epf_temp_*.html` (EPF × self-consistency overlays) |
| `smoke/mmau_smoke.jsonl` | initial 8‑item pipeline smoke |

Each `run_mmau` row: `{unique_id, method, arm, budget, category, length_type, correct, latency_s, error, content}`.

## 19. Reproduce

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
  --output benchmarking/mmau_pro/results/run04_full957/mmau_957_results.jsonl

# Run 5: full 957 x 9 greedy bake-off (resumable) + paginated HTML
uv run python -m benchmarking.mmau_pro.cot_compare \
  --endpoint http://localhost:8100/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --subset full --select all --audio-mode local-path --concurrency 24 \
  --jsonl benchmarking/mmau_pro/results/run05_cot957/cot957.jsonl \
  --csv   benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
  --log   benchmarking/mmau_pro/results/run05_cot957/cot957.log
uv run python -m benchmarking.mmau_pro.make_report \
  --in benchmarking/mmau_pro/results/run05_cot957/cot957.csv \
  --out-dir benchmarking/mmau_pro/results/run05_cot957/cot957_html --paginate category

# Run 6: EPF diversity sweep on both GPUs (start a 2nd replica on GPU1 first)
CUDA_VISIBLE_DEVICES=1 ... vllm serve ... --port 8101   # 2nd replica (same launch cmd as :8100)
uv run python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
  --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 --limit 100 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run06_epf_div/epf_div.jsonl \
  --csv   benchmarking/mmau_pro/results/run06_epf_div/epf_div.csv \
  --log   benchmarking/mmau_pro/results/run06_epf_div/epf_div.log

# Run 7: terminal answer-confidence re-rank (3 scorers x 2 rules vs baselines, both GPUs)
uv run python -m benchmarking.mmau_pro.rerank_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 --max-inflight 48 \
  --jsonl benchmarking/mmau_pro/results/run07_rerank/rerank.jsonl \
  --csv   benchmarking/mmau_pro/results/run07_rerank/rerank.csv \
  --log   benchmarking/mmau_pro/results/run07_rerank/rerank.log

# Run 8: diversity source — EPF (resample ON) vs INDEP (resample OFF) + per-step ESS
uv run python -m benchmarking.mmau_pro.divsource_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --budgets 8,16,32 --limit 100 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run08_divsource/divsource.jsonl \
  --csv   benchmarking/mmau_pro/results/run08_divsource/divsource.csv \
  --log   benchmarking/mmau_pro/results/run08_divsource/divsource.log

# Run 10: Run 6's EPF budget sweep on the FULL 957 (both signals, budgets 1/8/16/32).
# (mean_logprob{8,16,32} were seeded from Run 8's full-957 EPF arm to avoid recompute.)
uv run python -m benchmarking.mmau_pro.diversity_probe \
  --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
  --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
  --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets 1,8,16,32 \
  --select all --limit 2000 --temp 0.8 --max-inflight 64 \
  --jsonl benchmarking/mmau_pro/results/run10_run6_full/run6_full.jsonl \
  --csv   benchmarking/mmau_pro/results/run10_run6_full/run6_full.csv \
  --log   benchmarking/mmau_pro/results/run10_run6_full/run6_full.log

# Run 11: the grid on the FULL 5,090-MCQ test set (budget-staged; b1 needs low inflight,
# and serve with --gpu-memory-utilization 0.85 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# — see SETUP_GUIDE.md). Seed first: cp results/run10_run6_full/run6_full.jsonl results/run11_epf_full5090/epf_full5090.jsonl
for B in 1 8 16 32; do INFLIGHT=64; [ "$B" -eq 1 ] && INFLIGHT=24
  conda run -n epf python -m benchmarking.mmau_pro.diversity_probe \
    --endpoints http://localhost:8100/v1,http://localhost:8101/v1 --model-name qwen-omni \
    --data-root /home/exx/inference-time-scaling/mmau_pro_testmini \
    --subset test --audio-root /home/exx/inference-time-scaling/mmau_pro_audio \
    --prompts 4,5,7,9 --signals mean_logprob,entropy --budgets $B \
    --select all --limit 6000 --temp 0.8 --ess-threshold 0.6 --early-phase 0.7 \
    --max-inflight $INFLIGHT \
    --jsonl benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.jsonl \
    --csv   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv \
    --log   benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.log
done
conda run -n epf python -m benchmarking.mmau_pro.epf_bootstrap --n 10000 \
  --in  benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv \
  --out benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090_bootstrap.html

# Run 12: Omni-3B grid (serve Qwen/Qwen2.5-Omni-3B on both GPUs, same flags, served name qwen-omni-3b;
# P4,P5 only — P7/P9 don't chunk on 3B, see §14). Same staged loop as Run 11 with:
#   --model-name qwen-omni-3b --prompts 4,5 \
#   --jsonl/... benchmarking/mmau_pro/results/run12_omni3b_5090/epf3b_5090.{jsonl,csv,log}
# Combined 7B+3B report (one section per --in "LABEL=path"):
conda run -n epf python -m benchmarking.mmau_pro.epf_bootstrap --n 10000 \
  --in "Qwen2.5-Omni-7B — Run 11 (P4,P5,P7,P9)=benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090.csv" \
  --in "Qwen2.5-Omni-3B — Run 12 (P4,P5)=benchmarking/mmau_pro/results/run12_omni3b_5090/epf3b_5090.csv" \
  --out benchmarking/mmau_pro/results/run11_epf_full5090/epf_full5090_bootstrap.html

# Run 13 (P9 matched delimiter, 7B) / Run 14 (P5 stop-fix, 3B) — same probe, new flags:
#   Run 13: --prompts 9 --max-steps 14 --step-token $'\n'
#   Run 14: --prompts 5 --max-steps 10 --stop-regex 'Answer:\s*(\\boxed\{)?\(?[A-K]\b' --stop-on-repeat
# (957 items: --subset full --select all --limit 1000; budget-1 stage at --max-inflight 24)

# Run 15: Qwen2-Audio-7B-Instruct. Serve Qwen/Qwen2-Audio-7B-Instruct (served name qwen2-audio)
# with the same template but --max-model-len 8192 (its full context; Whisper-style encoder
# truncates clips at 30 s). Build the subset once — filter test-*.parquet to rows whose clips ALL
# have duration ≤30 s (soundfile) → data/test_le30s-00000-of-00001.parquet (2,190 MCQ) — then the
# same staged loop with:
#   --model-name qwen2-audio --subset test_le30s --prompts 4,9 \
#   --jsonl/... benchmarking/mmau_pro/results/run15_qwen2audio_le30s/epf_q2a_le30s.{jsonl,csv,log}
# Combined 3-model report: the Run-12 epf_bootstrap command plus
#   --in "Qwen2-Audio-7B-Instruct — Run 15 (P4,P9; le30s subset)=benchmarking/mmau_pro/results/run15_qwen2audio_le30s/epf_q2a_le30s.csv"

# Run 16 (b64/b128 extension of Runs 11/12/15, any machine): fully scripted —
#   see benchmarking/mmau_pro/run16/README.md; after setup/fetch/smoke it is one command:
#   nohup bash benchmarking/mmau_pro/run16/run_all.sh > run16.log 2>&1 &
```

## 20. Next lever

**Run 7 ruled out the obvious self-signal fix.** We tested the answer-choice-confidence reward as a terminal
re-rank (answer-letter confidence and option-text likelihood, with and without audio) and it does **not**
recover the oracle gap — it reshuffles correct↔incorrect roughly one-for-one (net ≈ 0). Combined with Runs
4/6, this rules out *any* self-generated confidence (fluency, answer-letter, option-text) as the weight: the
generator is confidently wrong on the items it misses, so no reweighting of its own signal can separate its
right answers from its wrong ones.

**Run 8 added a second structural finding:** EPF's *resampling is net-harmful* on this task — it culls the
correct (lower-fluency) minority and lowers the oracle ceiling ~12 pp vs plain independent sampling, with no
gain in selected accuracy. So the EPF/PF machinery isn't earning its keep here.

**The implied pipeline within the RL-free framing:** drop particle filtering for **plain best-of-N sampling**
(no resampling → oracle ~0.96 @N=32) and spend the budget on a **better *independent* selector** — a signal
that doesn't share the generator's blind spots:
- a **different judge model** scoring/ranking the N finished candidates (cross-model verification), or
- a **trained verifier / PRM** (supervised on correctness) — but that re-introduces training, leaving the
  RL-free framing.

Cheaper diagnostics worth a look before committing: (a) re-rank with a *stronger* model as judge on the
oracle-but-EPF-wrong items to confirm an external signal *can* recover them (upper-bound check); (b) a
self-critique/debate pass (still self-signal, likely same ceiling, but cheap to falsify). The honest read:
within the fixed-single-model, RL-free constraint, **inference-time scaling on MMAU-Pro is selection-limited
and that limit is the model's own calibration** — not solvable by more particles or a cleverer self-weight.
