# Change Report: Making `its_hub` Audio-LLM Compatible + MMAU-Pro Benchmarking

## Context

`its_hub` was originally a **text-only** inference-time-scaling (ITS) library aimed at
mathematical reasoning. It talked to models over an OpenAI-compatible chat API, exposed **no token
logprobs**, and its step-by-step algorithms (Beam Search, Particle Filtering, Entropic Particle
Filtering) **flattened every prompt to a plain string** before generation — which silently dropped any
non-text content. The probabilistic algorithms (PF/EPF) also required a separate trained **Process
Reward Model (PRM)** to weight particles.

The goal of this work was to run **PF / EPF on a Large *Audio* LM (Qwen2.5-Omni)** and evaluate it on
the **MMAU-Pro** audio MCQ benchmark, using the generator's *own* self-certainty as the particle weight
(since no audio PRM exists). That required three groups of changes:

1. **Foundation** — derive particle weights from the generator's own logprobs (no external PRM).
2. **Audio carry** — let structured audio survive through the step-by-step generation path.
3. **MMAU-Pro harness** — a new benchmarking package + the environment/serving recipe.

This report names every change. (Group 1 was committed separately as "added generator log probs";
Groups 2–3 are the audio/benchmark work.)

---

## File-by-file summary

| File | Status | Purpose |
|---|---|---|
| `its_hub/core/lms/openai_lm.py` | modified | request + capture token logprobs |
| `its_hub/core/utils.py` | modified | summarize a step's logprobs |
| `its_hub/core/lms/step_generation.py` | modified | return logprobs per step **+ carry structured (audio) messages** |
| `its_hub/core/algorithms/particle_gibbs.py` | modified | self-certainty weight source **+ thread audio messages through PF/EPF** |
| `its_hub/api/types.py` | modified | tolerate audio content; helpers to detect/carry structured messages |
| `tests/test_self_certainty.py` | new | self-certainty weight tests (9) |
| `tests/test_audio_carry.py` | new | audio-carry tests (9) |
| `benchmarking/mmau_pro/` (package) | new | MMAU-Pro MCQ loader/audio/prompt/scoring/runner + gates |
| `tests/test_mmau_pro.py` | new | harness offline tests (9) |

**Not changed (deliberately):** `openai_lm.py` needed **no audio-specific change** (an `input_audio`
content part is just a dict in `ChatMessage.content`, which `to_dict()` already passes through, and the
vLLM `continue_final_message` path was already wired). Beam Search, Self-Consistency, Best-of-N, and the
PRM path (`local_vllm_prm.py`) were left untouched — out of scope for the audio PF/EPF experiment.

---

## Group 1 — Foundation: generator self-certainty weights (no external PRM)

This replaces "weight particles with a separate PRM" by "weight them with the generator's own token
log-probabilities," which is what makes PF/EPF usable on audio (where no PRM exists).

### `its_hub/core/lms/openai_lm.py`
- `_prepare_request_data(...)`: added parameters **`logprobs: bool = False`** and
  **`top_logprobs: int | None = None`**; added a block that sets `request_data["logprobs"] = True` and
  `request_data["top_logprobs"] = top_logprobs` when requested. (Defaults off → existing requests
  unchanged.)
- `_agenerate(...)` (the batched path used by step generation): added `logprobs`/`top_logprobs`
  parameters; passed them into `_prepare_request_data`; and in the response extraction added
  **`if choice.get("logprobs") is not None: message["_logprobs"] = choice["logprobs"]`** so the
  per-token logprobs ride back on the returned message dict.
- `agenerate(...)`: added `logprobs`/`top_logprobs` parameters; forwarded to `_agenerate`.
- `agenerate_single(...)`: added `logprobs`/`top_logprobs` parameters (before `loop`); forwarded to
  `_prepare_request_data`; same `message["_logprobs"]` capture.

### `its_hub/core/utils.py`
- Added `import math`.
- Added **`summarize_step_logprobs(logprobs) -> {"mean_logprob", "entropy", "num_tokens"}`** — reduces an
  OpenAI-style `logprobs.content` list to (a) the mean per-token logprob and (b) the mean per-token
  entropy approximated over the returned `top_logprobs`. Returns zero/None gracefully when logprobs are
  absent.

### `its_hub/core/lms/step_generation.py`
- Imported `summarize_step_logprobs`.
- `StepGeneration.aforward(...)`: added **`return_logprobs: bool = False`** and
  **`top_logprobs: int | None = None`**. Built a `logprob_kwargs` dict that is forwarded to the LM **only
  when `return_logprobs` is True** (so LMs/mocks that don't accept those kwargs keep working). When
  enabled, both the single and batch paths return a **3-tuple** `(next_step, is_stopped,
  logprob_summary)` instead of the usual 2-tuple.

### `its_hub/core/algorithms/particle_gibbs.py`
- Added enum **`WeightSource { PRM, SELF_CERTAINTY }`**.
- `ParticleGibbs.__init__(...)`: made **`prm` optional** (`AbstractProcessRewardModel | None = None`);
  added **`weight_source`**, **`self_certainty_signal`** (`"mean_logprob" | "entropy"`),
  **`self_certainty_style`** (`"logit" | "raw"`), **`top_logprobs`**. Added validation (require `prm`
  when `weight_source == PRM`; validate signal/style; auto-set `top_logprobs=20` for the entropy signal);
  stored the new attributes.
- Added method **`_self_certainty_logweight(summary)`**: maps a step's logprob summary to a particle
  log-weight. Both signals reduce to a confidence `c ≤ 0` (`mean_logprob`, or `-entropy`); style `"raw"`
  uses `c` directly, style `"logit"` uses `_inv_sigmoid(exp(c))` (so it reuses the exact same transform
  the PRM path used, keeping EPF's annealing identical).
- `_apropagate(...)`: branched on `weight_source`. The **self-certainty** branch calls
  `sg.aforward(..., return_logprobs=True, top_logprobs=...)` and appends
  `_self_certainty_logweight(summary)` to each particle's `partial_log_weights` — **no PRM call**. The
  **PRM** branch is the original code, unchanged.
- `ParticleFiltering.__init__` and `EntropicParticleFiltering.__init__`: made `prm` optional and threaded
  the four new parameters (`weight_source`, `self_certainty_signal`, `self_certainty_style`,
  `top_logprobs`) through to `super().__init__`.

### `tests/test_self_certainty.py` (new — 9 tests)
A `LogprobMockLM` emitting fake logprobs; tests for `summarize_step_logprobs`, the
`_self_certainty_logweight` signal×style matrix, end-to-end PF/EPF with `weight_source="self_certainty"`,
and the `prm`-required / invalid-option guards.

---

## Group 2 — Audio step-path carry (the core audio-compatibility change)

**Problem:** PF/EPF generate step-by-step. `ParticleGibbs.ainfer` called `chat_messages.to_prompt()` (a
**string**), and `StepGeneration` rebuilt the user turn as `ChatMessage(role="user", content=<string>)`.
`to_prompt()` keeps only `type == "text"` parts, so **audio was dropped**; worse,
`ChatMessage.extract_text_content()` **raised** on an `input_audio` part. So the model never received the
audio.

**Fix:** carry the **structured user message(s)** (audio + text) verbatim through the step loop, and
append each particle's reasoning-so-far as a trailing **assistant** turn the model *continues* (vLLM
`continue_final_message`, already emitted when the last message is an assistant turn). The reasoning
steps stay text; only the user turn stays structured.

### `its_hub/api/types.py`
- `ChatMessage.extract_text_content()`: **removed the `raise ValueError`** on unknown content types; now
  unknown types (e.g. `input_audio`, `audio_url`) are **skipped** (the image warning is kept). This stops
  audio messages from crashing any text-extraction path.
- Added **`ChatMessages.has_nontext_content()`** — returns True if any message's `content` is a list
  containing a non-text part (audio/image). Used to decide whether to carry structured messages.
- Added **`ChatMessages.base_user_messages()`** — returns the underlying messages to carry verbatim; for
  the string case it returns `[ChatMessage(role="user", content=<str>)]`, identical to the old behavior
  (so the plain-text path is unchanged).

### `its_hub/core/lms/step_generation.py`
- `StepGeneration.aforward(...)`: added **`base_messages: list[ChatMessage] | None = None`**. When
  `None`, behavior is byte-identical to before. When provided, the per-step message list is built as
  **`[*base_messages, ChatMessage("assistant", post_process(steps_so_far))]`** (the assistant turn is
  only appended once there are steps). In the **batch** path the same `base_messages` is **broadcast**
  across all particles (so the multi-MB audio payload isn't duplicated per particle in Python). A shallow
  `list(base_messages)` copy avoids mutating the caller's list.

### `its_hub/core/algorithms/particle_gibbs.py`
- `_apropagate(...)`: added a **`base_messages`** parameter, forwarded to `sg.aforward(...)`.
- `ParticleGibbs.ainfer(...)`: after normalizing input, computes once
  **`carry_structured = chat_messages.has_nontext_content()`**,
  **`base_messages = chat_messages.base_user_messages() if carry_structured else None`**, and keeps
  `prompt_str = chat_messages.to_prompt()` (used for logging / the PRM path). The propagation loop now
  passes both `prompt_str` and `base_messages` to `_apropagate`. (The old `# TODO: support native
  ChatMessages` comment was removed since this implements it.) `ParticleFiltering`/`EntropicParticleFiltering`
  inherit this for free (they call `super().ainfer`).

### `tests/test_audio_carry.py` (new — 9 tests)
An `AudioEchoMockLM` that **records the exact messages it receives**, used to assert: the `input_audio`
part survives **verbatim for every particle**; multi-audio order is preserved; the first step has no
assistant turn (then a continued assistant turn appears); the base user turn is identical across
particles; the **plain-text path is byte-identical** (no audio → string path unchanged); and the
`types.py` helpers behave (`extract_text_content` no longer raises, `has_nontext_content`,
`base_user_messages`).

---

## Group 3 — MMAU-Pro MCQ benchmarking harness (new package)

New package **`benchmarking/mmau_pro/`** (kept out of the library import surface).

- **`__init__.py`** — package marker / docstring.
- **`loader.py`**
  - `MCQRecord` dataclass: `unique_id, question, choices, answer, audio_paths (absolute), category,
    length_type, answer_index`.
  - `record_from_row(row, data_root)` — converts one parquet row to an `MCQRecord`; **numpy-array-safe**
    (parquet list-columns come back as arrays, so it avoids truthiness on arrays); returns `None` for
    non-MCQ (empty `choices`); resolves relative `audio_path` entries to absolute; precomputes
    `answer_index` (incl. fuzzy match, see scoring).
  - `load_mmau_mcq(data_root, subset, limit, require_audio_exists)` — reads
    `testmini-…parquet` (or `testmini_le30s-…parquet`), filters to the **957 MCQ**, optional
    audio-existence filter, optional limit.
- **`audio.py`**
  - `audio_content_parts(audio_paths, mode)` — builds one content part per clip (preserving order):
    `mode="base64"` → `{"type":"input_audio","input_audio":{"data":<b64>,"format":<ext>}}`;
    `mode="local-path"` → `{"type":"audio_url","audio_url":{"url":"file://<abs>"}}`. Base64 results are
    cached per path. (No local decode/resample — vLLM resamples server-side.)
- **`prompt.py`**
  - `MMAU_MCQ_SYSTEM_PROMPT`, `format_choices(choices)` (letters A…K), and the original terse
    `build_messages(record, audio_mode, system_prompt)`.
  - `METHODS` (the 8 CoT-elicitation methods) and **`build(method, rec, audio_mode)`** — returns
    `(messages, assistant_seed)` for: 1 assistant-prefill CoT, 2 zero-shot CoT (user trigger), 3 few-shot
    CoT, 4 plan-and-solve, 5 least-to-most, 6 describe-then-reason (audio-grounded), 7 format-forcing
    (`## Step`), 8 anti-shortcut. Each builds a multimodal user turn (audio parts + lettered-choice text).
- **`scoring.py`** (pure, offline-testable)
  - `normalize(s)` (lowercase, strip punctuation/articles); `LETTERS`.
  - `match_answer_index(answer, choices)` — maps the gold *answer text* to a choice index (exact-normalized
    then fuzzy via `difflib`), since MMAU-Pro's `answer` is the choice **text**, not a letter, and ~23/957
    answers aren't verbatim in `choices`.
  - `extract_letter(text, num_choices)` — parses the model's chosen letter; the standalone-letter fallback
    is **uppercase-only** (so a natural-language "a"/"i" isn't mistaken for an option).
  - `predicted_index(text, choices)` — letter first, then normalized choice-text match.
  - `is_correct(text, choices, answer_index)` — `True/False`, or `None` when ungradeable.
- **`run_mmau.py`** — click CLI; the main evaluation runner.
  - `build_algorithm(arm, max_steps)` — constructs `StepGeneration(step_token="\n\n",
    stop_token="Answer:", max_steps=…)` so reasoning is **chunked into PF/EPF steps on blank lines**, and
    builds baseline/PF/EPF with `weight_source="self_certainty"` (PF: `mean_logprob`/`logit`; EPF:
    `entropy`).
  - Loops over **prompt-methods × arms × budgets × records**; tags each row with `method`; writes a
    **resumable** JSONL keyed by `(unique_id, method, arm, budget)`; per-item try/except so one failure
    doesn't abort the sweep; `report()` prints accuracy per `(prompt, arm, budget)`. Flags include
    `--subset`, `--limit`, `--single-audio`, `--prompt-methods`, `--arms`, `--budgets`, `--audio-mode`,
    `--max-steps`, `--max-tokens-per-step`, `--max-concurrency`. The LM is given a `max_tokens` cap so a
    `\n\n`-less step can't run away.
- **`phase0_gate.py`** — GPU go/no-go checks against the live endpoint: **GATE 1** (generated-token
  logprobs returned *with audio input*) and **GATE 2** (`continue_final_message` works *with an audio
  user turn*).
- **`ab_causality.py`** — runs each item **with** vs **without** the audio part to prove the model
  actually uses the audio (`_strip_audio` removes the audio content parts).
- **`cot_compare.py`** — compares all 8 prompts on (reasoned? / chunkability / accuracy) to pick the CoT
  prompt.

### `tests/test_mmau_pro.py` (new — 9 tests)
Pure tests for scoring (`match_answer_index`, `extract_letter`, `predicted_index`, `is_correct`), loader
(`record_from_row` incl. multi-audio + non-MCQ skip), prompt (`build` produces a structured audio user
turn with lettered choices), and a real-data check confirming **exactly 957 MCQ** load from the local
parquet (skipped if the data dir is absent).

---

## Group 4 — Environment & serving (required to actually run audio)

These aren't repository code, but were necessary to make audio inference work and are documented here for
reproducibility.

- **Conda env `epf` deps added:** `pyarrow` (parquet; already present), and for **server-side audio
  decode** `librosa`, `soundfile`, **`av` (PyAV)**, `resampy`. Without PyAV, vLLM raised *"Please install
  vllm[audio]"* / *"Invalid or unsupported audio file."*
- **vLLM serving recipe (Qwen2.5-Omni-7B on Blackwell):**
  - `VLLM_USE_FLASHINFER_SAMPLER=0` — the flashinfer JIT sampling kernel fails to build on sm_120;
    disabling it uses the native sampler.
  - `HF_HOME=<3.4TB vol>/hf_cache` (weights live off the full root disk).
  - `vllm serve Qwen/Qwen2.5-Omni-7B --served-model-name qwen-omni --port 8100 --trust-remote-code
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 --enforce-eager
    --allowed-local-media-path <data_root> --limit-mm-per-prompt '{"audio":3}'`.
  - Audio input that works end-to-end = **base64 `input_audio`**. Prefix caching is on by default, which
    reuses the (identical) audio+question prefix across PF steps and particles.

---

## Verification performed

- **Offline (CPU, no GPU):** full suite **243 tests pass** (216 legacy + 9 self-certainty + 9 audio-carry
  + 9 MMAU-Pro), ruff clean on all changed files. The audio-carry tests prove the audio bytes reach the
  model verbatim for every particle.
- **On the served model (GPU):**
  - **Phase-0 gates PASS** — generated-token logprobs *are* returned with audio (self-certainty is viable,
    no fallback needed); `continue_final_message` works with an audio user turn.
  - **A/B causality PASS** — accuracy with audio (0.533) > without audio (0.400) on 15 items, and
    audio-only questions flip correctly → the carry truly delivers audio.
  - **Pipeline smoke** — baseline/PF/EPF run end-to-end on audio MCQ with 0 errors.
  - **CoT prompt comparison** — fixed the "terse single-letter answer" issue; with the right prompt the
    model now produces genuine chunkable multi-step reasoning for PF/EPF to resample over.

## Net effect

`its_hub` can now run **Particle Filtering / Entropic Particle Filtering on an audio LM**, weighting
particles by the **generator's own self-certainty** (no PRM), with audio delivered to the model at every
reasoning step — and there is a self-contained **MMAU-Pro MCQ harness** to measure base→ITS uplift. The
text-only behavior of the library is unchanged (all new parameters default to the original behavior).
