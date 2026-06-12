# Chapter 12 — Running It: Environment, Tests, and Inference

> *Previous: [Putting It Together](11-putting-it-together.md) · Next: [Glossary & References](99-glossary-and-references.md)*

This chapter is the hands-on companion to the theory. It documents the **conda environment** built for
this repo, how to **run the tests**, the two **inference paths**, and — importantly — the
**machine-specific gotchas** discovered while setting it up. Everything here was executed and verified on
this workstation (2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each; CUDA capability `sm_120`).

## The conda environment (`epf`)

The repo's own docs use `uv`; we built a dedicated **conda** env instead. Python **3.11** is required
(`requires-python = ">=3.11"` in [`pyproject.toml`](../pyproject.toml)).

```bash
# 1) create the env
conda create -n epf python=3.11 -y

# 2) install the library editable, with the extras we want
#    lm        → openai, aiohttp, backoff (the OpenAICompatibleLanguageModel deps)
#    dev       → [lm] + pytest, pytest-asyncio, ruff
#    benchmark → [lm] + click, pandas, pyarrow (the MMAU-Pro audio benchmark)
/home/exx/miniconda3/envs/epf/bin/python -m pip install -e ".[dev,benchmark]"

# (optional) math-verify for the e2e math harness, vLLM to serve models locally
/home/exx/miniconda3/envs/epf/bin/python -m pip install math-verify vllm
```

> ### ⚠️ Gotcha #1 — `conda activate epf` does **not** win PATH on this machine
> A uv-managed CPython sits ahead of conda on `PATH`, and it **stays ahead even after
> `conda activate epf`**. So `python`/`pip`/`ruff` after activation resolve to the *uv* interpreter, not
> the env. **Always address the env explicitly:**
> ```bash
> /home/exx/miniconda3/envs/epf/bin/python  …      # absolute interpreter (most robust)
> #   or
> conda run -n epf python  …                        # conda picks the right interpreter regardless of PATH
> ```
> Using a bare `python` after `conda activate epf` will silently use the wrong interpreter.

> ### ⚠️ Gotcha #2 — the uv python is PEP 668 "externally-managed"
> If you *do* accidentally target the uv interpreter, `pip install` aborts with
> `error: externally-managed-environment`. That's a feature — it prevented polluting the uv install.
> The conda env python is **not** externally-managed, so installs there work normally.

> ### ⚠️ Gotcha #3 — `| tail` hides pip's exit code
> `pip install … | tail -20` reports **tail's** exit status (0), masking a pip failure. Capture to a log
> and check explicitly: `pip install … > log 2>&1; echo "exit=$?"`.

### What got installed (verified on Blackwell)

The GPU stack (installed for *serving* models — the library itself no longer imports torch/vLLM)
**works on sm_120 out of the box** — no manual CUDA wrangling was needed:

| Package | Version | Note |
|---|---|---|
| torch | `2.11.0+cu130` | CUDA 13 wheels; `torch.cuda.is_available()` → `True` |
| vLLM | `0.22.1` | serves the generator model (and exposes `logprobs`, which PF needs) |
| transformers | `4.57.3` | (vLLM warns it prefers v5; harmless here) |

`torch.cuda.get_device_capability(0)` returns `(12, 0)` and sees **2** GPUs. Note there is no
reward-hub / PRM stack anymore — particle weights come from the generator's own logprobs
(self-certainty), so the only model you serve is the generator.

## Running the tests

The unit suite is **mock-based** — no GPU, no server, no API key — and is the fastest way to confirm the
env and to exercise the PF/EPF code paths.

```bash
# all unit tests (exclude the e2e suite, which needs a live endpoint)
/home/exx/miniconda3/envs/epf/bin/python -m pytest tests/ --ignore=tests/e2e -q
# → 91 passed in ~4s   (verified)

# lint
/home/exx/miniconda3/envs/epf/bin/python -m ruff check its_hub/
# → All checks passed!

# a single focused file — e.g. the heart of the repo's novelty:
/home/exx/miniconda3/envs/epf/bin/python -m pytest tests/test_entropic_annealing.py -v
```

Notes:
- `pytest-asyncio` runs in strict mode; async tests carry explicit `@pytest.mark.asyncio` markers, so no
  extra config is needed.
- The mocks live in [`tests/mocks/`](../tests/mocks/) and the fixtures (including dummy vLLM/OpenAI HTTP
  servers) in [`tests/conftest.py`](../tests/conftest.py).
- **e2e tests** ([`tests/e2e/`](../tests/e2e/)) require `OPENAI_API_KEY` and a real OpenAI-compatible
  endpoint; they `pytest.skip` gracefully without one. They're a benchmarking harness (MATH500 / AIME
  subsets in [`tests/e2e/data/`](../tests/e2e/data/)), not part of the basic acceptance gate.

## Running inference

There are two paths. **Which one to run live is deferred** until after you've digested these docs — but
both are ready.

### Path A — Local vLLM + Particle Filtering on math (no API key; uses the GPUs)

This is the path that actually exercises the repo's namesake algorithm. It needs **one** model: a
generator served by vLLM. There is no reward model — the particle weights come from the generator's own
token logprobs, so the only requirement on the endpoint is that it supports `logprobs` (vLLM does).

```bash
# Terminal 1 — serve the generator (OpenAI-compatible) on :8100
conda run -n epf vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8100

# Terminal 2 — run the math e2e harness (MATH500 / AIME subsets) against it
conda run -n epf python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
    --verbose
```

The harness ([`tests/e2e/test_e2e.py`](../tests/e2e/test_e2e.py)) wires up exactly the pieces from
Chapters 3–8; the minimal version by hand is:

```python
from its_hub import (EntropicParticleFiltering, OpenAICompatibleLanguageModel,
                     ParticleFiltering, StepGeneration)

lm  = OpenAICompatibleLanguageModel(endpoint="http://localhost:8100/v1", api_key="NO_API_KEY",
                                    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                                    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT)
sg  = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
scaling_alg = ParticleFiltering(sg)        # no prm — self-certainty weights
result = scaling_alg.infer(lm, problem, budget=...)   # budget = number of particles
```

To run **Entropic** PF instead, swap the algorithm line for `EntropicParticleFiltering(sg)`. The first
run downloads the model from Hugging Face. Math answer-checking uses `math-verify` (install it in the
env; it's not part of any extra).

### Path B — Audio: the MMAU-Pro benchmark (Qwen2.5-Omni via vLLM)

The audio path runs PF/ePF on MMAU-Pro MCQ items, carrying the audio user turn through the step loop
verbatim (see [audio-mmau-changes.md](audio-mmau-changes.md)). It needs the `[benchmark]` extra
(click, pandas, pyarrow) and a served Qwen2.5-Omni:

```bash
conda run -n epf python -m benchmarking.mmau_pro.run_mmau \
    --endpoint http://localhost:8100/v1 --model-name qwen-omni \
    --data-root /home/exx/inference-time-scaling/mmau_pro_testmini --subset full \
    --prompt-methods 2,4 --arms baseline,pf,epf --budgets 4 \
    --audio-mode local-path --item-concurrency 10 \
    --output mmau_results.jsonl
```

See [`benchmarking/mmau_pro/run_mmau.py`](../benchmarking/mmau_pro/run_mmau.py) for all flags and
[`benchmarking/mmau_pro/RESULTS.md`](../benchmarking/mmau_pro/RESULTS.md) for recorded runs.

| | Path A (math PF/ePF) | Path B (audio PF/ePF) |
|---|---|---|
| Needs GPU | yes (the generator) | yes (Qwen2.5-Omni) |
| Needs API key | no | no |
| Exercises | the namesake algorithm on MATH500/AIME | the audio base_messages carry |
| Extra | `[lm]` (+ `math-verify`) | `[benchmark]` |

## Quick reference card

```bash
PY=/home/exx/miniconda3/envs/epf/bin/python   # or: conda run -n epf python

$PY -m pytest tests/ --ignore=tests/e2e -q     # 91 unit tests, ~4s
$PY -m ruff check its_hub/                      # lint
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
conda run -n epf vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8100               # path A, terminal 1
conda run -n epf python tests/e2e/test_e2e.py --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-Math-1.5B-Instruct                                      # path A, terminal 2
conda run -n epf python -m benchmarking.mmau_pro.run_mmau --help                      # path B (audio)
```

---

*Next: [Chapter 99 — Glossary & References](99-glossary-and-references.md).*
