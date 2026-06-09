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

# 2) install the library editable, with all the extras we want
#    dev        → pytest, ruff, jupyter, + lm + iaas
#    research   → math-verify, datasets, matplotlib (benchmarks/eval)
#    experimental → transformers, reward-hub[prm]  (pulls torch + vLLM; GPU)
/home/exx/miniconda3/envs/epf/bin/python -m pip install -e ".[dev,research]"
/home/exx/miniconda3/envs/epf/bin/python -m pip install -e ".[experimental]"
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

The `[experimental]` resolve pulled a recent GPU stack that **works on sm_120 out of the box** — no
manual CUDA wrangling was needed:

| Package | Version | Note |
|---|---|---|
| torch | `2.11.0+cu130` | CUDA 13 wheels; `torch.cuda.is_available()` → `True` |
| vLLM | `0.22.1` | serves the instruction model + backs the PRM |
| reward-hub | `0.1.10` | the PRM provider behind `LocalVllmProcessRewardModel` |
| transformers | `4.57.3` | (vLLM warns it prefers v5; harmless here) |

`torch.cuda.get_device_capability(0)` returns `(12, 0)` and sees **2** GPUs. The whole import chain
(`vllm`, `reward_hub`, `LocalVllmProcessRewardModel`) loads cleanly. The CLAUDE.md note about commenting
out `reward_hub`/`vllm` on dependency trouble was **not** needed here, but keep it in your back pocket
for other machines.

## Running the tests

The unit suite is **mock-based** — no GPU, no server, no API key — and is the fastest way to confirm the
env and to exercise the EPF/PF/Beam/SC/BoN code paths.

```bash
# all unit tests (exclude the e2e suite, which needs a live endpoint)
/home/exx/miniconda3/envs/epf/bin/python -m pytest tests/ --ignore=tests/e2e -q
# → 216 passed in ~4s   (verified)

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

### Path A — Local vLLM + Particle Filtering (no API key; uses the GPUs)

This is the path that actually exercises the repo's namesake algorithm. It needs **two** models: an
*instruction* model served by vLLM (the generator) and a *process reward* model loaded by
`LocalVllmProcessRewardModel` (the judge). With 96 GB cards, both fit comfortably.

```bash
# Terminal 1 — serve the instruction model (OpenAI-compatible) on :8100
conda run -n epf vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8100

# Terminal 2 — run the bundled particle-filtering example against it
conda run -n epf python examples/test_math_example.py
```

The example ([`examples/test_math_example.py`](../examples/test_math_example.py)) wires up exactly the
pieces from Chapters 3–8:

```python
lm  = OpenAICompatibleLanguageModel(endpoint="http://localhost:8100/v1", api_key="NO_API_KEY",
                                    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
                                    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT)
sg  = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(model_name="Qwen/Qwen2.5-Math-PRM-7B", device="cuda:0",
                                  aggregation_method="prod")     # ← the "prod" from Chapter 4
scaling_alg = ParticleFiltering(sg, prm)
result = scaling_alg.infer(lm, problem, budget=...)              # budget = number of particles
```

To run **Entropic** PF instead, swap the last two lines for
`EntropicParticleFiltering(sg, prm)`. The first run downloads the models from Hugging Face (the 7B PRM is
the big one). Put the PRM on the second GPU with `device="cuda:1"` to keep it off the vLLM card.

### Path B — Cloud API + Self-Consistency / Best-of-N (no GPU)

The whole-answer algorithms work against any OpenAI-compatible endpoint and need only the `[lm]` deps
(already in our env).

```bash
export OPENAI_API_KEY=sk-...                       # keys come from the ENV, never config files
conda run -n epf python examples/self-consistency.py
```

[`examples/self-consistency.py`](../examples/self-consistency.py) points
`OpenAICompatibleLanguageModel` at `https://api.openai.com/v1` and demonstrates tool-call voting
(`tool_vote="tool_hierarchical"`). Best-of-N with `LLMJudge` is the same shape (see
[`docs/quick-start.md`](../docs/quick-start.md), Example 2).

| | Path A (local PF/ePF) | Path B (cloud SC/BoN) |
|---|---|---|
| Needs GPU | yes (instruction + 7B PRM) | no |
| Needs API key | no | yes |
| Exercises | the namesake algorithm + PRM | voting / judging |
| Extra | `[experimental]` | `[lm]` |

## The plugin path (`its_scale.sh`)

This repo also ships as a Claude Code / Codex plugin. The bash entry point
[`scripts/its_scale.sh`](../scripts/its_scale.sh) reads config from `.its-hub/config.json` (endpoints +
model + algorithm; **API keys still come from env vars**) and calls
[`scripts/_its_scale_runner.py`](../scripts/_its_scale_runner.py). Note that the plugin **deliberately
refuses** particle filtering and beam search
([`_its_scale_runner.py:33-39`](../scripts/_its_scale_runner.py#L33-L39)):

```python
if algorithm in ("particle-filtering", "beam-search"):
    print("ERROR: ... requires process reward models and is experimental in v1. "
          "Use the Python API directly for advanced algorithms.", file=sys.stderr)
    sys.exit(1)
```

So the plugin covers the GPU-free algorithms (Self-Consistency, Best-of-N); for PF/ePF use the Python
API as in Path A.

## Benchmarking & evaluation

- [`benchmarking/benchmark.py`](../benchmarking/benchmark.py) — compares algorithms across budgets on
  MATH500 / AIME-2024 (needs the `[research]` extra, already installed). See `--help`.
- [`eval/score.py`](../eval/score.py) — scoring utilities; [`docs/benchmarking.md`](../docs/benchmarking.md)
  has the full story.
- Math answer-checking uses `math-verify` (the `[research]` extra), which is why benchmarks need it.

## Quick reference card

```bash
PY=/home/exx/miniconda3/envs/epf/bin/python   # or: conda run -n epf python

$PY -m pytest tests/ --ignore=tests/e2e -q     # 216 unit tests, ~4s
$PY -m ruff check its_hub/                      # lint
$PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
conda run -n epf vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --port 8100   # path A, terminal 1
conda run -n epf python examples/test_math_example.py                     # path A, terminal 2
```

---

*Next: [Chapter 99 — Glossary & References](99-glossary-and-references.md).*
