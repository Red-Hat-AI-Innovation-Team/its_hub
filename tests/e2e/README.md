# E2E Tests

End-to-end tests for the its_hub particle-filtering algorithms (PF and EPF) against an OpenAI-compatible API endpoint (vLLM, OpenAI, etc.).

The endpoint must support `logprobs` (vLLM does) — particle weights come from the generator's own token logprobs (self-certainty), so no separate reward model is needed.

## Prerequisites

- A running OpenAI-compatible server (e.g. vLLM)
- `pip install its_hub[dev]` (or `uv sync --extra dev`)
- `math-verify` package (`pip install math-verify`)

## Quick Start

```bash
python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --verbose
```

## Test Modes

**Async (default):** Uses `algorithm.ainfer()` with a single event loop.

```bash
python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct
```

**Sync:** Uses `algorithm.infer()` where each problem gets its own event loop.

```bash
python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --sync
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | *(required)* | OpenAI-compatible API endpoint |
| `--model_name` | *(required)* | Model name served at the endpoint |
| `--api_key` | `NO_API_KEY` | API key |
| `--temperature` | `0.7` | Sampling temperature |
| `--max_tokens` | None | Max tokens per generation |
| `--max_concurrency` | `32` | Max concurrent requests |
| `--budget` | `4` | Computation budget (number of particles) per problem |
| `--datasets` | `math500,aime2024` | Comma-separated list of datasets |
| `--algorithms` | all available | Comma-separated list: `particle-filtering`, `entropic-particle-filtering` |
| `--tokens_per_step` | None | Tokens per step for StepGeneration |
| `--verbose` | off | Print per-problem results |
| `--sync` | off | Use sync `infer()` instead of async `ainfer()` |

## Examples

Run only particle filtering on math500:

```bash
python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --algorithms particle-filtering \
    --datasets math500 \
    --verbose
```

Run with lower budget for faster iteration:

```bash
python tests/e2e/test_e2e.py \
    --endpoint http://localhost:8100/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --budget 2
```

## Datasets

Pre-saved subsets are in `tests/e2e/data/`:
- `math500_subset.jsonl` (5 problems from MATH500)
- `aime2024_subset.jsonl` (6 problems from AIME-2024)

## Output

The test produces a results table with accuracy and latency metrics (avg, min, max per problem) for each algorithm/dataset combination.
