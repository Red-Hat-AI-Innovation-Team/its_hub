# `its-hub`: Inference-time scaling with particle filtering

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)
[![PyPI version](https://badge.fury.io/py/its-hub.svg)](https://badge.fury.io/py/its-hub)

**its_hub** is a Python library for inference-time scaling of LLMs using **Particle Filtering (PF)** and **Entropic Particle Filtering (EPF)**.

Both algorithms weight particles from the **generator model's own token logprobs** (self-certainty) — no separate reward model or LLM judge is required. The only serving requirement is an OpenAI-compatible endpoint that supports `logprobs` (vLLM does). Multimodal (e.g. audio) user content is carried verbatim through step-by-step generation, so the algorithms work with audio language models such as Qwen2.5-Omni out of the box.

## How it works

1. A prompt (text, or structured messages with audio parts) is expanded into `budget` particles.
2. Each particle generates one reasoning step at a time (`StepGeneration` chunks on a step token or a token budget).
3. After every step, each particle's log-weight is derived from the generator's own per-token logprobs (`mean_logprob`) or top-k entropy for the step it just produced.
4. Particles are resampled in proportion to their weights (multinomial for PF; systematic for EPF).
5. **EPF only:** when the weights collapse early (low effective sample size), the resampling distribution is tempered so diversity survives the early phase of generation.
6. When all particles stop, the final response is the highest-weight particle (`argmax`, default) or sampled.

## 📚 Documentation

In-depth design docs live in [documentation/](documentation/) — including the particle-filtering weight derivation, entropic annealing, and the audio carry mechanism.

## Installation

### Core Installation (Algorithms Only)

```bash
pip install its_hub
```

This includes:
- ✓ `ParticleFiltering` and `EntropicParticleFiltering`
- ✓ `StepGeneration` and `LMOrchestrator`
- ✓ Abstract base classes (`AbstractLanguageModel`, `AbstractOrchestrator`)
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes the OpenAI-compatible language model client:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel` (requires `openai`, `aiohttp`, `backoff`)

### Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
# or using uv:
uv sync --extra dev
```

## Quick Start

**Installation required:** `pip install its_hub[lm]`

```python
import asyncio

from its_hub import (
    EntropicParticleFiltering,
    OpenAICompatibleLanguageModel,
    ParticleFiltering,
    StepGeneration,
)

lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8100/v1",  # vLLM endpoint with logprobs support
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-7B-Instruct",
)

# chunk reasoning into steps on blank lines; stop once the model writes "Answer:"
sg = StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=12)

# Particle filtering with self-certainty weights (mean step logprob)
pf = ParticleFiltering(sg=sg)
result = pf.infer(lm, "What is 6 * 7? Reason step by step.", budget=4)
print(result)  # {"role": "assistant", "content": "..."}

# Entropic particle filtering (entropy signal, tempered resampling)
epf = EntropicParticleFiltering(sg=sg, self_certainty_signal="entropy")
result = epf.infer(lm, "What is 6 * 7? Reason step by step.", budget=4)
print(result)

# Close lm for resource cleanup
asyncio.run(lm.close())
```

### Audio prompts

Pass structured messages instead of a string — audio parts reach the model verbatim at every step:

```python
from its_hub.api.types import ChatMessage

messages = [
    ChatMessage(
        role="user",
        content=[
            {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}},
            {"type": "text", "text": "What instrument is playing? A. piano B. violin"},
        ],
    )
]
result = pf.infer(lm, messages, budget=4)
```

See [benchmarking/mmau_pro/](benchmarking/mmau_pro/) for a complete audio MCQ benchmark (MMAU-Pro on Qwen2.5-Omni).

## Key Features

- 🔬 **Particle filtering algorithms**: PF and EPF, weighted by the generator's own logprobs (self-certainty) — no reward model needed
- 🎧 **Audio/multimodal support**: structured user content (e.g. `input_audio` parts) is carried verbatim through step-by-step generation
- 🚀 **Gateway Integration**: clean abstractions (`AbstractLanguageModel`, `AbstractOrchestrator`) for easy integration with AI gateways
- ⚡ **Async-First**: `ainfer()` is the primary method; `infer()` is a sync wrapper
- 🎯 **Minimal Core**: only 2 dependencies (numpy, typing-extensions) for core install
