# `its-hub`: A Python library for inference-time scaling

**its-hub** is a Python library for inference-time scaling of LLMs — generate multiple candidate responses and select the best using voting, scoring, or search algorithms. Developed by the [Red Hat AI Innovation Team](https://ai-innovation.team).

<p align="center">
  <a href="https://pypi.org/project/its-hub/">
    <img src="https://img.shields.io/pypi/v/its-hub?style=for-the-badge" alt="PyPI version">
  </a>
  <a href="https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/Red-Hat-AI-Innovation-Team/its_hub/tests.yaml?style=for-the-badge&label=tests" alt="Tests">
  </a>
  <a href="https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub">
    <img src="https://img.shields.io/codecov/c/gh/Red-Hat-AI-Innovation-Team/its_hub?style=for-the-badge" alt="Coverage">
  </a>
  <a href="https://github.com/Red-Hat-AI-Innovation-Team/its_hub/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/its_hub?style=for-the-badge" alt="License">
  </a>
  <a href="https://ai-innovation.team/its_hub">
    <img src="https://img.shields.io/badge/docs-ai--innovation.team-blue?style=for-the-badge" alt="Documentation">
  </a>
</p>

**New to its-hub?** Read the full documentation at **[ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**

## Algorithm Support Matrix

| Algorithm | Strategy | Budget | Reward Model | Status |
|-----------|----------|--------|--------------|--------|
| **Self-Consistency** | Generate N, majority vote | N parallel generations | None (voting) | ✅ Implemented |
| **Best-of-N** | Generate N, rank by score | N parallel generations | Outcome RM or LLM Judge | ✅ Implemented |
| **Beam Search** | Step-by-step with beam width | depth = budget / beam_width | Process RM | ✅ Implemented |
| **Particle Filtering** | Probabilistic resampling | Number of particles | Process RM | ✅ Implemented |

All algorithms share the same interface: `alg.infer(lm, prompt, budget=N)`

## Implemented Algorithms

### [Self-Consistency](./its_hub/algorithms/self_consistency.py)

Generate N responses and vote on the most common answer. No reward model needed — works out of the box:

```python
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import SelfConsistency

lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1",
    api_key="your-api-key",
    model_name="Qwen/Qwen2.5-32B-Instruct",
)

alg = SelfConsistency()
result = alg.infer(lm, "What is the integral of x^2?", budget=8)
print(result)  # Most common answer across 8 generations
```

### [Best-of-N](./its_hub/algorithms/bon.py)

Generate N responses, score each with a reward model, and pick the best. Supports local reward models or LLM-as-judge:

**Installation:** `pip install its_hub` (core)

```python
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

judge = LLMJudgeRewardModel(
    model="gpt-4o-mini",
    criterion="overall_quality",
    judge_type="groupwise",
    api_key="your-api-key",
)
alg = BestOfN(judge)

result = alg.infer(lm, "Explain quantum entanglement in simple terms", budget=4)
print(result)  # Highest-scoring response
```

### [Particle Filtering](./its_hub/algorithms/particle_gibbs.py)

Step-by-step reasoning with probabilistic resampling — prunes weak reasoning paths as it goes:

**Installation:** `pip install its_hub[prm]`

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8100/v1",
    api_key="NO_API_KEY",
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
)

sg = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod",
)
alg = ParticleFiltering(sg, prm)

result = alg.infer(lm, "Solve x^2 + 5x + 6 = 0", budget=8)
print(result)  # Best reasoning path
```

## Key Features

- **Multiple Algorithms** — Self-Consistency, Best-of-N, Beam Search, Particle Filtering
- **OpenAI-Compatible API** — works with vLLM, OpenAI, and any compatible endpoint
- **IaaS Server** — FastAPI service with OpenAI-compatible chat completions + `budget` parameter
- **Multi-Provider Support** — AWS Bedrock, Google Vertex, and more via LiteLLM
- **Async-First** — concurrent generation with configurable limits and retry logic
- **Tool Calling** — inference-time scaling with function/tool call support and voting strategies
- **Benchmarking** — compare algorithms on MATH500 and AIME-2024 datasets

## Demo

See the library in action with a walkthrough of inference-time scaling algorithms:

<div align="center">
  <a href="https://www.youtube.com/watch?v=qaXyvmR-YBU">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://img.youtube.com/vi/qaXyvmR-YBU/maxresdefault.jpg">
      <source media="(prefers-color-scheme: light)" srcset="https://img.youtube.com/vi/qaXyvmR-YBU/maxresdefault.jpg">
      <img src="https://img.youtube.com/vi/qaXyvmR-YBU/maxresdefault.jpg" alt="its-hub demo walkthrough" width="800">
    </picture>
  </a>
</div>

Try it in your browser: [https://red.ht/its-hub-demo](https://red.ht/its-hub-demo) | [Demo setup instructions](https://github.com/lukeinglis/its_hub_demo/blob/main/demo_ui/README.md)

## Installation

Choose the installation option based on which algorithms you need:

```bash
# Core — Self-Consistency, Best-of-N with LLM Judge, OpenAI-compatible models
pip install its_hub

# Process Reward Models — adds Particle Filtering, Beam Search, local PRM support
pip install its_hub[prm]

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

## Coding Agent Plugin

its-hub is available as a plugin for five major coding agents, bringing inference-time scaling directly into your coding workflow.

<details>
<summary><strong>Claude Code</strong></summary>

**Via org marketplace** (recommended — includes all Red Hat AI plugins):
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
/plugin install its-hub@Red-Hat-AI-Innovation-Team/plugins
```

**Via this repo directly:**
```
/plugin marketplace add Red-Hat-AI-Innovation-Team/its_hub
/plugin install its-hub@Red-Hat-AI-Innovation-Team/its_hub
```

**From a local clone:**
```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
/plugin marketplace add /path/to/its_hub
```
</details>

<details>
<summary><strong>Cursor</strong></summary>

Clone the repo and open it — Cursor discovers the plugin via `.cursor-plugin/plugin.json` automatically.
</details>

<details>
<summary><strong>Gemini CLI</strong></summary>

```bash
gemini extensions install https://github.com/Red-Hat-AI-Innovation-Team/its_hub
```
</details>

<details>
<summary><strong>Codex CLI</strong></summary>

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git ~/.codex/its-hub
mkdir -p ~/.agents/skills
ln -s ~/.codex/its-hub/skills ~/.agents/skills/its-hub
```

Restart Codex to discover the skills. See `.codex-plugin/INSTALL.md` for full instructions.
</details>

<details>
<summary><strong>OpenCode</strong></summary>

Add to your `opencode.json`:

```json
{
  "plugin": ["its-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git"]
}
```

Restart OpenCode. See `.opencode-plugin/INSTALL.md` for full instructions.
</details>

### After Installing

Run `/its-setup` (or invoke the `setup-guide` skill) to configure your model endpoint and algorithm.

| Command | Description |
|---|---|
| `/its-setup` | Guided first-time configuration |
| `/its-scale <prompt>` | Run inference-time scaling on a single prompt |
| `/its-scale-batch <file>` | Batch scaling from a JSONL/CSV/TXT file |
| `/its-server start\|stop\|status` | Manage the IaaS server lifecycle |

## Getting Started

For comprehensive documentation, tutorials, and API reference, see the [examples directory](./notebooks/) or visit:

**[https://ai-innovation.team/its_hub](https://ai-innovation.team/its_hub)**
