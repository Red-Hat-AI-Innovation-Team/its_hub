# A Python library for inference-time scaling LLMs

Example: Using the particle filtering from `[1]` for inference-time scaling

```python
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.algorithms import ParticleFiltering
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# NOTE launched via `CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --dtype float16`
lm = OpenAICompatibleLanguageModel(
    endpoint="http://0.0.0.0:8000/v1", 
    api_key="NO_API_KEY", 
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct", 
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
)
prompt = r"Let $a$ be a positive real number such that all the roots of \[x^3 + ax^2 + ax + 1 = 0\]are real. Find the smallest possible value of $a.$" # question from MATH500
budget = 8

sg = StepGeneration(r"\n\n", 32, r"\boxed")
prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B", device="cuda:1", aggregation_method="prod"
)
scaling_alg = ParticleFiltering(sg, prm)

scaling_alg.infer(lm, prompt, budget, show_progress=True) # => gives output
```

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. “A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods”, 2025.

## Installation

Latest release from PyPI

```sh
pip install its_hub
```

Latest main branch
```sh
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
```

## Benchmark

There is a script at `scripts/benchmark.py` that can be used to benchmark inference-time scaling algorithms.
The CLI of the script is self-contained so the usage can be checked via `python scripts/benchmark.py --help`.
Example command:
```
python scripts/benchmark.py --benchmark aime-2024 --model_name Qwen/Qwen2.5-Math-1.5B-Instruct --alg particle-filtering --rm_device cuda:1 --endpoint http://0.0.0.0:8000/v1 --shuffle_seed 1110 --does_eval --budgets 1,2,4,8,16,32,64 --rm_agg_method model
```

## Development

```sh
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
pytest tests
```