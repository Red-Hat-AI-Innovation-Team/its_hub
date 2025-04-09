# A Python library for inference-time scaling LLMs

Example of using the particle filtering approach from `[1]` for inference-time scaling

```python
from inference_time_scaling import ParticleFiltering

lm = ...
prompt = ...
budget = 16

prm = ...
scaling_alg = ParticleFiltering(prm, ...)

scaling_alg.infer(lm, prompt, budget) # => gives output
```

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. “A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods”, 2025.

## Installation

```sh
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/inference_time_scaling.git
```
