# A Python library for inference-time scaling LLMs

Example of using the particle filtering approach from `[1]` for inference-time scaling

```python
from inference_time_scaling import ParticleFiltering

llm = ...
prompt = ...
budget = 16

prm = ...
scaling_method = ParticleFiltering(prm, ...)

scaling_method.infer(llm, prompt, budget) # => gives output
```

`[1]`: Isha Puri, Shivchander Sudalairaj, Guangxuan Xu, Kai Xu, Akash Srivastava. “A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods”, 2025.

## Installation

```sh
pip install git+https://github.com/Red-Hat-AI-Innovation-Team/inference_time_scaling.git
```
