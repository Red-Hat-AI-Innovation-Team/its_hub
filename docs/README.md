# its_hub

[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/its_hub/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub/graph/badge.svg?token=6WD8NB9YPN)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/its_hub)

**its_hub** provides inference-time scaling for LLMs through multiple approaches:

1. **Direct Library Usage** - For Python integration
2. **Inference-as-a-Service (IaaS) API** - OpenAI-compatible HTTP API (⚠️ Alpha)

## What is Inference-Time Scaling?

Inference-time scaling improves LLM performance by using computational resources during inference to generate better responses. Unlike training-time scaling which requires more parameters or training data, inference-time scaling algorithms can improve any pre-trained model's performance by:

- **Generating multiple candidate responses** and selecting the best one
- **Using step-by-step reasoning** with reward models to guide generation
- **Applying probabilistic methods** like particle filtering for better exploration

## Key Features

- 🔬 **Multiple Algorithms**: Particle Filtering, Best-of-N, Beam Search, Self-Consistency
- 🚀 **OpenAI-Compatible API**: Easy integration with existing applications
- 🧮 **Math-Optimized**: Built for mathematical reasoning with specialized prompts and evaluation
- 📊 **Benchmarking Tools**: Compare algorithms on standard datasets like MATH500 and AIME-2024
- ⚡ **Production Ready**: Async generation, concurrency limits, and error handling

## Supported Algorithms

| Algorithm | Type | Best For | Budget Interpretation |
|-----------|------|----------|----------------------|
| **Self-Consistency** | Sampling | Consistent reasoning tasks | Number of parallel generations |
| **Best-of-N** | Sampling + Ranking | Quality over speed | Number of candidates to generate |
| **Beam Search** | Structured Search | Step-by-step problems | Total generations ÷ beam width |
| **Particle Filtering** | Probabilistic | Complex reasoning | Number of particles to maintain |