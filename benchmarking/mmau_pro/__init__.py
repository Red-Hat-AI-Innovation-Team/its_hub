"""MMAU-Pro MCQ evaluation harness for inference-time scaling (PF/EPF self-certainty).

Evaluates Qwen2.5-Omni (served via vLLM) on the 957 multiple-choice questions of
MMAU-Pro test-mini, comparing a baseline (budget=1) against Particle Filtering and
Entropic Particle Filtering whose particle weights come from the generator's own
self-certainty (no external reward model).
"""
