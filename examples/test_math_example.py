#!/usr/bin/env python3
"""
Example script demonstrating the use of its_hub for math problem solving.
This script tests the Qwen math model with various mathematical problems
using particle filtering for improved solution quality.

Requirements:
    pip install its_hub[experimental]

This example uses reward-hub integration for process reward models.
"""

import asyncio
import os

from its_hub import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_gibbs import ParticleFiltering
from its_hub.core.reward_models.local_vllm_prm import LocalVllmProcessRewardModel
from its_hub.core.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT


def main():
    # Get GPU ID from environment variable or default to 0
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    # Initialize the language model
    # Note: The endpoint port (8100) must match the port used when starting the vLLM server
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",  # Make sure this matches your vLLM server port
        api_key="NO_API_KEY",
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )

    # Test prompts
    test_prompts = [
        "What is 2+2? Show your steps.",
        "Solve the quadratic equation x^2 + 5x + 6 = 0. Show your steps.",
        "Find the derivative of f(x) = x^2 + 3x + 2. Show your steps.",
        "Let a be a positive real number such that all the roots of x^3 + ax^2 + ax + 1 = 0 are real. Find the smallest possible value of a.",
    ]

    # Initialize step generation and reward model
    sg = StepGeneration(step_token="\n\n", max_steps=32, stop_token=r"\boxed")
    prm = LocalVllmProcessRewardModel(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device=f"cuda:{gpu_id}",  # Use the same GPU as the vLLM server
        aggregation_method="prod",
    )
    scaling_alg = ParticleFiltering(sg, prm)

    # Run tests
    print("Testing Qwen Math Model with different approaches...")
    print(f"Using GPU {gpu_id} with memory optimization settings\n")

    for prompt in test_prompts:
        print(f"\nTesting: {prompt}")
        print("Response:", scaling_alg.infer(lm, prompt, budget=8))

    # Close lm for resource cleanup
    asyncio.run(lm.close())


if __name__ == "__main__":
    main()
