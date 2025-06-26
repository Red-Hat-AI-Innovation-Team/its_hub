#!/usr/bin/env python3
"""Test to verify GPU separation between inference and reward models."""

import os
import torch
import sys
sys.path.insert(0, '.')

from its_hub.algorithms.bon import BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def test_gpu_separation():
    """Test GPU separation for inference and reward models."""
    
    print("Testing GPU separation...")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Initialize language model (connects to vLLM server on GPU 0)
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",
        api_key="NO_API_KEY", 
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )
    
    # Force reward model on GPU 1 by setting environment variable
    print("Loading reward model on GPU 1...")
    
    # Set CUDA_VISIBLE_DEVICES to force reward model to use GPU 1
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    try:
        # This should force the reward model to GPU 1
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        
        prm = LocalVllmProcessRewardModel(
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device="cuda:0",  # This will be GPU 1 due to CUDA_VISIBLE_DEVICES
            aggregation_method="prod"
        )
        
        print("✅ Reward model loaded")
        
        # Test a simple generation
        prompt = "What is 2+2?"
        response = lm.generate(prompt)
        print(f"LM Response: {response[:100]}...")
        
        # Test reward model
        score = prm.score(prompt, response)
        print(f"Reward score: {score}")
        
    finally:
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    print("✅ GPU separation test completed")

if __name__ == "__main__":
    test_gpu_separation()