#!/usr/bin/env python3
"""Test Best-of-N with real models to verify context limits and performance."""

import sys
import os
sys.path.insert(0, '.')

from its_hub.algorithms.bon import BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    import re
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""

def test_real_best_of_n():
    """Test Best-of-N with real models."""
    print("Testing Best-of-N with real models...")
    
    # Initialize language model (connects to vLLM server on GPU 0)
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",
        api_key="NO_API_KEY",
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )
    
    # Initialize reward model on GPU 1 (separate from inference model)
    print("Loading reward model on GPU 1...")
    prm = LocalVllmProcessRewardModel(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device="cuda:1",  # Use GPU 1 for reward model
        aggregation_method="prod"
    )
    
    # Create outcome reward model that uses the process reward model
    class ProcessToOutcomeRewardModel:
        def __init__(self, process_rm):
            self.process_rm = process_rm
            
        def score(self, prompt, responses):
            """Convert process reward to outcome reward by taking final score."""
            if isinstance(responses, list):
                scores = []
                for response in responses:
                    # Get process scores and take the final one as outcome score
                    process_scores = self.process_rm.score(prompt, response)
                    if isinstance(process_scores, list):
                        scores.append(process_scores[-1] if process_scores else 0.0)
                    else:
                        scores.append(process_scores)
                return scores
            else:
                process_scores = self.process_rm.score(prompt, responses)
                if isinstance(process_scores, list):
                    return process_scores[-1] if process_scores else 0.0
                else:
                    return process_scores
    
    orm = ProcessToOutcomeRewardModel(prm)
    
    # Create Best-of-N algorithm
    bon = BestOfN(orm=orm)
    
    # Test with AIME-style problem
    prompt = "Let $a$ be a positive real number such that all the roots of $x^3 + ax^2 + ax + 1 = 0$ are real. Find the smallest possible value of $a$."
    
    print(f"Testing with prompt: {prompt}")
    print("="*80)
    
    # Test different budgets to check context limits
    budgets = [1, 2, 4, 8, 16]
    
    for budget in budgets:
        print(f"\n--- Testing with budget={budget} ---")
        try:
            result = bon.infer(lm, prompt, budget, return_response_only=False)
            
            print(f"Generated {len(result.responses)} responses:")
            for i, (response, score) in enumerate(zip(result.responses, result.scores)):
                marker = " <- SELECTED" if i == result.selected_index else ""
                extracted = extract_boxed(response)
                print(f"  {i+1}. Score: {score:.4f} | Answer: {extracted}{marker}")
                if budget <= 4:  # Only show full responses for small budgets
                    print(f"     Response snippet: {response[:200]}...")
            
            print(f"\nBest response (budget={budget}): {extract_boxed(result.the_one)}")
            
        except Exception as e:
            print(f"❌ Failed with budget {budget}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n✅ Best-of-N testing completed!")

if __name__ == "__main__":
    try:
        test_real_best_of_n()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)