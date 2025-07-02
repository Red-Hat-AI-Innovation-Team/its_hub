#!/usr/bin/env python3
"""Quick test of Planning-Enhanced Particle Filtering on one AIME question."""

import sys
import time
sys.path.insert(0, '.')

from its_hub.algorithms.planning_wrapper import create_planning_particle_filtering
from its_hub.algorithms import ParticleFiltering
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    import re
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""

class ProcessToOutcomeRewardModel:
    """Convert process reward model to outcome reward model."""
    
    def __init__(self, process_rm):
        self.process_rm = process_rm
        
    def score(self, prompt, responses):
        """Convert process reward to outcome reward by aggregating scores."""
        if isinstance(responses, list):
            scores = []
            for response in responses:
                try:
                    process_scores = self.process_rm.score(prompt, response)
                    if isinstance(process_scores, list) and len(process_scores) > 0:
                        final_score = process_scores[-1] if process_scores else 0.0
                    else:
                        final_score = process_scores if process_scores else 0.0
                    scores.append(final_score)
                except Exception as e:
                    print(f"Warning: Reward model scoring failed: {e}")
                    scores.append(0.0)
            return scores
        else:
            try:
                process_scores = self.process_rm.score(prompt, responses)
                if isinstance(process_scores, list) and len(process_scores) > 0:
                    return process_scores[-1]
                else:
                    return process_scores if process_scores else 0.0
            except Exception as e:
                print(f"Warning: Reward model scoring failed: {e}")
                return 0.0

def main():
    """Test Planning-Enhanced Particle Filtering on one AIME question."""
    
    print("🧪 Quick Test: Planning-Enhanced Particle Filtering")
    print("="*60)
    
    # Test problem
    problem = "Find the sum of the roots of the quadratic equation $2x^2 - 7x + 3 = 0$."
    expected_answer = "7/2"  # Sum of roots = -b/a = 7/2
    budget = 8
    
    print(f"Problem: {problem}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Budget: {budget}")
    print()
    
    # Initialize models
    print("Initializing models...")
    
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",
        api_key="NO_API_KEY", 
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )
    
    # Initialize process reward model
    prm = LocalVllmProcessRewardModel(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device="cuda:1",
        aggregation_method="prod"
    )
    
    # Create step generation
    sg = StepGeneration("\\n\\n", 32, r"\\boxed")
    
    print("Models initialized successfully!")
    print()
    
    # Test Vanilla Particle Filtering
    print("--- Testing Vanilla Particle Filtering ---")
    vanilla_start = time.time()
    
    try:
        vanilla_pf = ParticleFiltering(sg, prm)
        vanilla_result = vanilla_pf.infer(lm, problem, budget, return_response_only=False)
        vanilla_time = time.time() - vanilla_start
        vanilla_answer = extract_boxed(vanilla_result.the_one)
        
        print(f"✅ Vanilla PF Result:")
        print(f"  Answer: {vanilla_answer}")
        print(f"  Correct: {vanilla_answer == expected_answer}")
        print(f"  Time: {vanilla_time:.1f}s")
        print(f"  Particles: {len(vanilla_result.all_responses)}")
        
    except Exception as e:
        print(f"❌ Vanilla PF Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Test Planning-Enhanced Particle Filtering
    print("--- Testing Planning-Enhanced Particle Filtering ---")
    planning_start = time.time()
    
    try:
        planning_pf = create_planning_particle_filtering(sg, prm)
        planning_result = planning_pf.infer(lm, problem, budget, return_response_only=False)
        planning_time = time.time() - planning_start
        planning_answer = extract_boxed(planning_result.the_one)
        
        print(f"✅ Planning PF Result:")
        print(f"  Answer: {planning_answer}")
        print(f"  Correct: {planning_answer == expected_answer}")
        print(f"  Time: {planning_time:.1f}s")
        print(f"  Total Responses: {len(planning_result.combined_responses)}")
        print(f"  Approaches Used: {len(planning_result.approaches)}")
        
        print(f"\\n📋 Planning Details:")
        print(f"  Generated Plan: {planning_result.plan[:200]}...")
        print(f"  Parsed Approaches:")
        for i, approach in enumerate(planning_result.approaches, 1):
            budget_for_approach = planning_result.approach_budgets.get(approach, 0)
            print(f"    {i}. {approach} (budget: {budget_for_approach})")
        print(f"  Best Approach: {planning_result.best_approach}")
        
    except Exception as e:
        print(f"❌ Planning PF Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Comparison
    print("--- Comparison ---")
    print(f"Vanilla vs Planning:")
    print(f"  Answer Match: {vanilla_answer == planning_answer}")
    print(f"  Time Overhead: {planning_time - vanilla_time:+.1f}s")
    print(f"  Approach Diversity: {len(planning_result.approaches)} distinct approaches")
    
    if vanilla_answer == expected_answer and planning_answer == expected_answer:
        print(f"✅ Both methods found the correct answer!")
    elif vanilla_answer == expected_answer:
        print(f"⚠️ Only vanilla found the correct answer")
    elif planning_answer == expected_answer:
        print(f"⚠️ Only planning found the correct answer")
    else:
        print(f"❌ Neither method found the correct answer")

if __name__ == "__main__":
    main()