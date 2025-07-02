#!/usr/bin/env python3
"""Simple test of Planning-Enhanced Particle Filtering using existing setup."""

import sys
import time
import json
sys.path.insert(0, '.')

from its_hub.algorithms.planning_wrapper import create_planning_particle_filtering
from its_hub.algorithms import ParticleFiltering
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    import re
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""

# Mock reward model for quick testing
class MockProcessRewardModel:
    def __init__(self):
        import random
        self.random = random
    
    def score(self, prompt, response):
        # Handle both single response and list of responses
        if isinstance(response, list):
            return [self._score_single(r) for r in response]
        else:
            return self._score_single(response)
    
    def _score_single(self, response):
        # Return dummy scores based on response length and keywords
        base_score = self.random.uniform(0.3, 0.9)
        response_str = str(response)
        if "step" in response_str.lower() or "solve" in response_str.lower():
            base_score += 0.1
        if "\\boxed" in response_str:
            base_score += 0.2
        return min(base_score, 1.0)

def main():
    """Test Planning-Enhanced Particle Filtering on one simple question."""
    
    print("🧪 Simple Test: Planning-Enhanced Particle Filtering")
    print("="*60)
    
    # Simple test problem
    problem = "Solve for x: 2x + 6 = 14"
    expected_answer = "4"
    budget = 6
    
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
    
    # Use mock reward model for simplicity
    prm = MockProcessRewardModel()
    
    # Create step generation
    sg = StepGeneration("\\n\\n", 16, r"\\boxed")
    
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
        print(f"  Answer: '{vanilla_answer}'")
        print(f"  Correct: {vanilla_answer == expected_answer}")
        print(f"  Time: {vanilla_time:.1f}s")
        # Try to get particle count from available attributes
        if hasattr(vanilla_result, 'all_responses'):
            print(f"  Total particles: {len(vanilla_result.all_responses)}")
        elif hasattr(vanilla_result, 'responses'):
            print(f"  Total responses: {len(vanilla_result.responses)}")
        else:
            print(f"  Result type: {type(vanilla_result)}")
        
        # Show sample response
        print(f"  Sample response: {vanilla_result.the_one[:100]}...")
        
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
        print(f"  Answer: '{planning_answer}'")
        print(f"  Correct: {planning_answer == expected_answer}")
        print(f"  Time: {planning_time:.1f}s")
        print(f"  Total responses: {len(planning_result.combined_responses)}")
        print(f"  Approaches used: {len(planning_result.approaches)}")
        
        print(f"\\n📋 Planning Details:")
        print(f"  Generated plan (first 150 chars): {planning_result.plan[:150]}...")
        print(f"  Parsed approaches:")
        for i, approach in enumerate(planning_result.approaches, 1):
            budget_for_approach = planning_result.approach_budgets.get(approach, 0)
            print(f"    {i}. {approach[:60]}{'...' if len(approach) > 60 else ''} (budget: {budget_for_approach})")
        print(f"  Best approach: {planning_result.best_approach[:50]}{'...' if len(planning_result.best_approach) > 50 else ''}")
        
        # Show sample from best approach
        best_result = planning_result.approach_results[planning_result.best_approach]
        print(f"  Best approach sample: {best_result.the_one[:100]}...")
        
    except Exception as e:
        print(f"❌ Planning PF Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # Comparison
    print("--- Comparison Summary ---")
    print(f"Results:")
    print(f"  Vanilla:  '{vanilla_answer}' ({'✅' if vanilla_answer == expected_answer else '❌'})")
    print(f"  Planning: '{planning_answer}' ({'✅' if planning_answer == expected_answer else '❌'})")
    
    print(f"Performance:")
    print(f"  Answer match: {vanilla_answer == planning_answer}")
    print(f"  Time overhead: {planning_time - vanilla_time:+.1f}s")
    print(f"  Planning efficiency: {len(planning_result.approaches)} approaches with budget allocation")
    
    if vanilla_answer == expected_answer and planning_answer == expected_answer:
        print(f"\\n🎉 SUCCESS: Both methods found the correct answer!")
    elif planning_answer == expected_answer:
        print(f"\\n⭐ Planning method found the correct answer!")
    elif vanilla_answer == expected_answer:
        print(f"\\n📊 Vanilla method found the correct answer")
    else:
        print(f"\\n🤔 Neither method found the expected answer, but test completed successfully")
    
    print(f"\\n✅ Planning-Enhanced Particle Filtering test completed!")

if __name__ == "__main__":
    main()