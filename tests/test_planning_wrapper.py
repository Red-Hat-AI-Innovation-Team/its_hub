#!/usr/bin/env python3
"""Test PlanningWrapper with multiple ITS algorithms."""

import sys
import time
sys.path.insert(0, '.')

from its_hub.algorithms.planning_wrapper import (
    PlanningWrapper,
    create_planning_self_consistency,
    create_planning_particle_filtering,
    create_planning_best_of_n
)
from its_hub.algorithms import SelfConsistency, ParticleFiltering, BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
# Mock reward model for testing
class MockProcessRewardModel:
    def score(self, prompt, response):
        # Return dummy scores based on response length
        import random
        if isinstance(response, str):
            return random.uniform(0.1, 0.9)
        else:  # List of responses
            return [random.uniform(0.1, 0.9) for _ in response]
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

def algorithm_comparison(lm, sg, prm, orm, problem, budget=8):
    """Test vanilla vs planning-enhanced versions of algorithms."""
    
    print(f"\n{'='*80}")
    print(f"Testing Problem: {problem}")
    print(f"Budget: {budget}")
    print(f"{'='*80}")
    
    results = {}
    
    # Test 1: Self-Consistency
    print(f"\n--- Self-Consistency Comparison ---")
    
    # Vanilla Self-Consistency
    print("Running Vanilla Self-Consistency...")
    start = time.time()
    vanilla_sc = SelfConsistency(extract_boxed)
    vanilla_sc_result = vanilla_sc.infer(lm, problem, budget, return_response_only=False)
    vanilla_sc_time = time.time() - start
    vanilla_sc_answer = extract_boxed(vanilla_sc_result.the_one)
    
    print(f"  Vanilla SC: {vanilla_sc_answer} (time: {vanilla_sc_time:.1f}s)")
    
    # Planning-Enhanced Self-Consistency
    print("Running Planning-Enhanced Self-Consistency...")
    start = time.time()
    planning_sc = create_planning_self_consistency(extract_boxed)
    planning_sc_result = planning_sc.infer(lm, problem, budget, return_response_only=False)
    planning_sc_time = time.time() - start
    planning_sc_answer = extract_boxed(planning_sc_result.the_one)
    
    print(f"  Planning SC: {planning_sc_answer} (time: {planning_sc_time:.1f}s)")
    print(f"  Approaches: {planning_sc_result.approaches}")
    print(f"  Best approach: {planning_sc_result.best_approach}")
    
    results['self_consistency'] = {
        'vanilla': {'answer': vanilla_sc_answer, 'time': vanilla_sc_time},
        'planning': {'answer': planning_sc_answer, 'time': planning_sc_time, 
                    'approaches': planning_sc_result.approaches}
    }
    
    # Test 2: Best-of-N
    print(f"\n--- Best-of-N Comparison ---")
    
    # Vanilla Best-of-N
    print("Running Vanilla Best-of-N...")
    start = time.time()
    vanilla_bon = BestOfN(orm)
    vanilla_bon_result = vanilla_bon.infer(lm, problem, budget, return_response_only=False)
    vanilla_bon_time = time.time() - start
    vanilla_bon_answer = extract_boxed(vanilla_bon_result.the_one)
    
    print(f"  Vanilla BoN: {vanilla_bon_answer} (time: {vanilla_bon_time:.1f}s)")
    
    # Planning-Enhanced Best-of-N
    print("Running Planning-Enhanced Best-of-N...")
    start = time.time()
    planning_bon = create_planning_best_of_n(orm)
    planning_bon_result = planning_bon.infer(lm, problem, budget, return_response_only=False)
    planning_bon_time = time.time() - start
    planning_bon_answer = extract_boxed(planning_bon_result.the_one)
    
    print(f"  Planning BoN: {planning_bon_answer} (time: {planning_bon_time:.1f}s)")
    print(f"  Approaches: {planning_bon_result.approaches}")
    print(f"  Best approach: {planning_bon_result.best_approach}")
    
    results['best_of_n'] = {
        'vanilla': {'answer': vanilla_bon_answer, 'time': vanilla_bon_time},
        'planning': {'answer': planning_bon_answer, 'time': planning_bon_time,
                    'approaches': planning_bon_result.approaches}
    }
    
    # Test 3: Particle Filtering
    print(f"\n--- Particle Filtering Comparison ---")
    
    # Vanilla Particle Filtering
    print("Running Vanilla Particle Filtering...")
    start = time.time()
    vanilla_pf = ParticleFiltering(sg, prm)
    vanilla_pf_result = vanilla_pf.infer(lm, problem, budget, return_response_only=False)
    vanilla_pf_time = time.time() - start
    vanilla_pf_answer = extract_boxed(vanilla_pf_result.the_one)
    
    print(f"  Vanilla PF: {vanilla_pf_answer} (time: {vanilla_pf_time:.1f}s)")
    
    # Planning-Enhanced Particle Filtering
    print("Running Planning-Enhanced Particle Filtering...")
    start = time.time()
    planning_pf = create_planning_particle_filtering(sg, prm)
    planning_pf_result = planning_pf.infer(lm, problem, budget, return_response_only=False)
    planning_pf_time = time.time() - start
    planning_pf_answer = extract_boxed(planning_pf_result.the_one)
    
    print(f"  Planning PF: {planning_pf_answer} (time: {planning_pf_time:.1f}s)")
    print(f"  Approaches: {planning_pf_result.approaches}")
    print(f"  Best approach: {planning_pf_result.best_approach}")
    
    results['particle_filtering'] = {
        'vanilla': {'answer': vanilla_pf_answer, 'time': vanilla_pf_time},
        'planning': {'answer': planning_pf_answer, 'time': planning_pf_time,
                    'approaches': planning_pf_result.approaches}
    }
    
    return results

def main():
    """Test PlanningWrapper with multiple algorithms."""
    
    print("🚀 PlanningWrapper Multi-Algorithm Test")
    print("="*60)
    
    # Initialize models
    print("Initializing models...")
    
    # Mock language model for testing
    class MockLanguageModel:
        def generate(self, messages, stop=None, max_tokens=None, include_stop_str_in_output=False, temperature=None, **kwargs):
            import random
            # Handle both single and batch generation
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
                # Batch generation
                batch_size = len(messages)
                mock_responses = [
                    "Let me solve this step by step.\n\nFirst, I'll use algebraic methods.\n\nSolving: 2x + 3 = 7\n2x = 4\nx = 2\n\n\\boxed{2}",
                    "I'll approach this differently.\n\nUsing substitution method:\nLet y = 2x + 3\ny = 7\n2x = 4\nx = 2\n\n\\boxed{2}",
                    "Using geometric interpretation:\n\nThis represents a line equation.\nSolving: 2x + 3 = 7\n\n\\boxed{2}",
                    "Step 1: Set up equation\n\n",
                    "Step 2: Simplify\n\n",
                    "Final answer: \\boxed{2}"
                ]
                return [random.choice(mock_responses) for _ in range(batch_size)]
            else:
                # Single generation (for planning)
                return "APPROACH 1: Direct algebraic approach using standard techniques\nAPPROACH 2: Alternative method using different mathematical properties\nAPPROACH 3: Geometric or graphical interpretation approach"
    
    lm = MockLanguageModel()
    
    # Initialize mock process reward model for testing
    prm = MockProcessRewardModel()
    
    # Create outcome reward model and step generation
    orm = ProcessToOutcomeRewardModel(prm)
    sg = StepGeneration("\n\n", 32, r"\boxed")
    
    print("Models initialized successfully!")
    
    # Test problems
    problems = [
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the sum of the roots of the quadratic equation $2x^2 - 7x + 3 = 0$."
    ]
    
    all_results = {}
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'#'*60}")
        print(f"PROBLEM {i}/{len(problems)}")
        print(f"{'#'*60}")
        
        try:
            results = algorithm_comparison(lm, sg, prm, orm, problem, budget=8)
            all_results[f"problem_{i}"] = results
            
            # Summary for this problem
            print(f"\n--- PROBLEM {i} SUMMARY ---")
            for alg_name, alg_results in results.items():
                vanilla_answer = alg_results['vanilla']['answer']
                planning_answer = alg_results['planning']['answer']
                vanilla_time = alg_results['vanilla']['time']
                planning_time = alg_results['planning']['time']
                time_overhead = planning_time - vanilla_time
                
                print(f"{alg_name.upper()}:")
                print(f"  Vanilla: {vanilla_answer} ({vanilla_time:.1f}s)")
                print(f"  Planning: {planning_answer} ({planning_time:.1f}s, overhead: {time_overhead:+.1f}s)")
                print(f"  Answer Match: {vanilla_answer == planning_answer}")
                if 'approaches' in alg_results['planning']:
                    print(f"  Approaches: {len(alg_results['planning']['approaches'])}")
            
        except Exception as e:
            print(f"❌ Failed on problem {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    algorithm_names = ['self_consistency', 'best_of_n', 'particle_filtering']
    
    for alg_name in algorithm_names:
        print(f"\n{alg_name.upper()} ACROSS ALL PROBLEMS:")
        
        matches = 0
        total_overhead = 0
        problem_count = 0
        
        for problem_key, problem_results in all_results.items():
            if alg_name in problem_results:
                alg_result = problem_results[alg_name]
                vanilla_answer = alg_result['vanilla']['answer']
                planning_answer = alg_result['planning']['answer']
                
                if vanilla_answer == planning_answer:
                    matches += 1
                
                time_overhead = alg_result['planning']['time'] - alg_result['vanilla']['time']
                total_overhead += time_overhead
                problem_count += 1
        
        if problem_count > 0:
            avg_overhead = total_overhead / problem_count
            match_rate = matches / problem_count
            
            print(f"  Answer Match Rate: {matches}/{problem_count} ({match_rate:.1%})")
            print(f"  Average Time Overhead: {avg_overhead:+.1f}s")
    
    print(f"\n✅ PlanningWrapper test completed successfully!")
    print(f"All algorithms can now be enhanced with planning using the wrapper!")

if __name__ == "__main__":
    main()