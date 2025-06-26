#!/usr/bin/env python3
"""Compare Planning-Enhanced Best-of-N vs Vanilla Best-of-N on AIME problems."""

import sys
import time
import json
from typing import Dict, Any, List
sys.path.insert(0, '.')

from its_hub.algorithms.bon import BestOfN
from its_hub.algorithms.planning_bon import PlanningBestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    import re
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""


class SimpleOutcomeRewardModel:
    """Simple outcome reward model that scores based on answer extraction."""
    
    def score(self, prompt: str, responses):
        """Score responses based on whether they contain a boxed answer."""
        if isinstance(responses, list):
            scores = []
            for response in responses:
                # Basic scoring: higher score if answer is extracted
                extracted = extract_boxed(response)
                if extracted:
                    # Simple heuristic: longer explanations get slightly higher scores
                    base_score = 0.5
                    length_bonus = min(0.4, len(response) / 2000)  # Bonus up to 0.4
                    scores.append(base_score + length_bonus)
                else:
                    scores.append(0.1)  # Low score if no answer found
            return scores
        else:
            extracted = extract_boxed(responses)
            if extracted:
                base_score = 0.5
                length_bonus = min(0.4, len(responses) / 2000)
                return base_score + length_bonus
            else:
                return 0.1


def test_single_problem(
    lm: OpenAICompatibleLanguageModel,
    orm: SimpleOutcomeRewardModel,
    problem: str,
    budget: int,
    problem_name: str = "Test Problem"
) -> Dict[str, Any]:
    """Test both algorithms on a single problem."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {problem_name}")
    print(f"Budget: {budget}")
    print(f"{'='*80}")
    
    results = {}
    
    # Test Vanilla Best-of-N
    print(f"\n--- Running Vanilla Best-of-N (budget={budget}) ---")
    vanilla_start = time.time()
    
    vanilla_bon = BestOfN(orm=orm)
    vanilla_result = vanilla_bon.infer(lm, problem, budget, return_response_only=False)
    
    vanilla_time = time.time() - vanilla_start
    vanilla_answer = extract_boxed(vanilla_result.the_one)
    vanilla_best_score = max(vanilla_result.scores)
    vanilla_avg_score = sum(vanilla_result.scores) / len(vanilla_result.scores)
    
    print(f"Vanilla Best-of-N Results:")
    print(f"  Time: {vanilla_time:.2f}s")
    print(f"  Best Answer: {vanilla_answer}")
    print(f"  Best Score: {vanilla_best_score:.4f}")
    print(f"  Avg Score: {vanilla_avg_score:.4f}")
    print(f"  Responses: {len(vanilla_result.responses)}")
    
    results['vanilla'] = {
        'answer': vanilla_answer,
        'best_score': vanilla_best_score,
        'avg_score': vanilla_avg_score,
        'time': vanilla_time,
        'num_responses': len(vanilla_result.responses),
        'all_scores': vanilla_result.scores
    }
    
    # Test Planning-Enhanced Best-of-N
    print(f"\n--- Running Planning-Enhanced Best-of-N (budget={budget}) ---")
    planning_start = time.time()
    
    planning_bon = PlanningBestOfN(orm=orm)
    planning_result = planning_bon.infer(lm, problem, budget, return_response_only=False)
    
    planning_time = time.time() - planning_start
    planning_answer = extract_boxed(planning_result.the_one)
    planning_best_score = max(planning_result.all_scores)
    planning_avg_score = sum(planning_result.all_scores) / len(planning_result.all_scores)
    
    print(f"Planning-Enhanced Best-of-N Results:")
    print(f"  Time: {planning_time:.2f}s")
    print(f"  Best Answer: {planning_answer}")
    print(f"  Best Score: {planning_best_score:.4f}")
    print(f"  Avg Score: {planning_avg_score:.4f}")
    print(f"  Total Responses: {len(planning_result.all_responses)}")
    print(f"  Approaches: {len(planning_result.approaches)}")
    
    print(f"\n  Generated Plan:")
    print(f"  {planning_result.plan}")
    
    print(f"\n  Extracted Approaches:")
    for i, approach in enumerate(planning_result.approaches):
        budget_used = planning_result.approach_budgets[approach]
        approach_scores = planning_result.approach_scores[approach]
        best_approach_score = max(approach_scores) if approach_scores else 0
        print(f"    {i+1}. {approach} (budget: {budget_used}, best score: {best_approach_score:.4f})")
    
    results['planning'] = {
        'answer': planning_answer,
        'best_score': planning_best_score,
        'avg_score': planning_avg_score,
        'time': planning_time,
        'num_responses': len(planning_result.all_responses),
        'plan': planning_result.plan,
        'approaches': planning_result.approaches,
        'approach_budgets': planning_result.approach_budgets,
        'all_scores': planning_result.all_scores
    }
    
    # Comparison
    print(f"\n--- Comparison ---")
    print(f"Answer Match: {vanilla_answer == planning_answer}")
    print(f"Score Improvement: {planning_best_score - vanilla_best_score:+.4f}")
    print(f"Time Overhead: {planning_time - vanilla_time:+.2f}s ({((planning_time/vanilla_time - 1)*100):+.1f}%)")
    
    return results


def main():
    """Run the comparison experiment."""
    
    print("Planning-Enhanced vs Vanilla Best-of-N Comparison")
    print("="*60)
    
    # Initialize language model
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",
        api_key="NO_API_KEY",
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )
    
    # Initialize simple outcome reward model
    orm = SimpleOutcomeRewardModel()
    
    # Test problems (AIME-style)
    test_problems = [
        {
            "name": "AIME Cubic Roots",
            "problem": "Let $a$ be a positive real number such that all the roots of $x^3 + ax^2 + ax + 1 = 0$ are real. Find the smallest possible value of $a$."
        },
        {
            "name": "Simple Quadratic",
            "problem": "Find the sum of the roots of the quadratic equation $2x^2 - 7x + 3 = 0$."
        }
    ]
    
    # Test budgets
    budgets = [4, 8, 16]
    
    all_results = {}
    
    for problem_info in test_problems:
        problem_name = problem_info["name"]
        problem = problem_info["problem"]
        
        all_results[problem_name] = {}
        
        for budget in budgets:
            try:
                result = test_single_problem(lm, orm, problem, budget, problem_name)
                all_results[problem_name][f"budget_{budget}"] = result
            except Exception as e:
                print(f"❌ Failed {problem_name} with budget {budget}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    with open('planning_vs_vanilla_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("✅ Comparison completed! Results saved to 'planning_vs_vanilla_results.json'")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()