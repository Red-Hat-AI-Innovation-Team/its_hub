#!/usr/bin/env python3
"""
AIME2024 Experiment: Planning-Enhanced Best-of-N vs Vanilla Best-of-N
Run comparison on 5 AIME questions with proper reward model evaluation.
"""

import sys
import time
import json
import random
from typing import Dict, Any, List
import datasets
sys.path.insert(0, '.')

from its_hub.algorithms.bon import BestOfN
from its_hub.algorithms.planning_bon import PlanningBestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    import re
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""

def load_aime_questions(num_questions: int = 5, seed: int = 42) -> List[Dict[str, Any]]:
    """Load a subset of AIME2024 questions."""
    print(f"Loading {num_questions} AIME2024 questions...")
    
    # Load AIME2024 dataset
    ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
    
    # Process column names (following benchmark.py pattern)
    old_column_names = ds.column_names
    ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
    ds = ds.rename_column('id', 'unique_id')
    ds = ds.cast_column('answer', datasets.Value('string'))
    ds = ds.remove_columns(old_column_names)
    
    # Convert to list and sample
    all_questions = list(ds)
    random.seed(seed)
    selected_questions = random.sample(all_questions, min(num_questions, len(all_questions)))
    
    print(f"Selected {len(selected_questions)} questions:")
    for i, q in enumerate(selected_questions):
        print(f"  {i+1}. ID: {q['unique_id']} - {q['problem'][:100]}...")
    
    return selected_questions

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
                    # Get process scores and aggregate them
                    process_scores = self.process_rm.score(prompt, response)
                    if isinstance(process_scores, list) and len(process_scores) > 0:
                        # Use the final score as the outcome score
                        final_score = process_scores[-1] if process_scores else 0.0
                    else:
                        final_score = process_scores if process_scores else 0.0
                    scores.append(final_score)
                except Exception as e:
                    print(f"Warning: Reward model scoring failed for response: {e}")
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

def test_algorithms_on_question(
    lm: OpenAICompatibleLanguageModel,
    orm: ProcessToOutcomeRewardModel,
    question: Dict[str, Any],
    budgets: List[int]
) -> Dict[str, Any]:
    """Test both algorithms on a single AIME question."""
    
    problem = question['problem']
    true_answer = question['answer']
    question_id = question['unique_id']
    
    print(f"\n{'='*80}")
    print(f"Question ID: {question_id}")
    print(f"Problem: {problem}")
    print(f"True Answer: {true_answer}")
    print(f"{'='*80}")
    
    results = {
        'question_id': question_id,
        'problem': problem,
        'true_answer': true_answer,
        'results': {}
    }
    
    for budget in budgets:
        print(f"\n--- BUDGET {budget} ---")
        budget_results = {}
        
        # Test Vanilla Best-of-N
        print(f"Running Vanilla Best-of-N (budget={budget})...")
        vanilla_start = time.time()
        
        try:
            vanilla_bon = BestOfN(orm=orm)
            vanilla_result = vanilla_bon.infer(lm, problem, budget, return_response_only=False)
            vanilla_time = time.time() - vanilla_start
            vanilla_answer = extract_boxed(vanilla_result.the_one)
            vanilla_correct = vanilla_answer == true_answer
            vanilla_best_score = max(vanilla_result.scores) if vanilla_result.scores else 0.0
            vanilla_avg_score = sum(vanilla_result.scores) / len(vanilla_result.scores) if vanilla_result.scores else 0.0
            
            print(f"  Vanilla: {vanilla_answer} (correct: {vanilla_correct}, score: {vanilla_best_score:.4f}, time: {vanilla_time:.1f}s)")
            
            budget_results['vanilla'] = {
                'answer': vanilla_answer,
                'correct': vanilla_correct,
                'best_score': vanilla_best_score,
                'avg_score': vanilla_avg_score,
                'time': vanilla_time,
                'num_responses': len(vanilla_result.responses)
            }
        except Exception as e:
            print(f"  Vanilla FAILED: {e}")
            budget_results['vanilla'] = {'error': str(e)}
        
        # Test Planning-Enhanced Best-of-N
        print(f"Running Planning-Enhanced Best-of-N (budget={budget})...")
        planning_start = time.time()
        
        try:
            planning_bon = PlanningBestOfN(orm=orm)
            planning_result = planning_bon.infer(lm, problem, budget, return_response_only=False)
            planning_time = time.time() - planning_start
            planning_answer = extract_boxed(planning_result.the_one)
            planning_correct = planning_answer == true_answer
            planning_best_score = max(planning_result.all_scores) if planning_result.all_scores else 0.0
            planning_avg_score = sum(planning_result.all_scores) / len(planning_result.all_scores) if planning_result.all_scores else 0.0
            
            print(f"  Planning: {planning_answer} (correct: {planning_correct}, score: {planning_best_score:.4f}, time: {planning_time:.1f}s)")
            print(f"  Approaches: {len(planning_result.approaches)} - {planning_result.approaches}")
            
            budget_results['planning'] = {
                'answer': planning_answer,
                'correct': planning_correct,
                'best_score': planning_best_score,
                'avg_score': planning_avg_score,
                'time': planning_time,
                'num_responses': len(planning_result.all_responses),
                'approaches': planning_result.approaches,
                'approach_budgets': planning_result.approach_budgets,
                'plan': planning_result.plan
            }
        except Exception as e:
            print(f"  Planning FAILED: {e}")
            budget_results['planning'] = {'error': str(e)}
        
        # Comparison
        if 'vanilla' in budget_results and 'planning' in budget_results and 'error' not in budget_results['vanilla'] and 'error' not in budget_results['planning']:
            vanilla_data = budget_results['vanilla']
            planning_data = budget_results['planning']
            
            print(f"  Comparison:")
            print(f"    Accuracy: Vanilla {vanilla_data['correct']} vs Planning {planning_data['correct']}")
            print(f"    Score Improvement: {planning_data['best_score'] - vanilla_data['best_score']:+.4f}")
            print(f"    Time Overhead: {planning_data['time'] - vanilla_data['time']:+.1f}s")
        
        results['results'][f'budget_{budget}'] = budget_results
    
    return results

def main():
    """Run the AIME2024 experiment."""
    
    print("AIME2024 Experiment: Planning-Enhanced vs Vanilla Best-of-N")
    print("="*60)
    
    # Load 5 AIME questions
    questions = load_aime_questions(num_questions=5, seed=42)
    
    # Initialize language model
    lm = OpenAICompatibleLanguageModel(
        endpoint="http://localhost:8100/v1",
        api_key="NO_API_KEY",
        model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    )
    
    # Initialize process reward model and convert to outcome reward model
    print("Loading reward model...")
    prm = LocalVllmProcessRewardModel(
        model_name="Qwen/Qwen2.5-Math-PRM-7B",
        device="cuda:1",  # Use GPU 1 for reward model
        aggregation_method="prod"
    )
    orm = ProcessToOutcomeRewardModel(prm)
    print("Reward model loaded successfully!")
    
    # Test budgets
    budgets = [4, 8, 16]
    
    # Run experiments
    all_results = []
    
    for i, question in enumerate(questions):
        print(f"\n{'#'*80}")
        print(f"QUESTION {i+1}/{len(questions)}")
        print(f"{'#'*80}")
        
        try:
            result = test_algorithms_on_question(lm, orm, question, budgets)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed on question {question['unique_id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = 'aime_experiment_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    summary = {}
    for budget in budgets:
        budget_key = f'budget_{budget}'
        summary[budget_key] = {
            'vanilla_correct': 0,
            'planning_correct': 0,
            'vanilla_total': 0,
            'planning_total': 0,
            'vanilla_avg_score': 0,
            'planning_avg_score': 0,
            'vanilla_avg_time': 0,
            'planning_avg_time': 0
        }
        
        vanilla_scores = []
        planning_scores = []
        vanilla_times = []
        planning_times = []
        
        for result in all_results:
            if budget_key in result['results']:
                budget_result = result['results'][budget_key]
                
                if 'vanilla' in budget_result and 'error' not in budget_result['vanilla']:
                    vanilla_data = budget_result['vanilla']
                    summary[budget_key]['vanilla_total'] += 1
                    if vanilla_data['correct']:
                        summary[budget_key]['vanilla_correct'] += 1
                    vanilla_scores.append(vanilla_data['best_score'])
                    vanilla_times.append(vanilla_data['time'])
                
                if 'planning' in budget_result and 'error' not in budget_result['planning']:
                    planning_data = budget_result['planning']
                    summary[budget_key]['planning_total'] += 1
                    if planning_data['correct']:
                        summary[budget_key]['planning_correct'] += 1
                    planning_scores.append(planning_data['best_score'])
                    planning_times.append(planning_data['time'])
        
        if vanilla_scores:
            summary[budget_key]['vanilla_avg_score'] = sum(vanilla_scores) / len(vanilla_scores)
        if planning_scores:
            summary[budget_key]['planning_avg_score'] = sum(planning_scores) / len(planning_scores)
        if vanilla_times:
            summary[budget_key]['vanilla_avg_time'] = sum(vanilla_times) / len(vanilla_times)
        if planning_times:
            summary[budget_key]['planning_avg_time'] = sum(planning_times) / len(planning_times)
    
    # Print summary
    for budget in budgets:
        budget_key = f'budget_{budget}'
        s = summary[budget_key]
        
        vanilla_acc = s['vanilla_correct'] / s['vanilla_total'] if s['vanilla_total'] > 0 else 0
        planning_acc = s['planning_correct'] / s['planning_total'] if s['planning_total'] > 0 else 0
        
        print(f"\nBUDGET {budget}:")
        print(f"  Vanilla Best-of-N:     {s['vanilla_correct']}/{s['vanilla_total']} correct ({vanilla_acc:.2%}), avg score: {s['vanilla_avg_score']:.4f}, avg time: {s['vanilla_avg_time']:.1f}s")
        print(f"  Planning Best-of-N:    {s['planning_correct']}/{s['planning_total']} correct ({planning_acc:.2%}), avg score: {s['planning_avg_score']:.4f}, avg time: {s['planning_avg_time']:.1f}s")
        
        if s['vanilla_total'] > 0 and s['planning_total'] > 0:
            acc_improvement = planning_acc - vanilla_acc
            score_improvement = s['planning_avg_score'] - s['vanilla_avg_score']
            time_overhead = s['planning_avg_time'] - s['vanilla_avg_time']
            print(f"  Improvement:           Accuracy: {acc_improvement:+.2%}, Score: {score_improvement:+.4f}, Time: {time_overhead:+.1f}s")
    
    print(f"\n✅ Experiment completed! Results saved to '{output_file}'")

if __name__ == "__main__":
    main()