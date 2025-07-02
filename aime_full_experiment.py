#!/usr/bin/env python3
"""
AIME2024 Full Dataset Experiment: Planning-Enhanced Best-of-N vs Vanilla Best-of-N
Run comparison on all 30 AIME questions with proper reward model evaluation.
"""

import sys
import time
import json
import random
from typing import Dict, Any, List
import datasets
import traceback
from datetime import datetime
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

def load_full_aime_dataset() -> List[Dict[str, Any]]:
    """Load all AIME2024 questions."""
    print("Loading full AIME2024 dataset...")
    
    # Load AIME2024 dataset
    ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
    
    # Process column names (following benchmark.py pattern)
    old_column_names = ds.column_names
    ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
    ds = ds.rename_column('id', 'unique_id')
    ds = ds.cast_column('answer', datasets.Value('string'))
    ds = ds.remove_columns(old_column_names)
    
    # Convert to list
    all_questions = list(ds)
    
    print(f"Loaded {len(all_questions)} AIME2024 questions")
    for i, q in enumerate(all_questions[:5]):  # Show first 5
        print(f"  {i+1}. ID: {q['unique_id']} - {q['problem'][:80]}...")
    if len(all_questions) > 5:
        print(f"  ... and {len(all_questions) - 5} more questions")
    
    return all_questions

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

def save_progress(results: List[Dict], filename: str):
    """Save intermediate results to avoid data loss."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Progress saved to {filename}")

def test_algorithms_on_question(
    lm: OpenAICompatibleLanguageModel,
    orm: ProcessToOutcomeRewardModel,
    question: Dict[str, Any],
    budgets: List[int],
    question_idx: int,
    total_questions: int
) -> Dict[str, Any]:
    """Test both algorithms on a single AIME question."""
    
    problem = question['problem']
    true_answer = question['answer']
    question_id = question['unique_id']
    
    print(f"\n{'='*80}")
    print(f"Question {question_idx}/{total_questions}: {question_id}")
    print(f"Problem: {problem[:100]}..." if len(problem) > 100 else f"Problem: {problem}")
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
            traceback.print_exc()
        
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
            print(f"  Approaches: {len(planning_result.approaches)}")
            
            budget_results['planning'] = {
                'answer': planning_answer,
                'correct': planning_correct,
                'best_score': planning_best_score,
                'avg_score': planning_avg_score,
                'time': planning_time,
                'num_responses': len(planning_result.all_responses),
                'approaches': planning_result.approaches,
                'approach_budgets': planning_result.approach_budgets,
                'plan': planning_result.plan[:500] + "..." if len(planning_result.plan) > 500 else planning_result.plan  # Truncate long plans
            }
        except Exception as e:
            print(f"  Planning FAILED: {e}")
            budget_results['planning'] = {'error': str(e)}
            traceback.print_exc()
        
        # Comparison
        if ('vanilla' in budget_results and 'planning' in budget_results and 
            'error' not in budget_results['vanilla'] and 'error' not in budget_results['planning']):
            vanilla_data = budget_results['vanilla']
            planning_data = budget_results['planning']
            
            print(f"  Comparison:")
            print(f"    Accuracy: Vanilla {vanilla_data['correct']} vs Planning {planning_data['correct']}")
            print(f"    Score Improvement: {planning_data['best_score'] - vanilla_data['best_score']:+.4f}")
            print(f"    Time Overhead: {planning_data['time'] - vanilla_data['time']:+.1f}s")
        
        results['results'][f'budget_{budget}'] = budget_results
    
    return results

def calculate_summary_stats(all_results: List[Dict], budgets: List[int]) -> Dict:
    """Calculate summary statistics across all questions."""
    summary = {}
    
    for budget in budgets:
        budget_key = f'budget_{budget}'
        summary[budget_key] = {
            'vanilla_correct': 0,
            'planning_correct': 0,
            'vanilla_total': 0,
            'planning_total': 0,
            'vanilla_errors': 0,
            'planning_errors': 0,
            'vanilla_scores': [],
            'planning_scores': [],
            'vanilla_times': [],
            'planning_times': [],
            'score_improvements': [],
            'time_overheads': []
        }
        
        for result in all_results:
            if budget_key in result['results']:
                budget_result = result['results'][budget_key]
                
                # Vanilla stats
                if 'vanilla' in budget_result:
                    if 'error' in budget_result['vanilla']:
                        summary[budget_key]['vanilla_errors'] += 1
                    else:
                        vanilla_data = budget_result['vanilla']
                        summary[budget_key]['vanilla_total'] += 1
                        if vanilla_data['correct']:
                            summary[budget_key]['vanilla_correct'] += 1
                        summary[budget_key]['vanilla_scores'].append(vanilla_data['best_score'])
                        summary[budget_key]['vanilla_times'].append(vanilla_data['time'])
                
                # Planning stats
                if 'planning' in budget_result:
                    if 'error' in budget_result['planning']:
                        summary[budget_key]['planning_errors'] += 1
                    else:
                        planning_data = budget_result['planning']
                        summary[budget_key]['planning_total'] += 1
                        if planning_data['correct']:
                            summary[budget_key]['planning_correct'] += 1
                        summary[budget_key]['planning_scores'].append(planning_data['best_score'])
                        summary[budget_key]['planning_times'].append(planning_data['time'])
                        
                        # Comparisons (only if both succeeded)
                        if ('vanilla' in budget_result and 'error' not in budget_result['vanilla'] 
                            and 'error' not in budget_result['planning']):
                            vanilla_data = budget_result['vanilla']
                            score_improvement = planning_data['best_score'] - vanilla_data['best_score']
                            time_overhead = planning_data['time'] - vanilla_data['time']
                            summary[budget_key]['score_improvements'].append(score_improvement)
                            summary[budget_key]['time_overheads'].append(time_overhead)
    
    return summary

def print_summary_stats(summary: Dict, budgets: List[int], total_questions: int):
    """Print comprehensive summary statistics."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"Total Questions: {total_questions}")
    print(f"{'='*80}")
    
    for budget in budgets:
        budget_key = f'budget_{budget}'
        s = summary[budget_key]
        
        print(f"\n📊 BUDGET {budget}:")
        
        # Accuracy stats
        vanilla_acc = s['vanilla_correct'] / s['vanilla_total'] if s['vanilla_total'] > 0 else 0
        planning_acc = s['planning_correct'] / s['planning_total'] if s['planning_total'] > 0 else 0
        acc_improvement = planning_acc - vanilla_acc
        
        print(f"  🎯 Accuracy:")
        print(f"    Vanilla:    {s['vanilla_correct']}/{s['vanilla_total']} = {vanilla_acc:.1%} (errors: {s['vanilla_errors']})")
        print(f"    Planning:   {s['planning_correct']}/{s['planning_total']} = {planning_acc:.1%} (errors: {s['planning_errors']})")
        print(f"    Improvement: {acc_improvement:+.1%}")
        
        # Score stats
        if s['vanilla_scores'] and s['planning_scores']:
            vanilla_avg_score = sum(s['vanilla_scores']) / len(s['vanilla_scores'])
            planning_avg_score = sum(s['planning_scores']) / len(s['planning_scores'])
            print(f"  📈 Average Scores:")
            print(f"    Vanilla:    {vanilla_avg_score:.4f}")
            print(f"    Planning:   {planning_avg_score:.4f}")
            print(f"    Improvement: {planning_avg_score - vanilla_avg_score:+.4f}")
        
        # Time stats
        if s['vanilla_times'] and s['planning_times']:
            vanilla_avg_time = sum(s['vanilla_times']) / len(s['vanilla_times'])
            planning_avg_time = sum(s['planning_times']) / len(s['planning_times'])
            time_overhead_pct = ((planning_avg_time / vanilla_avg_time) - 1) * 100
            print(f"  ⏱️  Average Times:")
            print(f"    Vanilla:    {vanilla_avg_time:.1f}s")
            print(f"    Planning:   {planning_avg_time:.1f}s")
            print(f"    Overhead:   {time_overhead_pct:+.1f}%")
        
        # Direct comparison stats
        if s['score_improvements']:
            avg_score_improvement = sum(s['score_improvements']) / len(s['score_improvements'])
            score_improvements_positive = sum(1 for x in s['score_improvements'] if x > 0)
            print(f"  🔄 Direct Comparisons ({len(s['score_improvements'])} pairs):")
            print(f"    Avg Score Improvement: {avg_score_improvement:+.4f}")
            print(f"    Planning Better Scores: {score_improvements_positive}/{len(s['score_improvements'])}")
        
        if s['time_overheads']:
            avg_time_overhead = sum(s['time_overheads']) / len(s['time_overheads'])
            print(f"    Avg Time Overhead: {avg_time_overhead:+.1f}s")

def main():
    """Run the full AIME2024 experiment."""
    
    print("🚀 AIME2024 Full Dataset Experiment: Planning-Enhanced vs Vanilla Best-of-N")
    print("="*80)
    start_time = datetime.now()
    print(f"Start time: {start_time}")
    
    # Load all AIME questions
    questions = load_full_aime_dataset()
    total_questions = len(questions)
    
    # Initialize language model
    print("\nInitializing language model...")
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
    print("Models loaded successfully!")
    
    # Test budgets
    budgets = [4, 8, 16]
    
    # Run experiments
    all_results = []
    output_file = f'aime_full_experiment_results_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    
    try:
        for i, question in enumerate(questions, 1):
            print(f"\n{'#'*80}")
            print(f"QUESTION {i}/{total_questions} - Progress: {i/total_questions:.1%}")
            print(f"{'#'*80}")
            
            try:
                result = test_algorithms_on_question(lm, orm, question, budgets, i, total_questions)
                all_results.append(result)
                
                # Save progress every 5 questions
                if i % 5 == 0 or i == total_questions:
                    save_progress(all_results, output_file)
                    
            except Exception as e:
                print(f"❌ Failed on question {question['unique_id']}: {e}")
                traceback.print_exc()
                # Continue with next question
                continue
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user at question {len(all_results)}/{total_questions}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        traceback.print_exc()
    finally:
        # Always save final results
        save_progress(all_results, output_file)
    
    # Calculate and display summary statistics
    if all_results:
        summary = calculate_summary_stats(all_results, budgets)
        print_summary_stats(summary, budgets, len(all_results))
        
        # Save summary
        summary_file = f'aime_full_summary_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n📊 Summary statistics saved to '{summary_file}'")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n⏱️  Experiment completed!")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Duration: {duration}")
    print(f"Questions processed: {len(all_results)}/{total_questions}")
    print(f"Results saved to: '{output_file}'")

if __name__ == "__main__":
    main()