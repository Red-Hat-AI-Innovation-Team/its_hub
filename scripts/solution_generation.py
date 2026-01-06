#!/usr/bin/env python
"""
Solution Generation Script for Direct Best-of-N Experiments.

Generates N solutions using BestOfN with LLM judge, then simulates all power-of-2
budget levels from the same generations. No planning phase - directly solves problems.

Usage:
    # OpenAI API
    uv run python scripts/solution_generation.py --n-solutions 64 --output results/solutions.jsonl

    # Local vLLM endpoint
    uv run python scripts/solution_generation.py --local --endpoint http://localhost:8000/v1 \
        --model qwen2-math-1.5b-instruct --n-solutions 64
"""

import argparse
import os
from enum import Enum
from pathlib import Path

import datasets
import math_verify
from dotenv import load_dotenv
from tqdm import tqdm

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

import litellm

litellm.drop_params = True


# Direct solution prompt - step-by-step reasoning with final answer
QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class BenchmarkDataset(Enum):
    """Supported benchmark datasets."""

    MATH500 = "math500"
    AIME_2024 = "aime-2024"


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 1 up to n."""
    budgets = []
    power = 0
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def load_benchmark_dataset(dataset: BenchmarkDataset):
    """Load and normalize a benchmark dataset."""
    if dataset == BenchmarkDataset.MATH500:
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset == BenchmarkDataset.AIME_2024:
        ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
        old_column_names = ds.column_names
        ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
        ds = ds.rename_column("id", "unique_id")
        ds = ds.cast_column("answer", datasets.Value("string"))
        ds = ds.remove_columns(old_column_names)
    # add unique_id if it doesn't exist
    if "unique_id" not in ds.column_names:
        ds = ds.map(lambda _, idx: {"unique_id": idx}, with_indices=True)
    return ds


def extract_answer(response: str) -> str:
    """Extract answer from response (looks for content in \\boxed{})."""
    import re
    boxed_matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", response)
    return boxed_matches[-1] if boxed_matches else ""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate solutions for math problems using Best-of-N"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for solution generation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for solution evaluation (default: same as --model)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://api.openai.com/v1",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local endpoint (sets api_key to NO_API_KEY)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from OPENAI_API_KEY env var)",
    )

    # Generation arguments
    parser.add_argument(
        "--n-solutions",
        type=int,
        default=64,
        help="Number of solutions to generate per problem",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for solution generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per generation"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=32, help="Max concurrent API requests"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime-2024",
        choices=["math500", "aime-2024"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit number of problems to process",
    )
    parser.add_argument(
        "--output", type=str, default="results/solutions.jsonl", help="Output file path"
    )

    args = parser.parse_args()

    load_dotenv()

    # Determine API key
    if args.local:
        api_key = "NO_API_KEY"
    elif args.api_key:
        api_key = args.api_key
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and not args.local:
        raise ValueError(
            "API key required. Set OPENAI_API_KEY or use --api-key or --local"
        )

    # Default judge model to same as generation model
    judge_model = args.judge_model or args.model

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Local mode: {args.local}")

    # Set up LM for solution generation
    solution_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=QWEN_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )

    # Set up LLM judge as solution critic (uses same model/endpoint as solution generation)
    solution_critic = LLMJudgeRewardModel(
        model=judge_model,
        criterion="overall_quality",
        judge_type="pointwise",
        api_key=api_key,
        base_url=args.endpoint if args.local else None,
        temperature=0.0,
    )

    bon = BestOfN(orm=solution_critic)

    # Load dataset
    dataset_enum = BenchmarkDataset.AIME_2024 if args.dataset == "aime-2024" else BenchmarkDataset.MATH500
    problem_dataset = load_benchmark_dataset(dataset_enum)
    if args.max_problems:
        problem_dataset = problem_dataset.select(range(args.max_problems))

    # Get ground truth answers
    if dataset_enum == BenchmarkDataset.AIME_2024:
        aime_ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
        problem_to_answer = {row["Problem"]: str(row["Answer"]) for row in aime_ds}
    else:
        problem_to_answer = {row["problem"]: row["answer"] for row in problem_dataset}

    budgets = get_power_of_2_budgets(args.n_solutions)
    print(f"Processing {len(problem_dataset)} problems, {args.n_solutions} solutions each")
    print(f"Budget levels: {budgets}")

    results = []
    for idx, dataset_row in enumerate(tqdm(problem_dataset, desc="Problems")):
        problem = dataset_row["problem"]
        ground_truth = problem_to_answer.get(problem, None)

        if ground_truth is None:
            print(f"\nWarning: No ground truth found for problem {idx}")
            continue

        # Generate all N solutions and score them in one call
        result = bon.infer(
            solution_lm, problem, budget=args.n_solutions, return_response_only=False
        )

        solutions = [extract_content_from_lm_response(r) for r in result.responses]
        scores = result.scores

        result_row = {
            "problem_idx": idx,
            "problem": problem,
            "ground_truth": ground_truth,
            "all_solutions": solutions,
            "all_scores": scores,
        }

        # Simulate each budget level from the same N solutions
        for budget in budgets:
            subset_scores = scores[:budget]
            best_idx = subset_scores.index(max(subset_scores))
            solution = solutions[best_idx]

            # Extract answer and evaluate
            predicted_answer = extract_answer(solution)
            try:
                is_correct = math_verify.verify(
                    math_verify.parse(ground_truth),
                    math_verify.parse(predicted_answer),
                )
            except Exception as e:
                print(f"\nError verifying answer for problem {idx} budget {budget}: {e}")
                is_correct = False

            result_row[f"bo{budget}_solution"] = solution
            result_row[f"bo{budget}_score"] = scores[best_idx]
            result_row[f"bo{budget}_predicted_answer"] = predicted_answer
            result_row[f"bo{budget}_correct"] = is_correct

        results.append(result_row)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ds = datasets.Dataset.from_list(results)
    output_ds.to_json(output_path, orient="records", lines=True)

    # Print accuracy summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for budget in budgets:
        correct_col = f"bo{budget}_correct"
        if results:
            correct_values = [r[correct_col] for r in results if correct_col in r]
            if correct_values:
                accuracy = sum(correct_values) / len(correct_values)
                n_correct = sum(correct_values)
                n_total = len(correct_values)
                print(f"Budget {budget:3d}: {accuracy:.4f} ({int(n_correct):2d}/{int(n_total):2d})")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
