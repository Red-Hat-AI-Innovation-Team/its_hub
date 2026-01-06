#!/usr/bin/env python
"""
Pass@k Calculation Script for Solution Evaluation.

Calculates pass@k metrics by randomly sampling k solutions per problem
and checking if any solution is correct. Supports power-of-2 budgets from 1 to 64.

Usage:
    uv run python scripts/calculate_pass_at_k.py \
        --input results/aime2024/solutions/Qwen3-4B_no_planning.jsonl \
        --solution-column all_solutions
"""

import argparse
import random
import re
from pathlib import Path

import datasets
import math_verify
from tqdm import tqdm


def extract_answer(response: str) -> str:
    """Extract answer from response (looks for content in \\boxed{})."""
    boxed_matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", response)
    return boxed_matches[-1] if boxed_matches else ""


def evaluate_solution(solution: str, ground_truth: str) -> bool:
    """Evaluate if a solution is correct by comparing extracted answer to ground truth."""
    predicted_answer = extract_answer(solution)
    try:
        is_correct = math_verify.verify(
            math_verify.parse(ground_truth),
            math_verify.parse(predicted_answer),
        )
        return is_correct
    except Exception:
        return False


def calculate_pass_at_k(problems: list[dict], k: int, solution_column: str) -> float:
    """
    Calculate pass@k metric.

    For each problem, sample k solutions and check if any is correct.
    Returns the fraction of problems where at least one solution was correct.
    """
    passed_count = 0
    total_count = 0

    for problem in problems:
        solutions = problem.get(solution_column, [])
        ground_truth = problem.get("ground_truth")

        if not solutions or ground_truth is None:
            continue

        # Sample k solutions (with replacement if k > len(solutions))
        if len(solutions) >= k:
            sampled_solutions = random.sample(solutions, k)
        else:
            # If we don't have k solutions, sample with replacement
            sampled_solutions = random.choices(solutions, k=k)

        # Check if any solution is correct
        any_correct = any(
            evaluate_solution(sol, ground_truth)
            for sol in sampled_solutions
        )

        if any_correct:
            passed_count += 1
        total_count += 1

    return passed_count / total_count if total_count > 0 else 0.0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate pass@k metrics for solution evaluation"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL file with solutions",
    )
    parser.add_argument(
        "--solution-column",
        type=str,
        default="all_solutions",
        help="Column name containing list of solutions (default: all_solutions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Maximum power of 2 for k (0 to max-k, default: 6 gives k up to 64)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading dataset from {input_path}...")
    dataset = datasets.load_dataset("json", data_files=str(input_path))["train"]

    print(f"Loaded {len(dataset)} problems")
    print(f"Solution column: {args.solution_column}")
    print(f"Random seed: {args.seed}")
    print()

    # Calculate pass@k for each power of 2
    budgets = [2**i for i in range(args.max_k + 1)]
    results = []

    print("Calculating pass@k metrics...")
    for k in tqdm(budgets, desc="Budget levels"):
        pass_at_k = calculate_pass_at_k(dataset, k, args.solution_column)
        results.append((k, pass_at_k))

    # Print results
    print("\n" + "="*60)
    print("PASS@K RESULTS")
    print("="*60)
    print(f"{'k':<10} {'pass@k':<15} {'percentage':<15}")
    print("-"*60)

    for k, pass_rate in results:
        percentage = pass_rate * 100
        print(f"{k:<10} {pass_rate:<15.4f} {percentage:<15.2f}%")

    print("="*60)


if __name__ == "__main__":
    main()
