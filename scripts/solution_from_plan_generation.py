#!/usr/bin/env python
"""
Solution Generation from Plans Script.

Takes a JSONL file with generated plans (from plan_generation.py) and generates
solutions for each plan at different BoN budget levels.

Usage:
    # OpenAI API
    uv run python scripts/solution_from_plan_generation.py \
        --plans-file results/aime2024/Qwen3-4B-Instruct-2507_rmQwen3-4B-Instruct-2507.jsonl \
        --output results/aime2024/solutions.jsonl

    # Local vLLM endpoint
    uv run python scripts/solution_from_plan_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --local --endpoint http://localhost:8000/v1 \
        --model qwen2-math-1.5b-instruct \
        --output results/aime2024/solutions.jsonl
"""

import argparse
import os
from pathlib import Path

import datasets
import math_verify
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.types import ChatMessage
from its_hub.utils import extract_content_from_lm_response

# System prompt that instructs model to use the plan
PLAN_BASED_SYSTEM_PROMPT = """You are given a mathematical problem and a plan for solving it.

Follow the plan step-by-step to solve the problem. Put your final answer within \\boxed{}.
"""


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 2 up to n."""
    budgets = []
    power = 1
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def extract_answer(response: str) -> str:
    """Extract answer from response (looks for content in \\boxed{})."""
    import re
    boxed_matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", response)
    return boxed_matches[-1] if boxed_matches else ""


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate solutions from plans and evaluate them"
    )

    # Input/output arguments
    parser.add_argument(
        "--plans-file",
        type=str,
        required=True,
        help="Path to JSONL file with generated plans",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file path for solutions"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for solution generation",
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
        "--temperature", type=float, default=0.0, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per generation"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests",
    )

    # Dataset arguments
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit number of problems to process",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Regenerate solutions even if they exist",
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

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Local mode: {args.local}")

    # Set up LM for solution generation (use async for batching)
    solution_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=PLAN_BASED_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        is_async=True,  # Enable async for batch generation
    )

    # Load plans dataset
    print(f"Loading plans from {args.plans_file}...")
    plans_df = pd.read_json(args.plans_file, orient="records", lines=True)

    if args.max_problems:
        plans_df = plans_df.head(args.max_problems)

    print(f"Loaded {len(plans_df)} problems with plans")

    # Determine budget levels from column names
    budget_cols = [col for col in plans_df.columns if col.startswith("bo") and col.endswith("_plan")]
    budgets = sorted([int(col.replace("bo", "").replace("_plan", "")) for col in budget_cols])
    print(f"Budget levels found: {budgets}")

    # Load existing output if it exists
    output_path = Path(args.output)
    if output_path.exists() and not args.force_run:
        print(f"Loading existing solutions from {output_path}...")
        existing_df = pd.read_json(output_path, orient="records", lines=True)
        # Merge with plans_df
        plans_df = plans_df.merge(
            existing_df[[col for col in existing_df.columns if col not in plans_df.columns] + ["problem_idx"]],
            on="problem_idx",
            how="left",
        )

    # Load AIME dataset to get ground truth answers
    print("Loading AIME dataset for ground truth answers...")
    aime_ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
    # Create mapping from problem to answer
    problem_to_answer = {row["Problem"]: str(row["Answer"]) for row in aime_ds}

    # Prepare all generation tasks
    generation_tasks = []
    task_metadata = []  # Store (idx, budget, ground_truth) for each task

    for idx, row in plans_df.iterrows():
        problem = row["problem"]
        ground_truth = problem_to_answer.get(problem, None)

        if ground_truth is None:
            print(f"\nWarning: No ground truth found for problem {row['problem_idx']}")
            continue

        for budget in budgets:
            plan_col = f"bo{budget}_plan"
            solution_col = f"bo{budget}_solution"

            # Skip if solution already exists and not forcing re-run
            if not args.force_run and solution_col in plans_df.columns and pd.notna(plans_df.at[idx, solution_col]):
                continue

            plan = row[plan_col]

            # Create prompt with plan
            prompt = f"Problem: {problem}\n\nPlan:\n{plan}\n\nNow solve the problem following the plan above:"

            # Add to batch
            messages = [ChatMessage(role="user", content=prompt)]
            generation_tasks.append(messages)
            task_metadata.append((idx, budget, ground_truth))

    # Generate all solutions in batch
    print(f"\nGenerating {len(generation_tasks)} solutions in batch...")
    if generation_tasks:
        responses = solution_lm.generate(generation_tasks)

        # Process all responses
        for (idx, budget, ground_truth), response in tqdm(
            zip(task_metadata, responses),
            total=len(task_metadata),
            desc="Processing solutions"
        ):
            solution_col = f"bo{budget}_solution"
            correct_col = f"bo{budget}_correct"

            try:
                solution = extract_content_from_lm_response(response)

                # Extract answer and evaluate
                predicted_answer = extract_answer(solution)
                is_correct = math_verify.verify(
                    math_verify.parse(ground_truth),
                    math_verify.parse(predicted_answer),
                )

                # Store results
                plans_df.at[idx, solution_col] = solution
                plans_df.at[idx, correct_col] = is_correct

            except Exception as e:
                print(f"\nError processing response for problem {idx} budget {budget}: {e}")
                plans_df.at[idx, solution_col] = None
                plans_df.at[idx, correct_col] = None

    # Save results
    print(f"\nSaving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plans_df.to_json(output_path, orient="records", lines=True)

    # Print accuracy summary
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    for budget in budgets:
        correct_col = f"bo{budget}_correct"
        if correct_col in plans_df.columns:
            accuracy = plans_df[correct_col].mean()
            n_correct = plans_df[correct_col].sum()
            n_total = plans_df[correct_col].notna().sum()
            print(f"Budget {budget:3d}: {accuracy:.4f} ({int(n_correct):2d}/{int(n_total):2d})")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
