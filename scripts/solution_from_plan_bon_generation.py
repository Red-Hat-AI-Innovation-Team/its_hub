#!/usr/bin/env python
"""
Solution Generation from Plans Script with Best-of-N Sampling.

Takes a JSONL file with generated plans (from plan_generation.py) and generates
solutions using Best-of-N sampling at different budget levels. The solution budget
can be controlled via a ratio parameter relative to the plan budget.

Usage:
    # OpenAI API (ratio 1.0: bo32 plan → 32 solutions)
    uv run python scripts/solution_from_plan_bon_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --output results/aime2024/solutions_bon.jsonl

    # Local vLLM endpoint with custom ratio (ratio 0.5: bo32 plan → 16 solutions)
    uv run python scripts/solution_from_plan_bon_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --local --endpoint http://localhost:8000/v1 \
        --model qwen2-math-1.5b-instruct \
        --solution-budget-ratio 0.5 \
        --output results/aime2024/solutions_bon.jsonl
"""

import argparse
import os
from pathlib import Path

import datasets
import math_verify
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

import litellm

litellm.drop_params = True


# System prompt that instructs model to use the plan
PLAN_BASED_SYSTEM_PROMPT = """You are given a mathematical problem and a plan for solving it.

Follow the plan step-by-step to solve the problem. Put your final answer within \\boxed{}."""


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
        description="Generate solutions from plans using Best-of-N and evaluate them"
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
        "--solution-budget-ratio",
        type=float,
        default=1.0,
        help="Ratio of solution budget to plan budget (e.g., 1.0 means bo32 plan → 32 solutions)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for solution generation"
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

    # Default judge model to same as generation model
    judge_model = args.judge_model or args.model

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Local mode: {args.local}")
    print(f"Solution budget ratio: {args.solution_budget_ratio}")

    # Set up LM for solution generation
    solution_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=PLAN_BASED_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )

    # Set up LLM judge as solution critic
    solution_critic = LLMJudgeRewardModel(
        model=judge_model,
        criterion="overall_quality",
        judge_type="pointwise",
        api_key=api_key,
        base_url=args.endpoint if args.local else None,
        temperature=0.0,
    )

    bon = BestOfN(orm=solution_critic)

    # Load plans dataset
    print(f"Loading plans from {args.plans_file}...")
    plans_df = pd.read_json(args.plans_file, orient="records", lines=True)

    if args.max_problems:
        plans_df = plans_df.head(args.max_problems)

    print(f"Loaded {len(plans_df)} problems with plans")

    # Determine budget levels from column names
    budget_cols = [col for col in plans_df.columns if col.startswith("bo") and col.endswith("_plan")]
    budgets = sorted([int(col.replace("bo", "").replace("_plan", "")) for col in budget_cols])
    print(f"Plan budget levels found: {budgets}")

    # Calculate solution budgets based on ratio (minimum 1)
    solution_budgets = {budget: max(1, int(budget * args.solution_budget_ratio)) for budget in budgets}
    print(f"Solution budgets (ratio={args.solution_budget_ratio}): {solution_budgets}")

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

    # Convert to list of dicts for easier manipulation
    results = plans_df.to_dict('records')

    # Process each problem
    for idx, row in enumerate(tqdm(results, desc="Processing problems")):
        problem = row["problem"]
        ground_truth = problem_to_answer.get(problem, None)

        if ground_truth is None:
            print(f"\nWarning: No ground truth found for problem {row['problem_idx']}")
            continue

        for plan_budget in budgets:
            plan_col = f"bo{plan_budget}_plan"
            solution_col = f"bo{plan_budget}_solution"
            correct_col = f"bo{plan_budget}_correct"
            score_col = f"bo{plan_budget}_score"

            # Skip if solution already exists and not forcing re-run
            if not args.force_run and solution_col in row and row[solution_col] is not None:
                continue

            plan = row[plan_col]
            n_solutions = solution_budgets[plan_budget]

            # Create prompt with plan
            prompt = f"Problem: {problem}\n\nPlan:\n{plan}\n\nNow solve the problem following the plan above:"

            try:
                # Generate N solutions using Best-of-N
                result = bon.infer(
                    solution_lm, prompt, budget=n_solutions, return_response_only=False
                )

                solutions = [extract_content_from_lm_response(r) for r in result.responses]
                scores = result.scores

                # Select best solution
                best_idx = scores.index(max(scores))
                best_solution = solutions[best_idx]
                best_score = scores[best_idx]

                # Extract answer and evaluate
                predicted_answer = extract_answer(best_solution)
                is_correct = math_verify.verify(
                    math_verify.parse(ground_truth),
                    math_verify.parse(predicted_answer),
                )

                # Store results in dict
                row[solution_col] = best_solution
                row[correct_col] = is_correct
                row[score_col] = best_score
                row[f"bo{plan_budget}_all_solutions"] = solutions
                row[f"bo{plan_budget}_all_scores"] = scores

            except Exception as e:
                print(f"\nError processing problem {idx} budget {plan_budget}: {e}")
                row[solution_col] = None
                row[correct_col] = None
                row[score_col] = None

    # Convert back to DataFrame
    plans_df = pd.DataFrame(results)

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
            sol_budget = solution_budgets[budget]
            print(f"Plan Budget {budget:3d} (→ {sol_budget:3d} solutions): {accuracy:.4f} ({int(n_correct):2d}/{int(n_total):2d})")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
