#!/usr/bin/env python
"""
Solution Generation from Plans Script.

Takes a JSONL file with generated plans (from plan_generation.py) and generates
solutions for each plan at different BoN budget levels. Supports both single
solution generation and Best-of-N sampling with LLM judge.

Usage:
    # Single solution per plan (no execution scaling)
    uv run python scripts/solution_from_plan_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --n-solutions 1 \
        --output results/aime2024/solutions.jsonl

    # Best-of-N solutions per plan (with execution scaling)
    uv run python scripts/solution_from_plan_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --n-solutions 8 \
        --local --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --output results/aime2024/solutions_bon.jsonl

    # Use ratio-based solution budget (e.g., 0.5 means bo32 plan → 16 solutions)
    uv run python scripts/solution_from_plan_generation.py \
        --plans-file results/aime2024/plans.jsonl \
        --solution-budget-ratio 0.5 \
        --output results/aime2024/solutions_hybrid.jsonl
"""

import argparse
import os
from pathlib import Path

import datasets
import math_verify
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.types import ChatMessage
from its_hub.utils import extract_content_from_lm_response

from reward_hub.llm_judge.prompts import Criterion, CriterionRegistry

import litellm

litellm.drop_params = True

PLAN_CONDITIONED_SOLUTION_CRITIC_SYSTEM_PROMPT = """You are a strict mathematical solution critic for plan-conditioned contest solutions.

You will be given:
1. A math contest problem.
2. A plan.
3. A candidate solution generated using that plan.

Your task is to rigorously judge the mathematical correctness of the candidate solution.
The plan is provided only to help you detect missing steps, incorrect pivots, or misuse of the intended method.
Presentation quality does NOT matter unless it prevents verifying correctness.

You MUST NOT provide a corrected solution.
You MUST NOT reveal the correct final answer unless it is already present in the candidate solution.
You MAY do lightweight checking (spot-check algebra, verify key claims), but do not do full-length solving from scratch.

## Evaluation Criteria (primarily correctness)

1. Correctness of result (0-4 points):
- 4: Final answer is correct and consistent with the problem.
- 3: Final answer is likely correct but not fully justified; minor uncertainty remains.
- 2: Final answer might be correct but there are serious errors/unsupported leaps.
- 1: Final answer is likely incorrect.
- 0: No valid final result or clearly wrong.

2. Validity of reasoning (0-4 points):
- 4: All key steps are logically valid; no mathematical errors.
- 3: Mostly valid with one minor gap/unproven claim that likely does not change the result.
- 2: Multiple gaps or at least one substantial error that could change the result.
- 1: Reasoning is largely flawed or relies on unjustified assumptions.
- 0: Reasoning is irrelevant or nonsensical.

3. Constraint/case handling (0-2 points):
- 2: Handles all necessary cases/constraints (domain, edge cases, extraneous roots, counting over/under, etc.).
- 1: Some constraints/cases not explicitly checked but plausibly covered.
- 0: Missing critical constraints/cases.

## Plan-conditioned check (use only as evidence, not as a separate score)
If the solution skips a key step promised by the plan or misapplies the plan’s method, treat that as evidence of weaker validity/completeness.

## Scoring Rules
- Total score is 0 to 10.
- You MUST identify at least one weakness unless the solution is truly flawless.
- A score of 9+ is exceptional and rare; it must be competition-ready with no meaningful improvements needed.

## Required Output Format (JSON only)
{
  "reasoning": "<1-2 sentences summarizing the biggest mathematical flaw or uncertainty in correctness>",
  "score": <integer from 0 to 10>
}

Do not include any text outside the JSON.

"""



# System prompt that instructs model to use the plan
PLAN_BASED_SYSTEM_PROMPT = """You are given a mathematical problem and a plan for solving it.

Follow the plan step-by-step to solve the problem. Put your final answer within \\boxed{}."""

# Register the criterion
CriterionRegistry.register(Criterion(
    name="plan_conditioned_solution_quality",
    content=PLAN_CONDITIONED_SOLUTION_CRITIC_SYSTEM_PROMPT,
    description="Evaluates mathematical solutions",
))


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
        default=None,
        help="Fixed number of solutions per plan (overrides --solution-budget-ratio)",
    )
    parser.add_argument(
        "--solution-budget-ratio",
        type=float,
        default=1.0,
        help="Ratio of solution budget to plan budget (e.g., 0.5 means bo32 plan → 16 solutions)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384, help="Max tokens per generation"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=64,
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
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer model name for token counting (default: same as --model)",
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

    # Load tokenizer for token counting
    tokenizer_model = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Tokenizer: {tokenizer_model}")
    print(f"Local mode: {args.local}")

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

    # Set up LLM judge for Best-of-N (only needed if n_solutions > 1)
    use_bon = args.n_solutions is None or args.n_solutions > 1
    if use_bon:
        solution_critic = LLMJudgeRewardModel(
            model=judge_model,
            criterion="plan_conditioned_solution_quality",
            judge_type="pointwise",
            api_key=api_key,
            base_url=args.endpoint if args.local else None,
            temperature=0.0,
            max_tokens=4096,
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

    # Calculate solution budgets
    if args.n_solutions is not None:
        # Fixed number of solutions for all budgets
        solution_budgets = {budget: args.n_solutions for budget in budgets}
        print(f"Fixed solution budget: {args.n_solutions} solutions per plan")
    else:
        # Ratio-based solution budgets
        solution_budgets = {budget: max(1, int(budget * args.solution_budget_ratio)) for budget in budgets}
        print(f"Solution budgets (ratio={args.solution_budget_ratio}): {solution_budgets}")

    # Load AIME dataset to get ground truth answers
    print("Loading AIME dataset for ground truth answers...")
    aime_ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
    problem_to_answer = {row["Problem"]: str(row["Answer"]) for row in aime_ds}

    # Convert to list of dicts for easier manipulation
    results = plans_df.to_dict('records')

    # Collect all generation tasks for batching
    generation_tasks = []  # List of (messages)
    task_metadata = []  # List of (row_idx, plan_budget, ground_truth, plan_tokens)

    print("Collecting generation tasks...")
    for idx, row in enumerate(results):
        problem = row["problem"]
        ground_truth = problem_to_answer.get(problem, None)

        if ground_truth is None:
            print(f"\nWarning: No ground truth found for problem {row['problem_idx']}")
            continue

        for plan_budget in budgets:
            plan_col = f"bo{plan_budget}_plan"
            solution_col = f"bo{plan_budget}_solution"

            # Skip if solution already exists and not forcing re-run
            if not args.force_run and solution_col in row and row[solution_col] is not None:
                continue

            plan = row[plan_col]
            n_solutions = solution_budgets[plan_budget]

            # Get plan tokens (cumulative tokens for first N plans)
            plan_tokens_col = f"bo{plan_budget}_tokens"
            plan_tokens = row.get(plan_tokens_col, 0) or 0

            # Create prompt with plan
            prompt = f"Problem: {problem}\n\nPlan:\n{plan}\n\nNow solve the problem following the plan above:"

            if n_solutions == 1:
                # Single solution - add to batch
                messages = [ChatMessage(role="user", content=prompt)]
                generation_tasks.append(messages)
                task_metadata.append((idx, plan_budget, ground_truth, plan_tokens, n_solutions, prompt))
            else:
                # Best-of-N - will be handled separately (can't batch easily)
                task_metadata.append((idx, plan_budget, ground_truth, plan_tokens, n_solutions, prompt))

    # Separate single-solution tasks from BoN tasks
    single_tasks = [(i, meta) for i, meta in enumerate(task_metadata) if meta[4] == 1]
    bon_tasks = [meta for meta in task_metadata if meta[4] > 1]

    # Batch generate single solutions
    if generation_tasks:
        print(f"\nGenerating {len(generation_tasks)} single solutions in batch...")
        responses = solution_lm.generate(generation_tasks)

        # Process single-solution responses
        for (task_idx, meta), response in tqdm(
            zip(single_tasks, responses),
            total=len(single_tasks),
            desc="Processing single solutions"
        ):
            idx, plan_budget, ground_truth, plan_tokens, n_solutions, prompt = meta
            row = results[idx]

            solution_col = f"bo{plan_budget}_solution"
            correct_col = f"bo{plan_budget}_correct"
            score_col = f"bo{plan_budget}_score"
            solution_tokens_col = f"bo{plan_budget}_solution_tokens"
            total_tokens_col = f"bo{plan_budget}_total_tokens"

            try:
                solution = extract_content_from_lm_response(response)
                solution_token_count = len(tokenizer.encode(solution))

                row[solution_col] = solution
                row[score_col] = None
                row[f"bo{plan_budget}_all_solutions"] = [solution]
                row[f"bo{plan_budget}_all_scores"] = []
                row[f"bo{plan_budget}_all_solution_tokens"] = [solution_token_count]
                total_solution_tokens = solution_token_count

                # Extract answer and evaluate
                predicted_answer = extract_answer(solution)
                is_correct = math_verify.verify(
                    math_verify.parse(ground_truth),
                    math_verify.parse(predicted_answer),
                )

                row[correct_col] = is_correct
                row[solution_tokens_col] = total_solution_tokens
                row[total_tokens_col] = plan_tokens + total_solution_tokens

            except Exception as e:
                print(f"\nError processing problem {idx} budget {plan_budget}: {e}")
                row[solution_col] = None
                row[correct_col] = None
                row[score_col] = None
                row[solution_tokens_col] = None
                row[total_tokens_col] = None

    # Process Best-of-N tasks (sequential, as each requires multiple generations + scoring)
    if bon_tasks:
        print(f"\nProcessing {len(bon_tasks)} Best-of-N tasks...")
        for meta in tqdm(bon_tasks, desc="Processing BoN solutions"):
            idx, plan_budget, ground_truth, plan_tokens, n_solutions, prompt = meta
            row = results[idx]

            solution_col = f"bo{plan_budget}_solution"
            correct_col = f"bo{plan_budget}_correct"
            score_col = f"bo{plan_budget}_score"
            solution_tokens_col = f"bo{plan_budget}_solution_tokens"
            total_tokens_col = f"bo{plan_budget}_total_tokens"

            try:
                result = bon.infer(
                    solution_lm, prompt, budget=n_solutions, return_response_only=False
                )

                solutions = [extract_content_from_lm_response(r) for r in result.responses]
                scores = result.scores
                solution_token_counts = [len(tokenizer.encode(s)) for s in solutions]

                # Select best solution
                best_idx = scores.index(max(scores))
                solution = solutions[best_idx]

                row[solution_col] = solution
                row[score_col] = scores[best_idx]
                row[f"bo{plan_budget}_all_solutions"] = solutions
                row[f"bo{plan_budget}_all_scores"] = scores
                row[f"bo{plan_budget}_all_solution_tokens"] = solution_token_counts
                total_solution_tokens = sum(solution_token_counts)

                # Extract answer and evaluate
                predicted_answer = extract_answer(solution)
                is_correct = math_verify.verify(
                    math_verify.parse(ground_truth),
                    math_verify.parse(predicted_answer),
                )

                row[correct_col] = is_correct
                row[solution_tokens_col] = total_solution_tokens
                row[total_tokens_col] = plan_tokens + total_solution_tokens

            except Exception as e:
                print(f"\nError processing problem {idx} budget {plan_budget}: {e}")
                row[solution_col] = None
                row[correct_col] = None
                row[score_col] = None
                row[solution_tokens_col] = None
                row[total_tokens_col] = None

    # Convert back to DataFrame
    plans_df = pd.DataFrame(results)

    # Save results
    output_path = Path(args.output)
    print(f"\nSaving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plans_df.to_json(output_path, orient="records", lines=True)

    # Print accuracy summary
    print("\n" + "="*70)
    print("ACCURACY SUMMARY")
    print("="*70)
    for budget in budgets:
        correct_col = f"bo{budget}_correct"
        total_tokens_col = f"bo{budget}_total_tokens"
        sol_budget = solution_budgets[budget]
        if correct_col in plans_df.columns:
            accuracy = plans_df[correct_col].mean()
            n_correct = plans_df[correct_col].sum()
            n_total = plans_df[correct_col].notna().sum()
            avg_tokens = plans_df[total_tokens_col].mean() if total_tokens_col in plans_df.columns else 0
            print(f"Plan bo{budget:2d} → {sol_budget:2d} sol: {accuracy:.4f} ({int(n_correct):2d}/{int(n_total):2d}) | avg total tokens: {avg_tokens:.0f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
