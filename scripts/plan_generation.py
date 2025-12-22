#!/usr/bin/env python
"""
Plan Generation Script for Plan-then-Execute Experiments.

Generates N plans using BestOfN, then simulates all power-of-2 budget levels
from the same generations.

Usage:
    # OpenAI API
    uv run python scripts/plan_generation.py --n-plans 64 --output results/plans.jsonl

    # Local vLLM endpoint
    uv run python scripts/plan_generation.py --local --endpoint http://localhost:8000/v1 \
        --model qwen2-math-1.5b-instruct --n-plans 64
"""

import argparse
import os
from enum import Enum
from pathlib import Path

import datasets
from dotenv import load_dotenv
from tqdm import tqdm

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

# Plan generation prompt - no calculations, just strategy
PLAN_GENERATION_SYSTEM_PROMPT = """You are a mathematical problem planning strategist.

When given a problem, provide ONLY a high-level strategy or plan.
Do NOT perform any calculations or provide the final answer.

Your plan should include:
1. Key mathematical concepts/theorems to apply
2. The sequence of logical steps to take
3. What to solve for at each step

Keep the plan concise. Focus on the "what" and "why", not the "how" of calculations."""


class BenchmarkDataset(Enum):
    """Supported benchmark datasets."""

    MATH500 = "math500"
    AIME_2024 = "aime-2024"


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 2 up to n."""
    budgets = []
    power = 1
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate plans for math problems"
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name for plan generation")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name for plan evaluation (default: same as --model)")
    parser.add_argument("--endpoint", type=str, default="https://api.openai.com/v1",
                        help="API endpoint URL")
    parser.add_argument("--local", action="store_true",
                        help="Use local endpoint (sets api_key to NO_API_KEY)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (default: from OPENAI_API_KEY env var)")

    # Generation arguments
    parser.add_argument("--n-plans", type=int, default=64,
                        help="Number of plans to generate per problem")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for plan generation")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per generation")
    parser.add_argument("--max-concurrency", type=int, default=32,
                        help="Max concurrent API requests")

    # Dataset arguments
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems to process")
    parser.add_argument("--output", type=str, default="results/plans.jsonl",
                        help="Output file path")

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
        raise ValueError("API key required. Set OPENAI_API_KEY or use --api-key or --local")

    # Default judge model to same as generation model
    judge_model = args.judge_model or args.model

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Local mode: {args.local}")

    # Set up LM for plan generation
    plan_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=PLAN_GENERATION_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )

    # Set up LLM judge as plan critic (uses same model/endpoint as plan generation)
    plan_critic = LLMJudgeRewardModel(
        model=judge_model,
        criterion="overall_quality",
        judge_type="pointwise",
        api_key=api_key,
        base_url=args.endpoint if args.local else None,
        temperature=0.0,
    )

    bon = BestOfN(orm=plan_critic)

    # Load dataset
    problem_dataset = load_benchmark_dataset(BenchmarkDataset.AIME_2024)
    if args.max_problems:
        problem_dataset = problem_dataset.select(range(args.max_problems))

    budgets = get_power_of_2_budgets(args.n_plans)
    print(f"Processing {len(problem_dataset)} problems, {args.n_plans} plans each")
    print(f"Budget levels: {budgets}")

    results = []
    for idx, dataset_row in enumerate(tqdm(problem_dataset, desc="Problems")):
        problem = dataset_row["problem"]

        # Generate all N plans and score them in one call
        result = bon.infer(
            plan_lm, problem, budget=args.n_plans, return_response_only=False
        )

        plans = [extract_content_from_lm_response(r) for r in result.responses]
        scores = result.scores

        result_row = {
            "problem_idx": idx,
            "problem": problem,
            "all_plans": plans,
            "all_scores": scores,
        }

        # Simulate each budget level from the same N plans
        for budget in budgets:
            subset_scores = scores[:budget]
            best_idx = subset_scores.index(max(subset_scores))
            result_row[f"bo{budget}_plan"] = plans[best_idx]
            result_row[f"bo{budget}_plan_score"] = scores[best_idx]

        results.append(result_row)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ds = datasets.Dataset.from_list(results)
    output_ds.to_json(output_path, orient="records", lines=True)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
