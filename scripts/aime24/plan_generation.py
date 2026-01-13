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
import random
from enum import Enum
from pathlib import Path


class BenchmarkDataset(Enum):
    MATH500 = "math500"
    AIME_2024 = "aime_2024"

import datasets
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

from reward_hub.llm_judge.prompts import Criterion, CriterionRegistry

import litellm

litellm.drop_params = True


# Plan generation prompt - no calculations, just strategy
# PLAN_GENERATION_SYSTEM_PROMPT = """You are a mathematical problem planning strategist.

# When given a problem, provide ONLY a high-level strategy or plan.
# Do NOT perform any calculations or provide the final answer.

# Your plan should include:
# 1. Key mathematical concepts/theorems to apply
# 2. The sequence of logical steps to take
# 3. What to solve for at each step

# Keep the plan concise. Focus on the "what" and "why", not the "how" of calculations."""

PLAN_GENERATION_SYSTEM_PROMPT = """You are a mathematical problem planning strategist who outputs recipe-style solution plans. 

You will be given a math contest problem.
Your task is to output ONLY a plan: a short numbered recipe describing how to solve the problem, without actually solving it.

Strict rules:
- Do NOT solve the problem.
- Do NOT compute any values.
- Do NOT include specific numbers except those already given in the problem.
- Do NOT give the final answer or intermediate results.

Formatting rules:
- Start with exactly:
  Plan:
- Include 5 to 10 numbered steps.
- Each step must:
  - Be a single sentence.
  - Start with a strong action verb (Model, Interpret, Count, Apply, Use, Subtract, Verify, etc.).
  - Mention the relevant mathematical toolkit (e.g., combinations, symmetry, graph interpretation, invariants, casework, algebraic manipulation).
- At least one step must be a sanity-check / pitfall step (e.g., overcounting, ordered vs unordered, extraneous roots, missing cases, domain issues).

Style guidelines:
- Write as if giving instructions to a solver.
- Prefer precise mathematical language over vague phrasing.
- Explicitly describe how the problem is modeled 
- Emphasize structure, not results.

Output format must be EXACTLY:

Plan:
1. ...
2. ...
3. ...
...
"""

PLAN_CRITIC_SYSTEM_PROMPT = """You are a strict mathematical problem planning critic who evaluates solution plans.

You will be given:
1. A math contest problem.
2. A plan that describes how to solve it (without solving it).

Your task is to rigorously judge the mathematical quality of the plan as a problem-solving strategy.

## Evaluation Criteria

1. Correctness of approach (0-3 points):
- 3: The method will definitively solve the problem with no mathematical errors. The approach is the standard or optimal method for this problem type.
- 2: The method is sound but has minor gaps, could fail in edge cases, or uses a suboptimal technique.
- 1: The approach has significant flaws, may not lead to a solution, or misidentifies the problem type.
- 0: The approach is fundamentally wrong, irrelevant, or would never work.

2. Completeness of reasoning (0-3 points):
- 3: Every critical step is present with clear logical flow. A skilled solver could execute this plan directly.
- 2: Most steps are present but one step is implicit, vague, or missing minor details.
- 1: Multiple steps are missing, the plan is too vague to execute, or key transitions are unexplained.
- 0: The plan is severely incomplete or just restates the problem.

3. Appropriateness of mathematical tools (0-2 points):
- 2: Tools are perfectly matched to the problem structure and represent the best approach.
- 1: Tools are reasonable but not optimal, slightly misapplied, or overly generic.
- 0: Tools are inappropriate, misidentified, or would not help solve the problem.

4. Risk awareness and error prevention (0-2 points):
- 2: Identifies the most likely pitfalls SPECIFIC to THIS problem (e.g., "check if k=0 case is included" not just "check edge cases").
- 1: Generic sanity checks that could apply to any problem (e.g., "verify no overcounting" without specifying what might be overcounted).
- 0: No meaningful error prevention, irrelevant checks, or checks that don't match the problem.

## Required Output Format:
{
  "reasoning": "<1-2 sentences: give a brief summary of the flaws in the plan>"
  "score": <integer from 0 to 10>,
}

Rules:
- You MUST identify at least one weakness unless the plan is truly flawless.
- A score of 9+ is exceptional and rare - the plan must be competition-ready with no improvements needed.
- Do not include any text outside the JSON.
"""

# Register the criterion
CriterionRegistry.register(Criterion(
    name="math_plan_quality",
    content=PLAN_CRITIC_SYSTEM_PROMPT,
    description="Evaluates mathematical problem-solving plans",
))


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 0 up to n."""
    budgets = []
    power = 0
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def weighted_best_of_n(
    responses: list[str], scores: list[float]
) -> tuple[str, dict[str, float]]:
    """
    Weighted Best-of-N: group identical responses, sum scores, pick best.

    Args:
        responses: List of response strings
        scores: List of scores corresponding to each response

    Returns:
        best_response: The response with highest aggregated score
        weighted_scores: Dict mapping each unique response to its summed score
    """
    assert len(responses) == len(scores)

    weighted: dict[str, float] = {}
    for r, s in zip(responses, scores):
        weighted[r] = weighted.get(r, 0.0) + s

    best_response = max(weighted.items(), key=lambda x: x[1])[0]
    return best_response, weighted


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
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name for plan generation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for plan evaluation (default: same as --model)",
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
        "--n-plans",
        type=int,
        default=64,
        help="Number of plans to generate per problem",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for plan generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max tokens per generation"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=32, help="Max concurrent API requests"
    )

    # Dataset arguments
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit number of problems to process",
    )
    parser.add_argument(
        "--output", type=str, default="results/plans.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer model name for token counting (default: same as --model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible budget sampling (default: 42)",
    )
    parser.add_argument(
        "--weighted-bon",
        action="store_true",
        help="Use weighted Best-of-N: aggregate scores for identical plans",
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
        criterion="math_plan_quality",
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
    print(f"Random seed: {args.seed}")

    # Set random seed for reproducible budget sampling
    random.seed(args.seed)

    results = []
    for idx, dataset_row in enumerate(tqdm(problem_dataset, desc="Problems")):
        problem = dataset_row["problem"]

        # Generate all N plans and score them in one call
        result = bon.infer(
            plan_lm, problem, budget=args.n_plans, return_response_only=False
        )

        plans = [extract_content_from_lm_response(r) for r in result.responses]
        scores = result.scores
        token_counts = [len(tokenizer.encode(plan)) for plan in plans]

        result_row = {
            "problem_idx": idx,
            "problem": problem,
            "plans": plans,
            "token_counts": token_counts,
            "scores": scores,
        }

        # Simulate each budget level by randomly sampling from all N plans
        for budget in budgets:
            # Randomly sample 'budget' indices from all available plans
            all_indices = list(range(len(plans)))
            sampled_indices = random.sample(all_indices, min(budget, len(plans)))

            # Get sampled plans and scores
            sampled_plans = [plans[i] for i in sampled_indices]
            sampled_scores = [scores[i] for i in sampled_indices]

            # Sum tokens for the sampled plans
            sampled_tokens = sum(token_counts[i] for i in sampled_indices)

            if args.weighted_bon:
                # Weighted BoN: aggregate scores for identical plans
                best_plan, weighted_dict = weighted_best_of_n(sampled_plans, sampled_scores)
                result_row[f"bo{budget}_plan"] = best_plan
                result_row[f"bo{budget}_plan_score"] = weighted_dict[best_plan]
                result_row[f"bo{budget}_unique_plans"] = len(weighted_dict)
            else:
                # Standard BoN: argmax
                best_sampled_idx = max(sampled_indices, key=lambda i: scores[i])
                result_row[f"bo{budget}_plan"] = plans[best_sampled_idx]
                result_row[f"bo{budget}_plan_score"] = scores[best_sampled_idx]

            result_row[f"bo{budget}_tokens"] = sampled_tokens
            result_row[f"bo{budget}_sampled_indices"] = sampled_indices

        results.append(result_row)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ds = datasets.Dataset.from_list(results)
    output_ds.to_json(output_path, orient="records", lines=True)

    # Print summary
    print("\n" + "=" * 60)
    print("PLAN GENERATION SUMMARY")
    print("=" * 60)
    for budget in budgets:
        tokens_col = f"bo{budget}_tokens"
        if results:
            token_values = [r[tokens_col] for r in results if tokens_col in r]
            if token_values:
                avg_tokens = sum(token_values) / len(token_values)
                print(f"Budget {budget:3d}: avg tokens: {avg_tokens:.0f}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
