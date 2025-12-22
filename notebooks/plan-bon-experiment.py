# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: inference_time_scaling-dev
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.11
# ---

# %% [markdown]
# # Plan-then-Execute Experiment: Best-of-N for Plans
#
# This notebook tests the hypothesis that scaling plan generation (rather than full solution
# generation) is more compute-efficient.
#
# **Approach:**
# - Use existing BestOfN algorithm to generate and select plans
# - Use LLMJudgeRewardModel as a plan critic
# - Test with varying budgets: 2, 4, 8, 16, 32, 64
# - Evaluate on AIME 2024 problems

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

# %% [markdown]
# ## 1. Define Plan Generation System Prompt
#
# This prompt instructs the model to generate a high-level strategy WITHOUT
# performing calculations or giving the final answer.

# %%
PLAN_GENERATION_SYSTEM_PROMPT = """You are a mathematical problem-solving strategist.

When given a problem, provide ONLY a high-level solution strategy or plan.
Do NOT perform any calculations or provide the final answer.

Your plan should include:
1. Key mathematical concepts/theorems to apply
2. The sequence of logical steps to take
3. What to solve for at each step
4. Potential pitfalls to avoid

Keep the plan concise (3-7 steps). Focus on the "what" and "why", not the "how" of calculations."""

# %% [markdown]
# ## 2. Set up Language Model for Plan Generation

# %%
from its_hub.lms import OpenAICompatibleLanguageModel

plan_lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    system_prompt=PLAN_GENERATION_SYSTEM_PROMPT,
    is_async=True,
)

# %% [markdown]
# ## 3. Set up LLM Judge as Plan Critic
#
# We use LLMJudgeRewardModel with pointwise scoring to evaluate each plan independently.

# %%
from its_hub.integration.reward_hub import LLMJudgeRewardModel

plan_critic = LLMJudgeRewardModel(
    model="gpt-4o-mini",
    criterion="overall_quality",  # Can be replaced with custom plan criterion later
    judge_type="pointwise",  # Score each plan independently
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,  # Lower temperature for more consistent judging
)

# %% [markdown]
# ## 4. Set up Best-of-N Algorithm for Plans

# %%
from its_hub.algorithms import BestOfN

bon_plans = BestOfN(orm=plan_critic)

# %% [markdown]
# ## 5. Load AIME 2024 Problems

# %%
from datasets import load_dataset

# Load AIME validation dataset
aime_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")

# Show first few problems
print(f"Total problems: {len(aime_dataset)}")
print("\nFirst problem:")
print(aime_dataset[0]["problem"][:500] + "...")

# %% [markdown]
# ## 6. Single Problem Demo
#
# Let's first test with a single problem and budget to verify the setup works.

# %%
# Pick a sample problem
sample_problem = aime_dataset[0]["problem"]
print("Problem:")
print(sample_problem)
print("\n" + "="*80 + "\n")

# Generate plans with budget=4
budget = 4
result = bon_plans.infer(plan_lm, sample_problem, budget=budget, return_response_only=False)

print(f"Generated {len(result.responses)} plans")
print(f"Scores: {result.scores}")
print(f"Selected plan index: {result.selected_index}")
print("\n" + "="*80)
print("SELECTED PLAN:")
print("="*80)
from its_hub.utils import extract_content_from_lm_response
print(extract_content_from_lm_response(result.the_one))

# %% [markdown]
# ## 7. Show All Generated Plans
#
# Let's examine all the plans that were generated and their scores.

# %%
for i, (response, score) in enumerate(zip(result.responses, result.scores)):
    content = extract_content_from_lm_response(response)
    selected_marker = " [SELECTED]" if i == result.selected_index else ""
    print(f"\n{'='*80}")
    print(f"PLAN {i+1} (Score: {score:.3f}){selected_marker}")
    print("="*80)
    print(content)

# %% [markdown]
# ## 8. Experiment: Varying Budgets
#
# Test how plan quality scales with budget.

# %%
budgets = [2, 4, 8, 16, 32, 64]

# Pick a few problems for the experiment
num_problems = 3  # Adjust as needed
problems = [aime_dataset[i]["problem"] for i in range(num_problems)]

results_by_budget = {}

for budget in budgets:
    print(f"\n{'='*80}")
    print(f"Testing budget={budget}")
    print("="*80)

    budget_results = []
    for i, problem in enumerate(problems):
        print(f"  Problem {i+1}/{num_problems}...", end=" ")
        result = bon_plans.infer(plan_lm, problem, budget=budget, return_response_only=False)
        budget_results.append({
            "problem_idx": i,
            "scores": result.scores,
            "max_score": max(result.scores),
            "selected_score": result.scores[result.selected_index],
            "num_unique_scores": len(set(result.scores)),
        })
        print(f"max_score={max(result.scores):.3f}, selected_score={result.scores[result.selected_index]:.3f}")

    results_by_budget[budget] = budget_results

# %% [markdown]
# ## 9. Analyze Results
#
# Summarize how the best plan quality changes with budget.

# %%
import numpy as np

print("\nSummary: Average max score by budget")
print("="*40)
for budget in budgets:
    max_scores = [r["max_score"] for r in results_by_budget[budget]]
    avg_max = np.mean(max_scores)
    std_max = np.std(max_scores)
    print(f"Budget {budget:3d}: avg_max_score = {avg_max:.3f} (+/- {std_max:.3f})")

# %% [markdown]
# ## 10. Next Steps
#
# After validating plan generation and selection:
# 1. Execute the selected plan to produce a final answer
# 2. Compare accuracy with baseline (direct solution generation)
# 3. Measure token usage to verify compute efficiency
