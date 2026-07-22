"""Sampling harness for tool-call voting checkpoint.

Uses litellm to call LLMs (including Vertex AI Claude) and compares:

  1. Field-aware scorer (roll_up_scorer.score_tool_calls)
  2. Naive exact-match baseline (SelfConsistency with tool_vote="tool_hierarchical")

Reuses the existing SelfConsistency implementation rather than rebuilding
the baseline from scratch.

Tracks token usage per task for cost-normalized comparison (voting at N=5
vs single-shot at matched token budget).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).parent))

from roll_up_scorer import ScoredToolCall, score_tool_calls

from its_hub.core.algorithms.self_consistency import SelfConsistency, SelfConsistencyResult

logger = logging.getLogger(__name__)

BFCL_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ModelConfig:
    model_name: str


@dataclass
class CostInfo:
    """Token usage for cost-normalized comparison."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    num_calls: int = 0


@dataclass
class TaskResult:
    task_id: str
    field_aware: ScoredToolCall | None
    baseline_selected: dict
    baseline_counts: dict
    num_samples: int
    cost: CostInfo = field(default_factory=CostInfo)


def load_bfcl_tasks(filename: str) -> list[dict]:
    """Load BFCL tasks from a local JSONL file."""
    filepath = BFCL_DATA_DIR / filename
    tasks = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def build_tool_call_prompt(task: dict) -> tuple[str, list[dict]]:
    """Extract the user prompt and tool definitions from a BFCL task.

    Handles both v3 format (task["prompt"]) and v4 format
    (task["question"] — a list of message-lists).

    Returns:
        (prompt_text, tools_list) ready for LM inference.
    """
    prompt = ""

    # v4 format: "question" is a list of turn-lists, each containing message dicts
    question = task.get("question") or task.get("prompt") or []
    if isinstance(question, list):
        for item in question:
            # v4: item is a list of message dicts (one turn)
            if isinstance(item, list):
                for msg in item:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break
            # v3: item is a message dict directly
            elif isinstance(item, dict) and item.get("role") == "user":
                prompt = item.get("content", "")
            if prompt:
                break

    if not prompt:
        prompt = str(question)

    functions = task.get("function", [])
    tools = []
    for func in functions:
        if isinstance(func, dict):
            tool_def = {
                "type": "function",
                "function": func,
            }
            tools.append(tool_def)

    return prompt, tools


async def _litellm_sample(
    model: str,
    messages: list[dict],
    tools: list[dict],
    n: int,
) -> tuple[list[dict], CostInfo]:
    """Sample N completions via litellm and return (responses, cost).

    Calls litellm.acompletion N times concurrently (litellm doesn't
    support n>1 for all providers).
    """
    async def _single_call() -> tuple[dict, int, int]:
        resp = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
        )
        choice = resp.choices[0].message  # type: ignore[union-attr]
        msg: dict = {"role": "assistant", "content": choice.content}
        if choice.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.tool_calls
            ]
        usage = resp.usage  # type: ignore[union-attr]
        return msg, usage.prompt_tokens, usage.completion_tokens

    tasks = [_single_call() for _ in range(n)]
    results = await asyncio.gather(*tasks)

    responses = [r[0] for r in results]
    cost = CostInfo(
        prompt_tokens=sum(r[1] for r in results),
        completion_tokens=sum(r[2] for r in results),
        total_tokens=sum(r[1] + r[2] for r in results),
        num_calls=n,
    )
    return responses, cost


async def sample_and_score(
    model: str,
    task: dict,
    budget: int,
    threshold: float = 0.75,
) -> TaskResult:
    """Sample N tool calls for a single task and score with both methods."""
    prompt, tools = build_tool_call_prompt(task)
    task_id = task.get("id", "unknown")

    messages = [{"role": "user", "content": prompt}]
    responses, cost = await _litellm_sample(model, messages, tools, budget)

    # Extract tool calls from responses
    raw_tool_calls = []
    for resp in responses:
        tcs = resp.get("tool_calls", [])
        if tcs:
            raw_tool_calls.append(tcs[0])

    # Method 1: Field-aware scorer
    if raw_tool_calls:
        field_aware_result = score_tool_calls(raw_tool_calls, threshold=threshold)
    else:
        field_aware_result = ScoredToolCall(
            tool_name="",
            tool_name_vote_count=0,
            tool_name_total=budget,
            tool_name_is_tie=True,
            merged_args={},
            field_votes=[],
            confidence="forced",
            selected_index=0,
            num_samples=budget,
        )

    # Method 2: Baseline — reuse SelfConsistency with tool_hierarchical
    sc = SelfConsistency(tool_vote="tool_hierarchical")
    baseline_result = sc._process_responses(responses, return_response_only=False)

    return TaskResult(
        task_id=task_id,
        field_aware=field_aware_result,
        baseline_selected=baseline_result.the_one if isinstance(baseline_result, SelfConsistencyResult) else baseline_result,
        baseline_counts=dict(baseline_result.response_counts) if isinstance(baseline_result, SelfConsistencyResult) else {},
        num_samples=budget,
        cost=cost,
    )


async def run_checkpoint(
    model_config: ModelConfig,
    bfcl_file: str = "BFCL_v4_simple_python.json",
    budget: int = 8,
    max_tasks: int | None = None,
    threshold: float = 0.75,
) -> list[TaskResult]:
    """Run the checkpoint experiment."""
    tasks = load_bfcl_tasks(bfcl_file)
    if max_tasks:
        tasks = tasks[:max_tasks]

    results = []
    for i, task in enumerate(tasks):
        logger.info("Task %d/%d: %s", i + 1, len(tasks), task.get("id", "?"))
        try:
            result = await sample_and_score(model_config.model_name, task, budget, threshold)
            results.append(result)
        except Exception:
            logger.exception("Failed on task %s", task.get("id", "?"))

    # Summary
    high_conf = sum(
        1 for r in results
        if r.field_aware is not None and r.field_aware.confidence == "high_confidence"
    )
    total_prompt = sum(r.cost.prompt_tokens for r in results)
    total_completion = sum(r.cost.completion_tokens for r in results)
    total_tokens = total_prompt + total_completion
    avg_tokens_per_task = total_tokens / len(results) if results else 0

    print(f"\n--- Checkpoint Results ({len(results)} tasks, budget={budget}) ---")
    print(f"  Field-aware high_confidence: {high_conf}/{len(results)}")
    print(f"  Field-aware forced:          {len(results) - high_conf}/{len(results)}")
    print(f"\n--- Cost ---")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  Prompt tokens:      {total_prompt:,}")
    print(f"  Completion tokens:  {total_completion:,}")
    print(f"  Avg tokens/task:    {avg_tokens_per_task:,.0f}")
    print(f"  Cost per N=1 equiv: {avg_tokens_per_task / budget:,.0f} tokens/task")

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = os.environ.get("FORECAST_MODEL", "vertex_ai/claude-sonnet-4@20250514")
    config = ModelConfig(model_name=model)

    print(f"Model: {config.model_name}")

    asyncio.run(
        run_checkpoint(
            config,
            bfcl_file=os.environ.get("BFCL_FILE", "BFCL_v4_simple_python.json"),
            budget=int(os.environ.get("BUDGET", "5")),
            max_tasks=int(os.environ.get("MAX_TASKS", "10")),
            threshold=float(os.environ.get("THRESHOLD", "0.75")),
        )
    )


if __name__ == "__main__":
    main()
