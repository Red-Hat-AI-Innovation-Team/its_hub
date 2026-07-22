"""Sampling harness for tool-call voting checkpoint.

Wires up its_hub's OpenAICompatibleLanguageModel to pull N completions per
BFCL task from configured models, then compares:

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

sys.path.insert(0, str(Path(__file__).parent))

from roll_up_scorer import ScoredToolCall, score_tool_calls

from its_hub.api import ChatMessages, GenerationUsage
from its_hub.core.algorithms.self_consistency import SelfConsistency, SelfConsistencyResult
from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
from its_hub.core.orchestrator import LMOrchestrator

logger = logging.getLogger(__name__)

BFCL_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class ModelConfig:
    endpoint: str
    api_key: str
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

    Returns:
        (prompt_text, tools_list) ready for LM inference.
    """
    prompt = ""
    for msg in task.get("prompt", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            prompt = msg.get("content", "")
            break
    if not prompt:
        prompt = str(task.get("prompt", ""))

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


async def sample_and_score(
    lm: OpenAICompatibleLanguageModel,
    task: dict,
    budget: int,
    threshold: float = 0.75,
) -> TaskResult:
    """Sample N tool calls for a single task and score with both methods."""
    prompt, tools = build_tool_call_prompt(task)
    chat_messages = ChatMessages(prompt)
    task_id = task.get("id", "unknown")

    orchestrator = LMOrchestrator()
    usage = GenerationUsage()

    # Sample N completions
    responses = await orchestrator.agenerate(
        lm,
        chat_messages.to_batch(budget),
        tools=tools,
        tool_choice="auto",
        usage_accumulator=usage,
    )

    # Track cost
    cost = CostInfo(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.prompt_tokens + usage.completion_tokens,
        num_calls=budget,
    )

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
    sc = SelfConsistency(tool_vote="tool_hierarchical", orchestrator=orchestrator)
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
    bfcl_file: str = "BFCL_v3_simple.json",
    budget: int = 8,
    max_tasks: int | None = None,
    threshold: float = 0.75,
) -> list[TaskResult]:
    """Run the checkpoint experiment."""
    tasks = load_bfcl_tasks(bfcl_file)
    if max_tasks:
        tasks = tasks[:max_tasks]

    async with OpenAICompatibleLanguageModel(
        endpoint=model_config.endpoint,
        api_key=model_config.api_key,
        model_name=model_config.model_name,
        temperature=0.7,
    ) as lm:
        results = []
        for i, task in enumerate(tasks):
            logger.info("Task %d/%d: %s", i + 1, len(tasks), task.get("id", "?"))
            try:
                result = await sample_and_score(lm, task, budget, threshold)
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

    config = ModelConfig(
        endpoint=os.environ.get("ITS_ENDPOINT", "http://localhost:8100/v1"),
        api_key=os.environ.get("ITS_API_KEY", "NO_API_KEY"),
        model_name=os.environ.get("ITS_MODEL", ""),
    )

    if not config.model_name:
        print(
            "Set ITS_MODEL to the model name (e.g. 'meta-llama/Llama-3.1-8B-Instruct').\n"
            "Optionally set ITS_ENDPOINT (default: http://localhost:8100/v1) and ITS_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_checkpoint(
            config,
            bfcl_file=os.environ.get("BFCL_FILE", "BFCL_v3_simple.json"),
            budget=int(os.environ.get("BUDGET", "8")),
            max_tasks=int(os.environ.get("MAX_TASKS", "10")),
            threshold=float(os.environ.get("THRESHOLD", "0.75")),
        )
    )


if __name__ == "__main__":
    main()
