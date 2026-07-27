"""LLMJudge Best-of-N harness for tool-call voting research.

Generates N=5 tool call candidates per BFCL task, scores each with an
LLM judge, and selects the highest-scored candidate (Best-of-N). Compares
against single-shot, majority-voted, and oracle baselines.

Environment variables:
    FORECAST_MODEL  — litellm model string (default: gpt-4o-mini)
    TEMPERATURE     — sampling temperature (default: 1.0)
    MAX_TASKS       — number of tasks to evaluate (default: 20)
    MAX_CONCURRENT  — concurrency limit (default: 2)
    N_SAMPLES       — number of candidates per task (default: 5)
    BFCL_FILE       — BFCL data file (default: BFCL_v4_multiple.json)
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

from analyze_results import check_tool_call_against_gt, load_ground_truth
from roll_up_scorer import score_tool_calls
from sampling_harness import (
    CostInfo,
    build_tool_call_prompt,
    load_bfcl_tasks,
    _sanitize_function_schema,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating a function call made in response to a user query. "
    "Score how well the function call addresses the request: correct function "
    "selected? correct parameters? appropriate values? Score 0-10 where 10 is "
    "perfect. Return JSON: {\"score\": <number>, \"reasoning\": \"<brief explanation>\"}"
)

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


@dataclass
class JudgeScore:
    candidate_index: int
    score: float
    reasoning: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class BoNTaskResult:
    task_id: str
    n_samples: int
    candidates: list[dict]
    judge_scores: list[JudgeScore]
    generation_cost: CostInfo
    judge_cost: CostInfo
    single_shot_correct: bool
    voted_correct: bool
    bon_correct: bool
    oracle_correct: bool


async def _litellm_sample(
    model: str,
    messages: list[dict],
    tools: list[dict],
    n: int,
) -> tuple[list[dict], CostInfo]:
    """Sample N completions via litellm with rate limiting and retries."""
    sem = _get_semaphore()

    async def _single_call() -> tuple[dict, int, int]:
        max_retries = 5
        for attempt in range(max_retries):
            async with sem:
                try:
                    resp = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=float(os.environ.get("TEMPERATURE", "1.0")),
                    )
                    choice = resp.choices[0].message
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
                    usage = resp.usage
                    return msg, usage.prompt_tokens, usage.completion_tokens
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = 2**attempt * 5
                        logger.warning("Rate limited, retrying in %ds...", wait)
                        await asyncio.sleep(wait)
                    else:
                        raise
        raise RuntimeError(f"Failed after {max_retries} retries")

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


def _format_tool_call_for_judge(tool_call: dict) -> str:
    """Format a tool call dict as readable text for the judge."""
    func = tool_call.get("function", {})
    name = func.get("name", "unknown")
    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
            args_str = json.dumps(args, indent=2)
        except (json.JSONDecodeError, TypeError):
            args_str = args_raw
    else:
        args_str = json.dumps(args_raw, indent=2)
    return f"{name}({args_str})"


async def _judge_candidate(
    model: str,
    user_prompt: str,
    tool_call: dict,
    candidate_index: int,
    tools: list[dict],
) -> JudgeScore:
    """Score a single candidate tool call with the LLM judge."""
    sem = _get_semaphore()

    tool_call_text = _format_tool_call_for_judge(tool_call)

    available_functions = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            available_functions.append(fn.get("name", "?"))
    functions_str = ", ".join(available_functions)

    judge_user_msg = (
        f"User query: {user_prompt}\n\n"
        f"Available functions: {functions_str}\n\n"
        f"Function call made:\n{tool_call_text}\n\n"
        f"Score this function call 0-10."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_msg},
    ]

    max_retries = 5
    for attempt in range(max_retries):
        async with sem:
            try:
                resp = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                usage = resp.usage

                score = 5.0
                reasoning = ""
                try:
                    parsed = json.loads(content)
                    score = float(parsed.get("score", 5.0))
                    reasoning = parsed.get("reasoning", "")
                except (json.JSONDecodeError, TypeError, ValueError):
                    import re
                    match = re.search(r'"score"\s*:\s*([\d.]+)', content)
                    if match:
                        score = float(match.group(1))

                return JudgeScore(
                    candidate_index=candidate_index,
                    score=score,
                    reasoning=reasoning,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 2**attempt * 5
                    logger.warning("Judge rate limited, retrying in %ds...", wait)
                    await asyncio.sleep(wait)
                else:
                    raise

    return JudgeScore(
        candidate_index=candidate_index,
        score=5.0,
        reasoning="Failed after retries",
    )


def _parse_tool_call_args(tool_call: dict) -> tuple[str, dict]:
    """Extract (name, args_dict) from a tool call dict."""
    func = tool_call.get("function", {})
    name = func.get("name", "")
    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = args_raw if isinstance(args_raw, dict) else {}
    return name, args


async def process_task(
    model: str,
    task: dict,
    ground_truth: list[dict],
    n_samples: int,
) -> BoNTaskResult:
    """Generate candidates, judge them, and score against ground truth."""
    task_id = task.get("id", "unknown")
    prompt, tools = build_tool_call_prompt(task)

    messages = [{"role": "user", "content": prompt}]
    responses, gen_cost = await _litellm_sample(model, messages, tools, n_samples)

    # Extract tool calls from responses
    candidates: list[dict] = []
    for resp in responses:
        tcs = resp.get("tool_calls", [])
        if tcs:
            candidates.append(tcs[0])

    if not candidates:
        return BoNTaskResult(
            task_id=task_id,
            n_samples=n_samples,
            candidates=[],
            judge_scores=[],
            generation_cost=gen_cost,
            judge_cost=CostInfo(),
            single_shot_correct=False,
            voted_correct=False,
            bon_correct=False,
            oracle_correct=False,
        )

    # Judge each candidate
    judge_tasks = [
        _judge_candidate(model, prompt, tc, i, tools)
        for i, tc in enumerate(candidates)
    ]
    judge_scores = await asyncio.gather(*judge_tasks)
    judge_scores = list(judge_scores)

    judge_cost = CostInfo(
        prompt_tokens=sum(js.prompt_tokens for js in judge_scores),
        completion_tokens=sum(js.completion_tokens for js in judge_scores),
        total_tokens=sum(js.prompt_tokens + js.completion_tokens for js in judge_scores),
        num_calls=len(judge_scores),
    )

    # Select best-of-N by highest judge score
    best_idx = max(range(len(judge_scores)), key=lambda i: judge_scores[i].score)
    bon_tc = candidates[best_idx]

    # --- Score against ground truth ---

    # Single-shot: first candidate
    first_name, first_args = _parse_tool_call_args(candidates[0])
    ss_result = check_tool_call_against_gt(first_name, first_args, ground_truth)
    single_shot_correct = ss_result["args_correct"]

    # Majority voted: use score_tool_calls from roll_up_scorer
    voted_result = score_tool_calls(candidates)
    if voted_result is not None:
        voted_check = check_tool_call_against_gt(
            voted_result.tool_name, voted_result.merged_args, ground_truth
        )
        voted_correct = voted_check["args_correct"]
    else:
        voted_correct = False

    # Best-of-N (judge-selected)
    bon_name, bon_args = _parse_tool_call_args(bon_tc)
    bon_result = check_tool_call_against_gt(bon_name, bon_args, ground_truth)
    bon_correct = bon_result["args_correct"]

    # Oracle: any candidate correct
    oracle_correct = False
    for tc in candidates:
        tc_name, tc_args = _parse_tool_call_args(tc)
        tc_result = check_tool_call_against_gt(tc_name, tc_args, ground_truth)
        if tc_result["args_correct"]:
            oracle_correct = True
            break

    return BoNTaskResult(
        task_id=task_id,
        n_samples=n_samples,
        candidates=candidates,
        judge_scores=judge_scores,
        generation_cost=gen_cost,
        judge_cost=judge_cost,
        single_shot_correct=single_shot_correct,
        voted_correct=voted_correct,
        bon_correct=bon_correct,
        oracle_correct=oracle_correct,
    )


async def run_harness(
    model: str,
    bfcl_file: str,
    max_tasks: int,
    n_samples: int,
) -> list[BoNTaskResult]:
    """Run the LLMJudge BoN harness over BFCL tasks."""
    tasks = load_bfcl_tasks(bfcl_file)
    if max_tasks:
        tasks = tasks[:max_tasks]

    gt = load_ground_truth(bfcl_file)

    results: list[BoNTaskResult] = []
    for i, task in enumerate(tasks):
        task_id = task.get("id", "?")
        logger.info("Task %d/%d: %s", i + 1, len(tasks), task_id)

        gt_entry = gt.get(task_id)
        if not gt_entry:
            logger.warning("No ground truth for %s, skipping", task_id)
            continue

        try:
            result = await process_task(model, task, gt_entry, n_samples)
            results.append(result)

            status = (
                f"SS={'Y' if result.single_shot_correct else 'N'} "
                f"Vote={'Y' if result.voted_correct else 'N'} "
                f"BoN={'Y' if result.bon_correct else 'N'} "
                f"Oracle={'Y' if result.oracle_correct else 'N'}"
            )
            logger.info("  %s", status)
        except Exception:
            logger.exception("Failed on task %s", task_id)

    return results


def print_report(results: list[BoNTaskResult], model: str) -> None:
    """Print summary report comparing all methods."""
    n = len(results)
    if n == 0:
        print("No results to report.")
        return

    ss_correct = sum(1 for r in results if r.single_shot_correct)
    voted_correct = sum(1 for r in results if r.voted_correct)
    bon_correct = sum(1 for r in results if r.bon_correct)
    oracle_correct = sum(1 for r in results if r.oracle_correct)

    gen_prompt = sum(r.generation_cost.prompt_tokens for r in results)
    gen_completion = sum(r.generation_cost.completion_tokens for r in results)
    judge_prompt = sum(r.judge_cost.prompt_tokens for r in results)
    judge_completion = sum(r.judge_cost.completion_tokens for r in results)
    total_tokens = gen_prompt + gen_completion + judge_prompt + judge_completion

    print(f"\n{'='*60}")
    print(f"LLMJudge Best-of-N Results ({n} tasks, model={model})")
    print(f"{'='*60}")

    print(f"\n--- Accuracy ---")
    print(f"  Single-shot (1st sample): {ss_correct}/{n} ({ss_correct/n:.1%})")
    print(f"  Majority voted (N=5):     {voted_correct}/{n} ({voted_correct/n:.1%})")
    print(f"  LLMJudge BoN (N=5):       {bon_correct}/{n} ({bon_correct/n:.1%})")
    print(f"  Oracle (any correct):     {oracle_correct}/{n} ({oracle_correct/n:.1%})")

    print(f"\n--- Deltas ---")
    delta_bon_ss = (bon_correct - ss_correct) / n * 100
    delta_bon_voted = (bon_correct - voted_correct) / n * 100
    headroom = (oracle_correct - bon_correct) / n * 100
    print(f"  BoN vs single-shot:    {delta_bon_ss:+.1f}pp")
    print(f"  BoN vs majority vote:  {delta_bon_voted:+.1f}pp")
    print(f"  Headroom (oracle-BoN): {headroom:+.1f}pp")

    print(f"\n--- Cost ---")
    print(f"  Generation tokens:  {gen_prompt + gen_completion:,} "
          f"(prompt={gen_prompt:,}, completion={gen_completion:,})")
    print(f"  Judge tokens:       {judge_prompt + judge_completion:,} "
          f"(prompt={judge_prompt:,}, completion={judge_completion:,})")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  Judge overhead:     {(judge_prompt + judge_completion) / max(gen_prompt + gen_completion, 1):.1%} "
          f"of generation cost")

    # Per-task breakdown for disagreements
    disagree = [
        r for r in results
        if r.bon_correct != r.voted_correct or r.bon_correct != r.single_shot_correct
    ]
    if disagree:
        print(f"\n--- Disagreements ({len(disagree)} tasks) ---")
        for r in disagree:
            scores_str = ", ".join(f"{js.score:.0f}" for js in r.judge_scores)
            print(
                f"  {r.task_id}: SS={'Y' if r.single_shot_correct else 'N'} "
                f"Vote={'Y' if r.voted_correct else 'N'} "
                f"BoN={'Y' if r.bon_correct else 'N'} "
                f"Oracle={'Y' if r.oracle_correct else 'N'} "
                f"[scores: {scores_str}]"
            )


def save_results(results: list[BoNTaskResult], model: str, bfcl_file: str) -> Path:
    """Save detailed results to JSON."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_slug = model.replace("/", "_").replace("@", "_")
    bfcl_slug = Path(bfcl_file).stem
    output_path = results_dir / f"{bfcl_slug}_{model_slug}_bon_detail.json"

    serializable = []
    for r in results:
        entry = {
            "task_id": r.task_id,
            "n_samples": r.n_samples,
            "single_shot_correct": r.single_shot_correct,
            "voted_correct": r.voted_correct,
            "bon_correct": r.bon_correct,
            "oracle_correct": r.oracle_correct,
            "judge_scores": [
                {
                    "candidate_index": js.candidate_index,
                    "score": js.score,
                    "reasoning": js.reasoning,
                }
                for js in r.judge_scores
            ],
            "candidates": r.candidates,
            "generation_cost": {
                "prompt_tokens": r.generation_cost.prompt_tokens,
                "completion_tokens": r.generation_cost.completion_tokens,
                "total_tokens": r.generation_cost.total_tokens,
            },
            "judge_cost": {
                "prompt_tokens": r.judge_cost.prompt_tokens,
                "completion_tokens": r.judge_cost.completion_tokens,
                "total_tokens": r.judge_cost.total_tokens,
            },
        }
        serializable.append(entry)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model = os.environ.get("FORECAST_MODEL", "gpt-4o-mini")
    bfcl_file = os.environ.get("BFCL_FILE", "BFCL_v4_multiple.json")
    max_tasks = int(os.environ.get("MAX_TASKS", "20"))
    n_samples = int(os.environ.get("N_SAMPLES", "5"))

    print(f"Model: {model}")
    print(f"BFCL file: {bfcl_file}")
    print(f"Max tasks: {max_tasks}")
    print(f"N samples: {n_samples}")
    print(f"Concurrency: {MAX_CONCURRENT}")

    results = asyncio.run(run_harness(model, bfcl_file, max_tasks, n_samples))

    output_path = save_results(results, model, bfcl_file)
    print(f"\nDetailed results saved to {output_path}")

    print_report(results, model)


if __name__ == "__main__":
    main()
