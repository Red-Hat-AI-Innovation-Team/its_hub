"""AIME 2024 evaluation: 5 independent runs for statistical significance.
4 algorithms × 30 problems × 5 runs.
Saves per-run results incrementally and resumes from last completed run.
"""
import asyncio
import gc
import json
import logging
import os
import re
import time

from datasets import load_dataset
from openai import AsyncOpenAI

from its_hub.api import (
    AbstractOrchestrator,
    ChatMessage,
    GenerationUsage,
)
from its_hub.core.algorithms.self_consistency import SelfConsistency
from its_hub.core.algorithms.adaptive_self_consistency import AdaptiveSelfConsistency
from its_hub.core.algorithms.beta_self_consistency import BetaSelfConsistency

VLLM_URL = "http://localhost:8200/v1"
MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"
BUDGET = 16
N_QUESTIONS = 30
TEMPERATURE = 0.7
MAX_COMPLETION_TOKENS = 2048
MAX_CONCURRENCY = 32
COOLDOWN_ALGO = 5
COOLDOWN_RUN = 10
N_RUNS = 10
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


class AsyncOrchestrator(AbstractOrchestrator):
    """Orchestrator using pure asyncio.Semaphore (no thread pool)."""
    def __init__(self, max_concurrency=32):
        self._max_concurrency = max_concurrency

    async def agenerate(self, lm, messages_lst, **kwargs):
        if not messages_lst:
            return []
        sem = asyncio.Semaphore(self._max_concurrency)
        usage = kwargs.get("usage_accumulator")

        async def gen_one(msgs):
            async with sem:
                return await lm.agenerate_single(
                    msgs,
                    stop=kwargs.get("stop"),
                    max_completion_tokens=kwargs.get("max_completion_tokens"),
                    temperature=kwargs.get("temperature"),
                    tools=kwargs.get("tools"),
                    tool_choice=kwargs.get("tool_choice"),
                    usage_accumulator=usage,
                )

        return await asyncio.gather(*(gen_one(m) for m in messages_lst))


def extract_boxed(text):
    """Extract integer answer from \\boxed{...} in model output."""
    if text is None:
        return None
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        ans = matches[-1].strip().replace(",", "").replace(" ", "")
        ans = re.sub(r"[^\d]", "", ans)
        if ans:
            try:
                return str(int(ans))
            except (ValueError, OverflowError):
                pass
    numbers = re.findall(r"(?:answer is|=)\s*\$?(\d+)\$?", text, re.IGNORECASE)
    if numbers:
        return str(int(numbers[-1]))
    return None


def normalize_answer(ans):
    """Normalize AIME answer to integer string (strip leading zeros)."""
    if ans is None:
        return None
    ans = str(ans).strip().replace(",", "")
    ans = re.sub(r"[^\d]", "", ans)
    if not ans:
        return None
    return str(int(ans))


def projection_func(text):
    ans = extract_boxed(text)
    return ans if ans is not None else text.strip()


SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class OpenAIAdapterLM:
    """Wraps openai.AsyncOpenAI to match the its_hub LM interface."""
    def __init__(self, client, model, temperature, max_tokens):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def agenerate_single(self, messages, **kwargs):
        temp = kwargs.get("temperature") or self._temperature
        max_tokens = kwargs.get("max_completion_tokens") or self._max_tokens
        msg_dicts = []
        for m in messages:
            if isinstance(m, dict):
                msg_dicts.append(m)
            else:
                msg_dicts.append({"role": m.role, "content": m.content})
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=msg_dicts,
            temperature=temp,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        result = {"role": "assistant", "content": choice.message.content or ""}
        if kwargs.get("usage_accumulator") is not None:
            u = kwargs["usage_accumulator"]
            u.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return result


async def run_single_algo(algo_name, algo, lm, questions, budget):
    """Run one algorithm over all questions. Returns result summary dict."""
    correct = 0
    total_samples = 0
    per_question = []
    start_time = time.time()

    for i, item in enumerate(questions):
        q = item["problem"]
        gold = normalize_answer(item["answer"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        try:
            if algo is None:
                response = await lm.agenerate_single(messages)
                content = response.get("content", "")
                n_samples = 1
            else:
                result = await algo.ainfer(
                    lm, messages, budget=budget, return_response_only=False
                )
                content = result.the_one.get("content", "")
                n_samples = len(result.responses)

            pred = extract_boxed(content)
            pred_norm = normalize_answer(pred)
            is_correct = (pred_norm is not None) and (pred_norm == gold)
            if is_correct:
                correct += 1
            total_samples += n_samples
            per_question.append({
                "question_idx": i,
                "unique_id": item.get("unique_id", f"q{i}"),
                "gold": gold,
                "predicted": pred_norm,
                "correct": is_correct,
                "n_samples": n_samples,
            })
            elapsed = time.time() - start_time
            acc = correct / (i + 1) * 100
            avg_s = total_samples / (i + 1)
            print(f"    [{i+1}/{N_QUESTIONS}] acc={acc:.1f}% "
                  f"avg_samples={avg_s:.1f} time={elapsed:.1f}s")
        except Exception as e:
            print(f"    ERROR on question {i}: {e}")
            import traceback
            traceback.print_exc()
            per_question.append({
                "question_idx": i,
                "unique_id": item.get("unique_id", f"q{i}"),
                "gold": gold,
                "predicted": None,
                "correct": False,
                "n_samples": 0,
                "error": str(e),
            })

    elapsed = time.time() - start_time
    accuracy = correct / N_QUESTIONS * 100
    avg_samples = total_samples / N_QUESTIONS

    return {
        "algorithm": algo_name, "budget": budget,
        "accuracy": accuracy, "correct": correct, "total": N_QUESTIONS,
        "avg_samples": avg_samples, "total_samples": total_samples,
        "total_time_s": round(elapsed, 2),
        "avg_time_per_question_s": round(elapsed / N_QUESTIONS, 2),
        "per_question": per_question,
    }


def get_completed_runs():
    """Return set of run numbers that already have result files."""
    completed = set()
    for i in range(1, N_RUNS + 1):
        path = os.path.join(RESULTS_DIR, f"aime_results_run{i}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if len(data) == 4:
                    completed.add(i)
                    print(f"  Run {i}: already complete (skipping)")
                else:
                    print(f"  Run {i}: incomplete ({len(data)}/4 algos), will re-run")
            except (json.JSONDecodeError, KeyError):
                print(f"  Run {i}: corrupt file, will re-run")
    return completed


def build_algo_specs():
    """Create fresh algorithm instances (one orchestrator per algo)."""
    return [
        ("no_its", None, 1),
        ("self_consistency",
         SelfConsistency(
             consistency_space_projection_func=projection_func,
             orchestrator=AsyncOrchestrator(max_concurrency=MAX_CONCURRENCY),
         ), BUDGET),
        ("adaptive_sc",
         AdaptiveSelfConsistency(
             threshold=0.75,
             consistency_space_projection_func=projection_func,
             orchestrator=AsyncOrchestrator(max_concurrency=MAX_CONCURRENCY),
         ), BUDGET),
        ("beta_sc",
         BetaSelfConsistency(
             confidence_threshold=0.95,
             consistency_space_projection_func=projection_func,
             orchestrator=AsyncOrchestrator(max_concurrency=MAX_CONCURRENCY),
         ), BUDGET),
    ]


def compute_aggregate(all_run_results):
    """Compute mean and std across runs for each algorithm."""
    import numpy as np
    algo_names = ["no_its", "self_consistency", "adaptive_sc", "beta_sc"]
    aggregate = {}
    for algo in algo_names:
        accuracies = [r[algo]["accuracy"] for r in all_run_results]
        avg_samples_list = [r[algo]["avg_samples"] for r in all_run_results]
        times = [r[algo]["total_time_s"] for r in all_run_results]
        aggregate[algo] = {
            "accuracy_mean": round(float(np.mean(accuracies)), 2),
            "accuracy_std": round(float(np.std(accuracies, ddof=1)), 2),
            "avg_samples_mean": round(float(np.mean(avg_samples_list)), 2),
            "avg_samples_std": round(float(np.std(avg_samples_list, ddof=1)), 2),
            "total_time_mean": round(float(np.mean(times)), 2),
            "total_time_std": round(float(np.std(times, ddof=1)), 2),
            "per_run_accuracy": accuracies,
            "per_run_avg_samples": avg_samples_list,
            "per_run_total_time": times,
        }
    return aggregate


def load_aime2024():
    """Load AIME 2024 dataset (all 30 problems).
    HF dataset has uppercase columns: ID, Problem, Solution, Answer.
    """
    ds = load_dataset("Maxwell-Jia/AIME_2024")["train"]
    questions = []
    for item in ds:
        problem = item.get("Problem") or item.get("problem")
        answer = item.get("Answer") or item.get("answer")
        uid = item.get("ID") or item.get("id") or item.get("unique_id", "")
        questions.append({
            "problem": problem,
            "answer": str(answer),
            "unique_id": str(uid),
        })
    print(f"Loaded {len(questions)} AIME 2024 problems")
    for i, q in enumerate(questions[:3]):
        print(f"  Sample {i}: id={q['unique_id']}, answer={q['answer']}, "
              f"problem={q['problem'][:60]}...")
    return questions


async def main():
    logging.basicConfig(level=logging.WARNING)
    print("Loading AIME 2024 dataset...")
    questions = load_aime2024()

    if len(questions) != N_QUESTIONS:
        print(f"WARNING: Expected {N_QUESTIONS} questions, got {len(questions)}")

    print(f"\nChecking for completed runs...")
    completed = get_completed_runs()
    remaining = [i for i in range(1, N_RUNS + 1) if i not in completed]

    if not remaining:
        print("All runs already complete!")
    else:
        print(f"Runs to execute: {remaining}")

    for run_idx in remaining:
        print(f"\n{'#'*70}")
        print(f"# RUN {run_idx}/{N_RUNS}")
        print(f"{'#'*70}")

        algo_specs = build_algo_specs()
        run_results = {}

        for j, (algo_name, algo, budget) in enumerate(algo_specs):
            print(f"\n  === {algo_name} (budget={budget}) ===")

            client = AsyncOpenAI(
                base_url=VLLM_URL,
                api_key="NO_API_KEY",
                max_retries=5,
                timeout=120.0,
            )
            lm = OpenAIAdapterLM(client, MODEL, TEMPERATURE, MAX_COMPLETION_TOKENS)

            result = await run_single_algo(algo_name, algo, lm, questions, budget)
            run_results[algo_name] = result

            await client.close()
            gc.collect()

            print(f"    DONE: acc={result['accuracy']:.1f}% "
                  f"samples={result['avg_samples']:.1f} "
                  f"time={result['total_time_s']:.1f}s")

            if j < len(algo_specs) - 1:
                await asyncio.sleep(COOLDOWN_ALGO)

        run_path = os.path.join(RESULTS_DIR, f"aime_results_run{run_idx}.json")
        with open(run_path, "w") as f:
            json.dump(run_results, f, indent=2)
        print(f"\n  Run {run_idx} saved to {run_path}")

        if run_idx < N_RUNS:
            print(f"  Cooling down {COOLDOWN_RUN}s before next run...")
            await asyncio.sleep(COOLDOWN_RUN)

    # Aggregate all runs
    print(f"\n{'='*70}")
    print("AGGREGATING ALL RUNS")
    print(f"{'='*70}")

    all_run_results = []
    for i in range(1, N_RUNS + 1):
        path = os.path.join(RESULTS_DIR, f"aime_results_run{i}.json")
        with open(path) as f:
            all_run_results.append(json.load(f))

    aggregate = compute_aggregate(all_run_results)

    agg_path = os.path.join(RESULTS_DIR, "aime_results_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregate saved to {agg_path}")

    print(f"\n{'Algorithm':<20} {'Accuracy':>16} {'Avg Samples':>16} {'Time (s)':>16}")
    print(f"{'-'*70}")
    for algo in ["no_its", "self_consistency", "adaptive_sc", "beta_sc"]:
        a = aggregate[algo]
        print(f"{algo:<20} "
              f"{a['accuracy_mean']:>6.1f} ± {a['accuracy_std']:<5.1f} "
              f"{a['avg_samples_mean']:>6.1f} ± {a['avg_samples_std']:<5.1f} "
              f"{a['total_time_mean']:>6.1f} ± {a['total_time_std']:<5.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
