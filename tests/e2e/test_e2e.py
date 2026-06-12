"""
End-to-end test framework for its_hub algorithms.

Tests the particle-filtering algorithms (PF and EPF) against pre-saved subsets
of MATH500 and AIME-2024 datasets using an OpenAI-compatible API endpoint.
The endpoint must support ``logprobs`` (vLLM does) — particle weights come
from the generator's own token logprobs (self-certainty).

Two test modes:
  - async (default): uses algorithm.ainfer() — single event loop
  - sync  (--sync):  uses algorithm.infer()  — each problem gets its own event loop

Usage:
    # Async mode (default, ainfer):
    python tests/e2e/test_e2e.py --endpoint http://localhost:8100/v1 \
        --model_name Qwen/Qwen2.5-Math-7B-Instruct

    # Sync mode (infer):
    python tests/e2e/test_e2e.py --endpoint http://localhost:8100/v1 \
        --model_name Qwen/Qwen2.5-Math-7B-Instruct --sync

    # Select specific algorithms / datasets:
    python tests/e2e/test_e2e.py --endpoint http://localhost:8100/v1 \
        --model_name Qwen/Qwen2.5-Math-7B-Instruct \
        --algorithms particle-filtering --datasets math500
"""

import argparse
import asyncio
import os
import sys
import time

from its_hub import OpenAICompatibleLanguageModel
from its_hub.core.utils import QWEN_SYSTEM_PROMPT, SAL_STEP_BY_STEP_SYSTEM_PROMPT
from tests.e2e.utils.algorithms import ALL_ALGORITHM_NAMES, build_algorithms
from tests.e2e.utils.datasets import load_datasets
from tests.e2e.utils.evaluation import TestResult
from tests.e2e.utils.report import print_report
from tests.e2e.utils.runner import arun_test, run_test


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E test framework for its_hub algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- required ---
    p.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible API endpoint (e.g. http://localhost:8100/v1)",
    )
    p.add_argument(
        "--model_name", required=True, help="Model name served at the endpoint"
    )

    # --- optional LM config ---
    p.add_argument(
        "--api_key", default="NO_API_KEY", help="API key (default: NO_API_KEY)"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--max_tokens", type=int, default=None, help="Max tokens per generation"
    )
    p.add_argument(
        "--max_concurrency",
        type=int,
        default=32,
        help="Max concurrent requests (default: 32)",
    )

    # --- step generation ---
    p.add_argument(
        "--tokens_per_step",
        type=int,
        default=None,
        help="Tokens per step (alternative to step_token for StepGeneration)",
    )

    # --- test config ---
    p.add_argument(
        "--budget",
        type=int,
        default=4,
        help="Computation budget per problem (default: 4)",
    )
    p.add_argument(
        "--datasets",
        default="math500,aime2024",
        help="Comma-separated list of datasets (default: math500,aime2024)",
    )
    p.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated list of algorithms to test (default: all available). "
        "Options: " + ", ".join(ALL_ALGORITHM_NAMES),
    )
    p.add_argument(
        "--verbose", action="store_true", help="Print per-problem results"
    )
    p.add_argument(
        "--sync",
        dest="use_sync",
        action="store_true",
        default=False,
        help="Use sync infer instead of async ainfer (default: async with shared orchestrator)",
    )

    return p.parse_args()


def _print_result(r: TestResult):
    status = "PASS" if r.passed else "FAIL"
    print(
        f"  => {status}: {r.correct}/{r.evaluated} correct "
        f"({r.accuracy:.1%}), {r.errors} error(s), "
        f"{r.elapsed:.1f}s  "
        f"[avg={r.avg_latency:.2f}s min={r.min_latency:.2f}s "
        f"max={r.max_latency:.2f}s]\n"
    )


def run_sync_tests(algs, loaded, lm, budget, verbose):
    """Run all tests using algorithm.infer() (sync)."""
    results: list[TestResult] = []
    try:
        for ds_name, dataset in loaded.items():
            for alg_name, algorithm in algs.items():
                print(f"Running {alg_name} on {ds_name} (budget={budget})...")
                r = run_test(
                    algorithm, alg_name, lm, dataset, ds_name, budget, verbose,
                )
                results.append(r)
                _print_result(r)
    except KeyboardInterrupt:
        print("\nInterrupted -- reporting partial results")
    return results


async def run_async_tests(algs, loaded, lm, budget, verbose):
    """Run all tests using algorithm.ainfer() with a shared orchestrator."""
    results: list[TestResult] = []
    try:
        for ds_name, dataset in loaded.items():
            for alg_name, algorithm in algs.items():
                print(f"Running {alg_name} on {ds_name} (budget={budget})...")
                r = await arun_test(
                    algorithm, alg_name, lm, dataset, ds_name, budget, verbose,
                )
                results.append(r)
                _print_result(r)
    except KeyboardInterrupt:
        print("\nInterrupted -- reporting partial results")
    return results


def main():
    args = parse_args()

    mode = "sync (infer)" if args.use_sync else "async (ainfer)"
    print("=" * 60)
    print("its_hub E2E Test Framework")
    print("=" * 60)
    print(f"  endpoint:          {args.endpoint}")
    print(f"  model_name:        {args.model_name}")
    print(f"  budget:            {args.budget}")
    print(f"  temperature:       {args.temperature}")
    print(f"  max_concurrency:   {args.max_concurrency}")
    print(f"  datasets:          {args.datasets}")
    print(f"  algorithms:        {args.algorithms or 'all available'}")
    print(f"  mode:              {mode}")
    print()

    # ---- create language model ----
    system_prompt = (
        QWEN_SYSTEM_PROMPT
        if "qwen" in args.model_name.lower()
        else SAL_STEP_BY_STEP_SYSTEM_PROMPT
    )
    lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model_name=args.model_name,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
    )

    # ---- build algorithms ----
    print("Initializing algorithms...")
    all_algs = build_algorithms(args.model_name, args.tokens_per_step)

    # Filter to requested algorithms if specified
    if args.algorithms:
        requested = [a.strip() for a in args.algorithms.split(",")]
        unknown = set(requested) - set(ALL_ALGORITHM_NAMES)
        if unknown:
            print(f"  Warning: unknown algorithm(s) ignored: {unknown}")
        algs = {k: v for k, v in all_algs.items() if k in requested}
        if not algs:
            sys.exit("Error: no valid algorithms selected")
    else:
        algs = all_algs

    print(f"  Testing: {', '.join(algs.keys())}\n")

    # ---- load datasets ----
    requested_datasets = [d.strip() for d in args.datasets.split(",")]
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    loaded = load_datasets(requested_datasets, data_dir)
    if not loaded:
        sys.exit("Error: no datasets loaded")
    print()

    # ---- run tests ----
    total_start = time.time()

    async def _run_async_and_cleanup():
        try:
            return await run_async_tests(algs, loaded, lm, args.budget, args.verbose)
        finally:
            await lm.close()

    if args.use_sync:
        results = run_sync_tests(algs, loaded, lm, args.budget, args.verbose)
        asyncio.run(lm.close())
    else:
        results = asyncio.run(_run_async_and_cleanup())

    total_elapsed = time.time() - total_start

    # ---- report ----
    if results:
        print_report(results)
        print(f"\nTotal time: {total_elapsed:.1f}s")

    # ---- exit code ----
    failed = any(not r.passed for r in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
