"""Core test runner for e2e tests."""

import time
import traceback

from its_hub.core.utils import extract_content_from_lm_response
from tests.e2e.utils.evaluation import TestResult, evaluate_answer


async def arun_test(
    algorithm,
    alg_name: str,
    lm,
    dataset: list[dict],
    dataset_name: str,
    budget: int,
    verbose: bool = False,
) -> TestResult:
    """Run *algorithm* on every problem in *dataset* using ainfer and return a `TestResult`."""
    result = TestResult(algorithm=alg_name, dataset=dataset_name, total=len(dataset))
    start = time.time()

    for i, example in enumerate(dataset):
        uid = example.get("unique_id", i)
        problem_start = time.time()
        try:
            response = await algorithm.ainfer(
                lm, example["problem"], budget, return_response_only=True
            )
            content = (
                extract_content_from_lm_response(response)
                if isinstance(response, dict)
                else response
            )
            correct = evaluate_answer(content, example["answer"])
            if correct:
                result.correct += 1
            problem_elapsed = time.time() - problem_start
            result.latencies.append(problem_elapsed)
            if verbose:
                mark = "OK" if correct else "WRONG"
                print(
                    f"    [{mark}] {dataset_name}#{uid}  "
                    f"expected={example['answer']}  ({problem_elapsed:.1f}s)"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            problem_elapsed = time.time() - problem_start
            result.latencies.append(problem_elapsed)
            result.errors += 1
            msg = f"{dataset_name}#{uid}: {e}"
            result.error_messages.append(msg)
            if verbose:
                print(f"    [ERR] {msg}  ({problem_elapsed:.1f}s)")
                traceback.print_exc()

    result.elapsed = time.time() - start
    return result


def run_test(
    algorithm,
    alg_name: str,
    lm,
    dataset: list[dict],
    dataset_name: str,
    budget: int,
    verbose: bool = False,
) -> TestResult:
    """Run *algorithm* on every problem in *dataset* and return a `TestResult`."""
    result = TestResult(algorithm=alg_name, dataset=dataset_name, total=len(dataset))
    start = time.time()

    for i, example in enumerate(dataset):
        uid = example.get("unique_id", i)
        problem_start = time.time()
        try:
            response = algorithm.infer(
                lm, example["problem"], budget, return_response_only=True
            )
            content = (
                extract_content_from_lm_response(response)
                if isinstance(response, dict)
                else response
            )
            correct = evaluate_answer(content, example["answer"])
            if correct:
                result.correct += 1
            problem_elapsed = time.time() - problem_start
            result.latencies.append(problem_elapsed)
            if verbose:
                mark = "OK" if correct else "WRONG"
                print(
                    f"    [{mark}] {dataset_name}#{uid}  "
                    f"expected={example['answer']}  ({problem_elapsed:.1f}s)"
                )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            problem_elapsed = time.time() - problem_start
            result.latencies.append(problem_elapsed)
            result.errors += 1
            msg = f"{dataset_name}#{uid}: {e}"
            result.error_messages.append(msg)
            if verbose:
                print(f"    [ERR] {msg}  ({problem_elapsed:.1f}s)")
                traceback.print_exc()

    result.elapsed = time.time() - start
    return result
