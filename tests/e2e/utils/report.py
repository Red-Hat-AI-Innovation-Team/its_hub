"""Reporting utilities for e2e tests."""

from tests.e2e.utils.evaluation import TestResult

LINE_W = 120
SEP = "-" * LINE_W


def print_report(results: list[TestResult]) -> None:
    """Print a summary table of all test results."""
    print()
    print("=" * LINE_W)
    print(
        f"{'DATASET':>12s}   {'ALGORITHM':30s}   "
        f"{'TOTAL':>5s}  {'OK':>4s}  {'ERR':>4s}  {'ACC':>7s}  {'TIME':>7s}  STATUS"
    )
    print(SEP)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{r.dataset:>12s}   {r.algorithm:30s}   {r.total:5d}  {r.correct:4d}  "
            f"{r.errors:4d}  {r.accuracy:6.1%}  {r.elapsed:6.1f}s  {status}"
        )
    print(SEP)

    # Latency summary
    print()
    print("Latency Summary (seconds per problem):")
    print(SEP)
    print(
        f"{'DATASET':>12s}   {'ALGORITHM':30s}   "
        f"{'AVG':>7s}  {'MIN':>7s}  {'MAX':>7s}"
    )
    print(SEP)
    for r in results:
        if r.latencies:
            print(
                f"{r.dataset:>12s}   {r.algorithm:30s}   "
                f"{r.avg_latency:6.2f}s  {r.min_latency:6.2f}s  {r.max_latency:6.2f}s"
            )
    print(SEP)

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} FAILED test(s):")
        for r in failed:
            print(f"  - {r.dataset} / {r.algorithm} ({r.errors} error(s))")
            for msg in r.error_messages[:5]:
                print(f"      {msg}")
            if len(r.error_messages) > 5:
                print(f"      ... and {len(r.error_messages) - 5} more")
    else:
        print(f"\nAll {len(results)} test(s) PASSED")
