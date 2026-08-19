"""
Performance test for IaaS + Envoy integration.

Measures latency overhead and throughput for the ITS gateway stack.
Compares direct LLM access vs IaaS vs Envoy+IaaS routing.

Prerequisites:
    - LLM endpoint running (or use --mock-llm for a built-in mock)
    - IaaS service running (started automatically with --start-services)
    - Envoy running (optional, started automatically with --start-services)

Usage:
    # Quick test with mock LLM (measures gateway overhead only):
    python tests/e2e/test_iaas_envoy_perf.py --mock-llm --start-services

    # Against running services:
    python tests/e2e/test_iaas_envoy_perf.py \\
        --llm-url http://localhost:8100/v1 \\
        --iaas-url http://localhost:8109 \\
        --envoy-url http://localhost:8108 \\
        --model_name Qwen/Qwen2.5-Math-7B-Instruct \\
        --concurrency 10 --num-requests 50

    # Vary budget to see scaling:
    python tests/e2e/test_iaas_envoy_perf.py --mock-llm --start-services \\
        --budgets 1,4,8,16
"""

import argparse
import asyncio
import statistics
import sys
import time

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from tests.e2e.utils.iaas_helpers import (
    configure_iaas,
    find_free_port,
    http_post,
    start_envoy,
    start_iaas_stack,
    start_mock_llm,
    stop_processes,
)


# ---------------------------------------------------------------------------
# Async benchmark
# ---------------------------------------------------------------------------


def _compute_stats(latencies, errors, error_details, wall_time, num_requests):
    if error_details:
        print(f"    Error details (first 5):")
        for detail in error_details[:5]:
            print(f"      {detail}")

    if not latencies:
        return None

    latencies.sort()
    return {
        "count": len(latencies),
        "errors": errors,
        "wall_time_s": round(wall_time, 2),
        "rps": round(len(latencies) / wall_time, 1),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "min_ms": round(latencies[0], 1),
        "max_ms": round(latencies[-1], 1),
    }


async def benchmark_endpoint(url, model_name, num_requests, concurrency, budget=None, headers=None, timeout_s=120):
    """Send num_requests concurrent requests to an HTTP endpoint and collect latencies."""
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0
    completed = 0

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }
    if budget is not None:
        payload["budget"] = budget

    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    error_details = []

    async def _single_request(session, i):
        nonlocal errors, completed
        async with sem:
            start = time.perf_counter()
            try:
                async with session.post(url, json=payload, headers=req_headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    body = await resp.read()
                    elapsed = (time.perf_counter() - start) * 1000
                    if resp.status == 200:
                        latencies.append(elapsed)
                    else:
                        errors += 1
                        error_details.append(f"req {i}: HTTP {resp.status}: {body[:200]}")
            except Exception as e:
                errors += 1
                error_details.append(f"req {i}: {type(e).__name__}: {e}")
            completed += 1
            if completed % 10 == 0 or completed == num_requests:
                print(f"    {completed}/{num_requests} done", flush=True)

    try:
        async with aiohttp.ClientSession() as session:
            wall_start = time.perf_counter()
            tasks = [asyncio.create_task(_single_request(session, i)) for i in range(num_requests)]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_s)
            wall_time = time.perf_counter() - wall_start
    except asyncio.TimeoutError:
        wall_time = timeout_s
        cancelled = num_requests - completed
        errors += cancelled
        print(f"    TIMEOUT after {timeout_s}s ({completed}/{num_requests} completed, {cancelled} cancelled)")

    return _compute_stats(latencies, errors, error_details, wall_time, num_requests)


async def benchmark_algorithm(llm_url, model_name, api_key, num_requests, concurrency, budget, timeout_s=120):
    """Benchmark using SelfConsistency algorithm directly (no IaaS service).

    This is the true baseline: same algorithm and orchestrator as IaaS, but
    without the FastAPI/Envoy service layers. Overhead of IaaS vs this baseline
    isolates the HTTP service cost.
    """
    from its_hub.core.algorithms.self_consistency import SelfConsistency
    from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel

    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0
    completed = 0
    error_details = []

    lm = OpenAICompatibleLanguageModel(
        endpoint=llm_url, api_key=api_key, model_name=model_name,
    )
    alg = SelfConsistency()

    async def _single_request(i):
        nonlocal errors, completed
        async with sem:
            start = time.perf_counter()
            try:
                await alg.ainfer(
                    lm, "What is 2+2?", budget=budget, return_response_only=True,
                )
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
            except Exception as e:
                errors += 1
                error_details.append(f"req {i}: {type(e).__name__}: {e}")
            completed += 1
            if completed % 10 == 0 or completed == num_requests:
                print(f"    {completed}/{num_requests} done", flush=True)

    try:
        wall_start = time.perf_counter()
        tasks = [asyncio.create_task(_single_request(i)) for i in range(num_requests)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_s)
        wall_time = time.perf_counter() - wall_start
    except asyncio.TimeoutError:
        wall_time = timeout_s
        cancelled = num_requests - completed
        errors += cancelled
        print(f"    TIMEOUT after {timeout_s}s ({completed}/{num_requests} completed, {cancelled} cancelled)")
    finally:
        await lm.close()

    return _compute_stats(latencies, errors, error_details, wall_time, num_requests)


def print_stats(label, stats):
    if stats is None:
        print(f"  {label}: no results (all requests failed)")
        return
    print(f"  {label}:")
    print(f"    Requests:   {stats['count']} ok, {stats['errors']} errors")
    print(f"    Wall time:  {stats['wall_time_s']}s ({stats['rps']} req/s)")
    print(f"    Latency:    p50={stats['p50_ms']}ms  p95={stats['p95_ms']}ms  p99={stats['p99_ms']}ms")
    print(f"    Range:      min={stats['min_ms']}ms  mean={stats['mean_ms']}ms  max={stats['max_ms']}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Performance test for IaaS + Envoy integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("endpoints")
    g.add_argument("--llm-url", help="Direct LLM endpoint (e.g., http://localhost:8100/v1)")
    g.add_argument("--iaas-url", help="IaaS service URL (e.g., http://localhost:8109)")
    g.add_argument("--envoy-url", help="Envoy gateway URL (e.g., http://localhost:8108)")
    g.add_argument("--model_name", default="mock-model")
    g.add_argument("--api_key", default="NO_API_KEY")

    g = p.add_argument_group("auto-start")
    g.add_argument("--mock-llm", action="store_true", help="Start built-in mock LLM")
    g.add_argument("--mock-latency-ms", type=int, default=10, help="Mock LLM response latency (default: 10ms)")
    g.add_argument("--start-services", action="store_true", help="Auto-start IaaS + ext_proc + Envoy")

    g = p.add_argument_group("benchmark config")
    g.add_argument("--num-requests", type=int, default=50, help="Total requests per benchmark (default: 50)")
    g.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests (default: 10)")
    g.add_argument("--budgets", default="1,4", help="Comma-separated budgets to test (default: 1,4)")
    g.add_argument("--warmup", type=int, default=3, help="Warmup requests before benchmark (default: 3)")

    return p.parse_args()


def main():
    args = parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    processes = []
    servers = []
    envoy_tmpdir = None

    if not HAS_AIOHTTP:
        print("Error: aiohttp is required. Install with: pip install aiohttp")
        sys.exit(1)

    try:
        # --- Resolve endpoints ---
        llm_url = args.llm_url
        iaas_url = args.iaas_url
        envoy_url = args.envoy_url
        llm_port = None

        if args.mock_llm:
            llm_port = find_free_port()
            servers.append(start_mock_llm(llm_port, latency_ms=args.mock_latency_ms))
            llm_url = f"http://127.0.0.1:{llm_port}/v1"
            print(f"Mock LLM started on port {llm_port} (latency={args.mock_latency_ms}ms)")

        if args.start_services:
            if not llm_url:
                print("Error: --start-services requires --llm-url or --mock-llm")
                sys.exit(1)

            from urllib.parse import urlparse
            if llm_port is None:
                llm_port = urlparse(llm_url).port or 80

            stack_procs, iaas_url, ext_proc_port = start_iaas_stack(llm_port)
            processes.extend(stack_procs)

            iaas_port = urlparse(iaas_url).port
            envoy_result = start_envoy(ext_proc_port, iaas_port, llm_port)
            if envoy_result:
                envoy_proc, envoy_url, envoy_tmpdir, envoy_admin_port = envoy_result
                processes.append(("envoy", envoy_proc))
            else:
                envoy_admin_port = None
                print("Envoy not available, skipping Envoy benchmarks")

        # --- Configure IaaS ---
        if iaas_url:
            if not llm_url:
                print("Error: --iaas-url requires --llm-url or --mock-llm")
                sys.exit(1)
            configure_iaas(iaas_url, llm_url, args.model_name, args.api_key)
            print("IaaS configured")

        # --- Warmup ---
        print(f"\nWarming up ({args.warmup} requests per endpoint)...")
        for _ in range(args.warmup):
            if llm_url:
                http_post(f"{llm_url}/chat/completions", {
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                })
            if iaas_url:
                http_post(f"{iaas_url}/v1/chat/completions", {
                    "model": args.model_name,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "budget": 1,
                })

        # --- Benchmarks ---
        print(f"\nBenchmark: {args.num_requests} requests, concurrency={args.concurrency}")
        print("=" * 60)

        for budget in budgets:
            print(f"\n--- budget={budget} ---")

            baseline_p50 = None

            if llm_url:
                print(f"\n[Algorithm direct, budget={budget}]")
                stats = asyncio.run(benchmark_algorithm(
                    llm_url, args.model_name, args.api_key,
                    args.num_requests, args.concurrency, budget=budget,
                ))
                print_stats(f"algorithm(budget={budget})", stats)
                baseline_p50 = stats["p50_ms"] if stats else None

            if iaas_url:
                print(f"\n[IaaS, budget={budget}]")
                stats = asyncio.run(benchmark_endpoint(
                    f"{iaas_url}/v1/chat/completions",
                    args.model_name,
                    args.num_requests,
                    args.concurrency,
                    budget=budget,
                ))
                print_stats(f"iaas(budget={budget})", stats)
                if stats and baseline_p50:
                    overhead = stats["p50_ms"] - baseline_p50
                    print(f"    Overhead vs direct: {overhead:+.1f}ms (p50)")

            if envoy_url:
                print(f"\n[Envoy -> IaaS, budget={budget}]")
                stats = asyncio.run(benchmark_endpoint(
                    f"{envoy_url}/v1/chat/completions",
                    args.model_name,
                    args.num_requests,
                    args.concurrency,
                    headers={
                        "X-ITS-Budget": str(budget),
                        "X-ITS-Endpoint": llm_url,
                        "X-ITS-API-Key": args.api_key,
                    },
                ))
                print_stats(f"envoy(budget={budget})", stats)
                if stats and baseline_p50:
                    overhead = stats["p50_ms"] - baseline_p50
                    print(f"    Overhead vs direct: {overhead:+.1f}ms (p50)")

        if envoy_url:
            print("\n--- pass-through ---")
            print("\n[Envoy pass-through (no ITS)]")
            stats = asyncio.run(benchmark_endpoint(
                f"{envoy_url}/v1/chat/completions",
                args.model_name,
                args.num_requests,
                args.concurrency,
            ))
            print_stats("envoy-passthrough", stats)

        print("\n" + "=" * 60)
        print("Done.")

    finally:
        print("\nShutting down services...")
        stop_processes(processes)
        for server in servers:
            server.shutdown()
        if envoy_tmpdir:
            import shutil
            shutil.rmtree(envoy_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
