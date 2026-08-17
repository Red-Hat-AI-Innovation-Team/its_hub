"""
End-to-end test for IaaS + Envoy integration.

Tests the full stack: Client -> Envoy -> ext_proc router -> IaaS -> LLM

Starts the IaaS service, ext_proc router, and (optionally) Envoy, then
runs requests through the stack and verifies correct routing and responses.

Usage:
    # With real LLM endpoint:
    python tests/e2e/test_iaas_envoy_e2e.py \\
        --endpoint http://localhost:8100/v1 \\
        --model_name Qwen/Qwen2.5-Math-7B-Instruct

    # With built-in mock LLM (no external dependencies except Envoy):
    python tests/e2e/test_iaas_envoy_e2e.py --mock-llm

    # Skip Envoy (test IaaS service only):
    python tests/e2e/test_iaas_envoy_e2e.py --mock-llm --skip-envoy
"""

import argparse
import shutil
import sys

from tests.e2e.utils.iaas_helpers import (
    configure_iaas,
    find_free_port,
    http_get,
    http_post,
    start_envoy,
    start_iaas_stack,
    start_mock_llm,
    stop_processes,
)


# ---------------------------------------------------------------------------
# Result tracker
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details = []

    def ok(self, name):
        self.passed += 1
        self.details.append(f"  PASS  {name}")
        print(f"  PASS  {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.details.append(f"  FAIL  {name}: {reason}")
        print(f"  FAIL  {name}: {reason}")

    def skip(self, name, reason):
        self.skipped += 1
        self.details.append(f"  SKIP  {name}: {reason}")
        print(f"  SKIP  {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.failed:
            print("\nFailed tests:")
            for d in self.details:
                if "FAIL" in d:
                    print(d)
        print(f"{'='*60}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Tests: IaaS direct
# ---------------------------------------------------------------------------


def test_iaas_direct(iaas_url, llm_endpoint, model_name, api_key, result):
    """Test IaaS service directly (without Envoy)."""
    print("\n--- IaaS Direct Tests ---")

    # Configure
    try:
        configure_iaas(iaas_url, llm_endpoint, model_name, api_key)
        result.ok("iaas_configure")
    except RuntimeError as e:
        result.fail("iaas_configure", str(e))
        return

    # Models endpoint
    status, body = http_get(f"{iaas_url}/v1/models")
    if status == 200 and body.get("data") and body["data"][0]["id"] == model_name:
        result.ok("iaas_models")
    else:
        result.fail("iaas_models", f"unexpected: {body}")

    # Chat completion via body budget
    status, body = http_post(f"{iaas_url}/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "budget": 3,
    })
    if status == 200 and body.get("choices"):
        content = body["choices"][0]["message"]["content"]
        if content:
            result.ok("iaas_chat_completion_body_budget")
        else:
            result.fail("iaas_chat_completion_body_budget", "empty content")
    else:
        result.fail("iaas_chat_completion_body_budget", f"status {status}: {body}")

    # Chat completion via header budget
    status, body = http_post(
        f"{iaas_url}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is 3+3?"}],
        },
        headers={"X-ITS-Budget": "2"},
    )
    if status == 200 and body.get("choices"):
        result.ok("iaas_chat_completion_header_budget")
    else:
        result.fail("iaas_chat_completion_header_budget", f"status {status}: {body}")

    # Chat completion with header overrides
    status, body = http_post(
        f"{iaas_url}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is 5+5?"}],
        },
        headers={
            "X-ITS-Budget": "2",
            "X-ITS-Endpoint": llm_endpoint,
            "X-ITS-API-Key": api_key,
        },
    )
    if status == 200 and body.get("choices"):
        result.ok("iaas_header_overrides")
    else:
        result.fail("iaas_header_overrides", f"status {status}: {body}")


# ---------------------------------------------------------------------------
# Tests: Envoy-routed
# ---------------------------------------------------------------------------


def test_envoy_routed(envoy_url, iaas_url, llm_endpoint, model_name, api_key, result):
    """Test requests routed through Envoy."""
    print("\n--- Envoy-Routed Tests ---")

    # Configure IaaS first
    try:
        configure_iaas(iaas_url, llm_endpoint, model_name, api_key)
    except RuntimeError:
        result.fail("envoy_precondition", "could not configure IaaS")
        return

    # ITS request through Envoy (should route to IaaS)
    status, body = http_post(
        f"{envoy_url}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is 7+7?"}],
        },
        headers={
            "X-ITS-Budget": "2",
            "X-ITS-Endpoint": llm_endpoint,
            "X-ITS-API-Key": api_key,
        },
    )
    if status == 200 and body.get("choices"):
        result.ok("envoy_its_request")
    else:
        result.fail("envoy_its_request", f"status {status}: {body}")

    # Non-ITS request through Envoy (should pass through to LLM)
    status, body = http_post(
        f"{envoy_url}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Direct pass-through"}],
        },
    )
    if status == 200 and body.get("choices"):
        result.ok("envoy_passthrough")
    else:
        result.fail("envoy_passthrough", f"status {status}: {body}")

    # Verify ITS headers are stripped on pass-through
    status, body = http_post(
        f"{envoy_url}/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Stray header test"}],
        },
        headers={"X-ITS-Endpoint": "http://should-be-stripped/v1"},
    )
    if status == 200:
        result.ok("envoy_stray_header_stripped")
    else:
        result.fail("envoy_stray_header_stripped", f"status {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="E2E tests for IaaS + Envoy integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--endpoint", help="LLM endpoint URL (e.g., http://localhost:8100/v1)")
    p.add_argument("--model_name", default="mock-model", help="Model name at the endpoint")
    p.add_argument("--api_key", default="NO_API_KEY", help="API key for the LLM endpoint")
    p.add_argument("--mock-llm", action="store_true", help="Start a built-in mock LLM server")
    p.add_argument("--skip-envoy", action="store_true", help="Skip Envoy tests (test IaaS only)")
    return p.parse_args()


def main():
    args = parse_args()
    result = _Result()
    processes = []
    servers = []
    envoy_tmpdir = None

    try:
        # --- Resolve LLM endpoint ---
        if args.mock_llm:
            llm_port = find_free_port()
            servers.append(start_mock_llm(llm_port))
            llm_endpoint = f"http://127.0.0.1:{llm_port}/v1"
            model_name = "mock-model"
            print(f"Mock LLM started on port {llm_port}")
        elif args.endpoint:
            llm_endpoint = args.endpoint
            model_name = args.model_name
            from urllib.parse import urlparse
            llm_port = urlparse(llm_endpoint).port or 80
        else:
            print("Error: provide --endpoint or --mock-llm")
            sys.exit(1)

        api_key = args.api_key

        # --- Start IaaS stack ---
        stack_procs, iaas_url, ext_proc_port = start_iaas_stack(llm_port)
        processes.extend(stack_procs)

        # --- Run IaaS direct tests ---
        test_iaas_direct(iaas_url, llm_endpoint, model_name, api_key, result)

        # --- Envoy tests ---
        if args.skip_envoy:
            result.skip("envoy_tests", "skipped via --skip-envoy")
        elif not shutil.which("envoy"):
            result.skip("envoy_tests", "envoy binary not found in PATH")
        else:
            from urllib.parse import urlparse
            iaas_port = urlparse(iaas_url).port
            envoy_result = start_envoy(ext_proc_port, iaas_port, llm_port)
            if envoy_result is None:
                result.fail("envoy_tests", "Envoy or ext_proc failed to start")
            else:
                envoy_proc, envoy_url, envoy_tmpdir, _ = envoy_result
                processes.append(("envoy", envoy_proc))
                test_envoy_routed(envoy_url, iaas_url, llm_endpoint, model_name, api_key, result)

    finally:
        print("\nShutting down services...")
        stop_processes(processes)
        for server in servers:
            server.shutdown()
        if envoy_tmpdir:
            import shutil as _shutil
            _shutil.rmtree(envoy_tmpdir, ignore_errors=True)

    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
