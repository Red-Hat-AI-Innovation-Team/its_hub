"""Shared helpers for IaaS + Envoy e2e and performance tests."""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.conftest import find_free_port


def wait_for_port(port, host="127.0.0.1", timeout=15):
    """Wait until a port is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_for_http(url, timeout=15):
    """Wait until an HTTP endpoint returns 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# HTTP helpers (urllib-based, no extra dependencies)
# ---------------------------------------------------------------------------


def _decode_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"raw": raw.decode(errors="replace")[:500]}


def http_post(url, data, headers=None, timeout=30):
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode_json(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, _decode_json(e.read())
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, _decode_json(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, _decode_json(e.read())
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}


# ---------------------------------------------------------------------------
# Mock LLM server
# ---------------------------------------------------------------------------


class MockLLMHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible mock LLM with configurable latency."""

    latency_ms = 0

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            if self.latency_ms:
                time.sleep(self.latency_ms / 1000.0)

            messages = body.get("messages", [])
            user_msg = messages[-1]["content"] if messages else "unknown"

            its_headers = {
                k: "<redacted>" if k.lower() == "x-its-api-key" else v
                for k, v in self.headers.items()
                if k.lower().startswith("x-its-")
            }

            response = {
                "id": "mock-llm-001",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock LLM response to: {user_msg}",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                "its_headers_received": its_headers,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/v1/models":
            response = {"data": [{"id": "mock-model", "object": "model"}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_mock_llm(port, latency_ms=0):
    MockLLMHandler.latency_ms = latency_ms
    server = ThreadingHTTPServer(("127.0.0.1", port), MockLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Envoy config generation
# ---------------------------------------------------------------------------


_PROD_ENVOY_CONFIG = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "its_hub", "integration", "iaas", "envoy_config.yaml",
)


def generate_envoy_config(envoy_port, ext_proc_port, iaas_port, llm_port):
    """Load the production envoy_config.yaml and adapt it for local testing."""
    import yaml

    admin_port = find_free_port()

    with open(_PROD_ENVOY_CONFIG) as f:
        config = yaml.safe_load(f)

    # --- Admin: simple, no access log ---
    config["admin"] = {
        "address": {
            "socket_address": {"address": "127.0.0.1", "port_value": admin_port},
        },
    }

    # --- Listener port ---
    listener = config["static_resources"]["listeners"][0]
    sock_addr = listener["address"]["socket_address"]
    sock_addr["port_value"] = envoy_port
    sock_addr.pop("protocol", None)

    # --- HTTP connection manager tweaks ---
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    hcm.pop("access_log", None)

    for hf in hcm.get("http_filters", []):
        if "ext_proc" in hf.get("name", ""):
            ep = hf["typed_config"]
            ep["failure_mode_allow"] = False
            ep["message_timeout"] = "10s"
            ep["grpc_service"]["timeout"] = "10s"

    for vh in hcm.get("route_config", {}).get("virtual_hosts", []):
        for route in vh.get("routes", []):
            if "route" in route:
                route["route"]["timeout"] = "60s"

    # --- Clusters: set test ports, remove health checks, use STATIC ---
    port_map = {
        "ext_proc_cluster": ext_proc_port,
        "iaas_upstream": iaas_port,
        "llm_upstream": llm_port,
    }
    for cluster in config["static_resources"]["clusters"]:
        name = cluster["name"]
        if name in port_map:
            cluster["type"] = "STATIC"
            cluster.pop("dns_lookup_family", None)
            cluster.pop("health_checks", None)
            ep = cluster["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"]
            ep["address"]["socket_address"]["address"] = "127.0.0.1"
            ep["address"]["socket_address"]["port_value"] = port_map[name]

    return yaml.dump(config, default_flow_style=False, sort_keys=False), admin_port


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


def start_iaas_stack(llm_port):
    """Start ext_proc + IaaS service. Returns (processes, iaas_url, ext_proc_port).

    Caller is responsible for calling stop_processes(processes) in a finally block.
    """
    processes = []

    ext_proc_port = find_free_port()
    ext_proc_proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import sys; sys.argv = ['its-iaas-ext-proc', '--port', '{ext_proc_port}']; "
         f"from its_hub.integration.iaas.grpc_server import main; main()"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    processes.append(("ext_proc", ext_proc_proc))

    iaas_port = find_free_port()
    iaas_proc = subprocess.Popen(
        [sys.executable, "-c",
         f"import sys; sys.argv = ['its-iaas', '--port', '{iaas_port}']; "
         f"from its_hub.integration.iaas.app_server import main; main()"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    processes.append(("iaas", iaas_proc))

    iaas_url = f"http://127.0.0.1:{iaas_port}"

    print(f"Waiting for IaaS service on port {iaas_port}...")
    if not wait_for_http(f"{iaas_url}/docs", timeout=20):
        stop_processes(processes)
        raise RuntimeError("IaaS service failed to start (health check timeout)")
    print(f"IaaS service ready on port {iaas_port}")

    return processes, iaas_url, ext_proc_port


def start_envoy(ext_proc_port, iaas_port, llm_port):
    """Start Envoy with generated config. Returns (process, envoy_url, tmpdir, admin_port) or None."""
    if not shutil.which("envoy"):
        return None

    print(f"Waiting for ext_proc on port {ext_proc_port}...")
    if not wait_for_port(ext_proc_port, timeout=15):
        print("Warning: ext_proc not ready, skipping Envoy")
        return None
    print(f"ext_proc ready on port {ext_proc_port}")

    envoy_port = find_free_port()
    tmpdir = tempfile.mkdtemp(prefix="its_envoy_")
    config_path = os.path.join(tmpdir, "envoy.yaml")
    config_text, admin_port = generate_envoy_config(envoy_port, ext_proc_port, iaas_port, llm_port)
    with open(config_path, "w") as f:
        f.write(config_text)

    envoy_proc = subprocess.Popen(
        ["envoy", "-c", config_path, "--log-level", "warn"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    print(f"Waiting for Envoy on port {envoy_port}...")
    if not wait_for_port(envoy_port, timeout=15):
        envoy_proc.terminate()
        envoy_proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Warning: Envoy failed to start")
        return None

    print(f"Envoy ready on port {envoy_port} (admin: {admin_port})")
    time.sleep(1)
    return envoy_proc, f"http://127.0.0.1:{envoy_port}", tmpdir, admin_port


def configure_iaas(iaas_url, llm_endpoint, model_name, api_key):
    """Configure IaaS service. Raises on failure."""
    status, body = http_post(f"{iaas_url}/configure", {
        "endpoint": llm_endpoint,
        "api_key": api_key,
        "model": model_name,
        "alg": "self-consistency",
        "regex_patterns": [r"\\boxed{([^}]+)}"],
    })
    if status != 200:
        raise RuntimeError(f"Failed to configure IaaS (status {status}): {body}")


def stop_processes(processes):
    """Terminate and wait for all managed processes."""
    for name, proc in reversed(processes):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"  {name} stopped")
