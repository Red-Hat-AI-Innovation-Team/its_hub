"""Shared test configuration and fixtures."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def find_free_port() -> int:
    """Find a free port to use for test servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DummyVLLMHandler(BaseHTTPRequestHandler):
    """A dummy HTTP handler that mimics a vLLM server."""

    def do_POST(self):
        """Handle POST requests to the /v1/chat/completions endpoint."""
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            # Simulate some processing time
            time.sleep(0.01)

            # Extract the user message
            messages = request_data.get("messages", [])
            user_content = messages[-1]["content"] if messages else "unknown"

            # Check for error triggers
            if "error" in user_content.lower():
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_response = {
                    "error": {
                        "message": "Simulated vLLM error",
                        "type": "server_error",
                        "code": 500,
                    }
                }
                self.wfile.write(json.dumps(error_response).encode("utf-8"))
                return

            # Create a response that includes the request content for testing
            response_content = f"vLLM response to: {user_content}"

            # Check if we should include stop tokens
            stop = request_data.get("stop")
            include_stop = request_data.get("include_stop_str_in_output", False)

            if stop and include_stop:
                response_content += stop

            # Create vLLM-like response
            response = {
                "id": "vllm-test-id",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request_data.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Suppress log messages to keep test output clean."""
        pass


# Pytest fixtures


@pytest.fixture(scope="session")
def vllm_server():
    """Start a vLLM mock server for the test session."""
    port = find_free_port()
    server = HTTPServer(("localhost", port), DummyVLLMHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Give the server a moment to start
    time.sleep(0.1)

    yield f"http://localhost:{port}"

    server.shutdown()
    server_thread.join()
