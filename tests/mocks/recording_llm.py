"""Controllable upstream LLM server for concurrency / regression tests.

Lives outside conftest.py: pytest loads conftest.py as plugin module ``conftest``
(distinct from ``tests.conftest``), so a class defined there would be duplicated.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler


class RecordingLLMHandler(BaseHTTPRequestHandler):
    """Upstream LLM stand-in that records requests and can hold them."""

    received_bodies: list[dict] = []  # noqa: RUF012 - shared across per-request instances
    _hold = threading.Event()
    _hold.set()
    _lock = threading.Lock()

    @classmethod
    def reset(cls):
        with cls._lock:
            cls.received_bodies.clear()
        cls._hold.set()

    @classmethod
    def hold(cls):
        cls._hold.clear()

    @classmethod
    def release(cls):
        cls._hold.set()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with type(self)._lock:
            type(self).received_bodies.append(body)

        type(self)._hold.wait(timeout=10)

        payload = json.dumps(
            {
                "model": body.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"answer from {body.get('model', 'unknown')}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }
        ).encode()

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
