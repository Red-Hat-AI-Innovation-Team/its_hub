"""Shared fixtures for tool-call voting scorer tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def make_tool_call():
    """Factory fixture for creating tool call dicts."""

    def _make(name: str, args: dict | str | None = None) -> dict:
        if args is None:
            args = {}
        if isinstance(args, dict):
            import json

            args = json.dumps(args)
        return {"function": {"name": name, "arguments": args}}

    return _make
