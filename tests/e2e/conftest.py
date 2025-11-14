"""Pytest configuration for E2E tests."""

import os
from pathlib import Path

# Load .env file IMMEDIATELY before pytest even starts
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    load_dotenv(env_file, override=True)
except ImportError:
    pass
