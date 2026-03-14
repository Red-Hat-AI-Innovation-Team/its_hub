"""Fixtures for E2E tests with real API calls."""

import os

import pytest
from dotenv import load_dotenv

from its_hub import OpenAICompatibleLanguageModel


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-"):
        pytest.skip("OPENAI_API_KEY not set - skipping E2E tests")
    return api_key


@pytest.fixture(scope="session")
def openai_lm(openai_api_key):
    """Create OpenAI-compatible LM for testing."""
    return OpenAICompatibleLanguageModel(
        endpoint=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        api_key=openai_api_key,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        max_concurrent_requests=5,
    )
