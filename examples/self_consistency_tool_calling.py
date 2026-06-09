"""
Self-Consistency with tool calling — the documentation-website Example 1, corrected for the
current (post api/core refactor) version of its_hub.

What changed vs. the website snippet:
  - `from its_hub.lms   import OpenAICompatibleLanguageModel`  ->  `from its_hub import ...`
  - `from its_hub.algorithms import SelfConsistency`          ->  `from its_hub import ...`  (old path is deprecated)
  - `from its_hub.types import ChatMessage, ChatMessages`     ->  `from its_hub.api import ...`
The usage (tools schema, ChatMessages, SelfConsistency(tool_vote=...), .infer(...)) is unchanged.

Run (in the project's conda env):
    export OPENAI_API_KEY=sk-...                 # or put it in a .env file at the repo root
    conda run -n epf python examples/self_consistency_tool_calling.py

Point it elsewhere with env vars (any OpenAI-compatible endpoint, incl. a local vLLM server):
    export ITS_ENDPOINT=http://localhost:8000/v1
    export ITS_MODEL=Qwen/Qwen2.5-7B-Instruct
    export ITS_API_KEY=NO_API_KEY
"""

import asyncio
import os

# Optional: load a .env file at the repo root if python-dotenv is installed (it is, in [dev]).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── corrected imports for this version ───────────────────────────────────────────────────
from its_hub import OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

ENDPOINT = os.getenv("ITS_ENDPOINT", "https://api.openai.com/v1")
MODEL = os.getenv("ITS_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("ITS_API_KEY") or os.getenv("OPENAI_API_KEY") or "MISSING_API_KEY"

# Initialize language model
lm = OpenAICompatibleLanguageModel(endpoint=ENDPOINT, api_key=API_KEY, model_name=MODEL)

# Define tools (OpenAI format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

# Create messages
messages = ChatMessages(
    [
        ChatMessage(
            role="system",
            content="You are a precise calculator. Always use the calculator tool for arithmetic.",
        ),
        ChatMessage(role="user", content="What is 847 * 293 + 156?"),
    ]
)

# Use hierarchical tool voting: vote on the tool NAME first, then on its ARGUMENTS.
sc = SelfConsistency(tool_vote="tool_hierarchical")


def main() -> None:
    # return_response_only=False so we can also see the vote tally (response_counts).
    result = sc.infer(lm, messages, budget=5, tools=tools, tool_choice="auto", return_response_only=False)
    print("######## the_one (selected tool call) ########")
    print(result.the_one)
    print("\n######## response_counts (the vote tally) ########")
    print(result.response_counts)
    asyncio.run(lm.close())  # clean up HTTP sessions


if __name__ == "__main__":
    main()
