"""
Plain Self-Consistency (answer voting) against a LOCAL OpenAI-compatible server.

This is the reliable way to *see self-consistency work* with any text model — including
Qwen2.5-Omni, whose chat template does NOT support tool calling (so the tool-calling variant
isn't applicable here; this votes on the extracted \\boxed{} answer instead).

Defaults target a local vLLM server on :8000; override with env vars:
    ITS_ENDPOINT (default http://localhost:8000/v1)
    ITS_MODEL    (default Qwen2.5-Omni-3B)
    ITS_API_KEY  (default NO_API_KEY)
    ITS_BUDGET   (default 5)

Run:
    conda run -n epf python examples/self_consistency_local.py
"""

import asyncio
import os
import re

from its_hub import OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.core.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

ENDPOINT = os.getenv("ITS_ENDPOINT", "http://localhost:8000/v1")
MODEL = os.getenv("ITS_MODEL", "Qwen2.5-Omni-3B")
API_KEY = os.getenv("ITS_API_KEY", "NO_API_KEY")
BUDGET = int(os.getenv("ITS_BUDGET", "5"))

# A problem with a clean, checkable final answer.
PROBLEM = (
    "What is the value of 847 * 293 + 156? "
    "Reason step by step and put your final answer in \\boxed{}."
)


def extract_boxed(s: str) -> str:
    """Project a full response down to its last \\boxed{...} answer (the thing we vote on)."""
    matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", s)
    return matches[-1].strip() if matches else ""


def main() -> None:
    lm = OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT,
        api_key=API_KEY,
        model_name=MODEL,
        system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
        max_tokens=1024,
        temperature=0.8,  # diversity across the `budget` samples is what makes voting meaningful
    )
    sc = SelfConsistency(extract_boxed)

    print(f"endpoint={ENDPOINT}  model={MODEL}  budget={BUDGET}")
    print(f"problem: {PROBLEM}\n(correct answer: {847 * 293 + 156})\n")

    result = sc.infer(lm, PROBLEM, budget=BUDGET, return_response_only=False)

    print("######## response_counts (vote tally over extracted answers) ########")
    print(dict(result.response_counts))
    print("\n######## the_one (winning full response) ########")
    print(result.the_one["content"])
    print("\n######## extracted winning answer ########")
    print(extract_boxed(result.the_one["content"]))

    asyncio.run(lm.close())


if __name__ == "__main__":
    main()
