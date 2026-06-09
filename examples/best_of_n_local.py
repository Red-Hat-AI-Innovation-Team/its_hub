"""
Best-of-N with an LLM judge — the documentation-website Example 2, corrected for the current
version of its_hub, pointed at a LOCAL OpenAI-compatible server (no API key needed).

KEY POINT: its_hub does NOT bundle a judge model or any credentials. The "judge" is just
another LLM call through an AbstractLanguageModel YOU provide. By default LLMJudge reuses the
SAME `lm` you're scaling — so here Qwen grades Qwen, entirely locally.

What changed vs. the website snippet (pre-refactor → current):
  - from its_hub.lms import ...                                   -> from its_hub import ...
  - from its_hub.algorithms import BestOfN                        -> from its_hub import BestOfN  (old path deprecated)
  - from its_hub.integration.reward_hub import LLMJudgeRewardModel-> from its_hub import LLMJudge  (renamed + simplified)
  - LLMJudgeRewardModel(model=..., criterion=..., judge_type="groupwise", api_key=...)
        -> LLMJudge(lm)   # takes an LM instance; the criterion/judge_type/api_key params no longer exist

Run (server must be up on :8000 first):
    conda run -n epf python examples/best_of_n_local.py

Env overrides: ITS_ENDPOINT, ITS_MODEL, ITS_API_KEY, ITS_BUDGET (default 4).
Use a different/stronger judge by setting ITS_JUDGE_ENDPOINT / ITS_JUDGE_MODEL / ITS_JUDGE_API_KEY.
"""

import asyncio
import os

from its_hub import BestOfN, LLMJudge, OpenAICompatibleLanguageModel

ENDPOINT = os.getenv("ITS_ENDPOINT", "http://localhost:8000/v1")
MODEL = os.getenv("ITS_MODEL", "Qwen2.5-Omni-3B")
API_KEY = os.getenv("ITS_API_KEY", "NO_API_KEY")
BUDGET = int(os.getenv("ITS_BUDGET", "4"))

PROMPT = "Explain quantum entanglement in simple terms."


def main() -> None:
    # The model that GENERATES the N candidate answers.
    lm = OpenAICompatibleLanguageModel(
        endpoint=ENDPOINT, api_key=API_KEY, model_name=MODEL, max_tokens=512, temperature=0.8
    )

    # The model that JUDGES them. By default reuse `lm` (Qwen judging Qwen, no extra creds).
    # To use a different/stronger judge, point these env vars elsewhere.
    judge_lm = lm
    if os.getenv("ITS_JUDGE_ENDPOINT") or os.getenv("ITS_JUDGE_MODEL"):
        judge_lm = OpenAICompatibleLanguageModel(
            endpoint=os.getenv("ITS_JUDGE_ENDPOINT", ENDPOINT),
            api_key=os.getenv("ITS_JUDGE_API_KEY", API_KEY),
            model_name=os.getenv("ITS_JUDGE_MODEL", MODEL),
        )

    judge = LLMJudge(judge_lm)        # ORM: prompts the LLM for a 0-10 score + reasoning (JSON)
    alg = BestOfN(judge)

    print(f"endpoint={ENDPOINT}  model={MODEL}  budget={BUDGET}")
    print(f"prompt: {PROMPT}\n")

    # return_response_only=False so we can see every candidate's judge score.
    result = alg.infer(lm, PROMPT, budget=BUDGET, return_response_only=False)

    print("######## per-candidate judge scores (0-10) ########")
    for i, (resp, score) in enumerate(zip(result.responses, result.scores)):
        mark = "  <-- winner" if i == result.selected_index else ""
        preview = (resp.get("content") or "")[:80].replace("\n", " ")
        print(f"  [{i}] score={score:>5}  {preview!r}{mark}")

    print("\n######## the_one (winning answer) ########")
    print(result.the_one["content"])

    # If every score is identical (e.g. judge JSON failed -> fallback_score=5.0 for all),
    # Best-of-N just picks the first — a sign the judge couldn't differentiate. Try
    # response_format=None in LLMJudge(...) to rely on prompt-based JSON instead of strict schema.
    asyncio.run(lm.close())
    if judge_lm is not lm:
        asyncio.run(judge_lm.close())


if __name__ == "__main__":
    main()
