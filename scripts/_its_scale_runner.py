"""Run inference-time scaling on a prompt using config and env vars.

Called by its_scale.sh — not intended for direct use.

Environment variables:
    ITS_CONFIG      — path to config JSON
    ITS_MODEL_KEY   — key into config's models dict
    ITS_ALGORITHM   — algorithm override (empty = use config default)
    ITS_PROMPT      — the prompt to scale
    ITS_BUDGET      — budget override (empty = use config default)
    ITS_METADATA    — "true" to include full algorithm metadata
    OPENAI_API_KEY / ANTHROPIC_API_KEY — API key (read from env, not config)
"""

import json
import os
import sys

config = json.load(open(os.environ["ITS_CONFIG"]))
model_key = os.environ["ITS_MODEL_KEY"]
model_cfg = config.get("models", {}).get(model_key, {})
alg_cfg = config.get("algorithm_config", {})
prompt = os.environ["ITS_PROMPT"]
show_metadata = os.environ["ITS_METADATA"] == "true"

algorithm = os.environ.get("ITS_ALGORITHM") or config.get("algorithm", "self-consistency")
budget = int(os.environ.get("ITS_BUDGET") or config.get("budget", 8))

if not model_cfg.get("model"):
    print(f"ERROR: Model '{model_key}' not found in config. Run /its-setup to add it.", file=sys.stderr)
    sys.exit(1)

if algorithm in ("particle-filtering", "beam-search"):
    print(
        f"ERROR: {algorithm} requires process reward models and is experimental in v1. "
        "Use the Python API directly for advanced algorithms.",
        file=sys.stderr,
    )
    sys.exit(1)

from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel

api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

lm = OpenAICompatibleLanguageModel(
    endpoint=model_cfg["endpoint"],
    api_key=api_key,
    model_name=model_cfg["model"],
)

if algorithm == "self-consistency":
    from its_hub.core.algorithms.self_consistency import (
        SelfConsistency,
        create_regex_projection_function,
    )

    patterns = alg_cfg.get("regex_patterns")
    proj = create_regex_projection_function(patterns) if patterns else None
    alg = SelfConsistency(
        proj,
        tool_vote=alg_cfg.get("tool_vote"),
        exclude_args=alg_cfg.get("exclude_tool_args"),
    )
elif algorithm == "best-of-n":
    from its_hub.core.algorithms.bon import BestOfN
    from its_hub.core.reward_models.llm_judge import LLMJudge

    judge_cfg = {k: v for k, v in alg_cfg.items() if k.startswith("judge_")}
    judge_endpoint = judge_cfg.get("judge_endpoint", model_cfg["endpoint"])
    judge_model = judge_cfg.get("judge_model", model_cfg["model"])

    judge_lm = OpenAICompatibleLanguageModel(
        endpoint=judge_endpoint,
        api_key=api_key,
        model_name=judge_model,
    )
    judge = LLMJudge(lm=judge_lm)
    alg = BestOfN(judge)
else:
    print(json.dumps({"error": f"Unsupported algorithm: {algorithm}"}))
    sys.exit(1)

result = alg.infer(lm, prompt, budget=budget, return_response_only=not show_metadata)

if show_metadata and hasattr(result, "the_one"):
    output = {"selected": result.the_one, "type": type(result).__name__}
    print(json.dumps(output, default=str))
elif isinstance(result, dict):
    print(json.dumps(result))
else:
    print(result)
