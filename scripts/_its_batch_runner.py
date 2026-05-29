"""Run inference-time scaling on a batch of prompts from a file.

Called by its_batch_scale.sh — not intended for direct use.
Loads config once and processes all prompts in a single process.

Environment variables:
    ITS_CONFIG      — path to config JSON
    ITS_MODEL_KEY   — key into config's models dict
    ITS_ALGORITHM   — algorithm override (empty = use config default)
    ITS_BUDGET      — budget override (empty = use config default)
    ITS_INPUT       — input file path (JSONL, CSV, or TXT)
    ITS_OUTPUT      — output file path
    OPENAI_API_KEY / ANTHROPIC_API_KEY — API key (read from env, not config)
"""

import csv
import json
import os
import sys

config = json.load(open(os.environ["ITS_CONFIG"]))
model_key = os.environ["ITS_MODEL_KEY"]
model_cfg = config.get("models", {}).get(model_key, {})
alg_cfg = config.get("algorithm_config", {})
input_file = os.environ["ITS_INPUT"]
output_file = os.environ["ITS_OUTPUT"]

algorithm = os.environ.get("ITS_ALGORITHM") or config.get("algorithm", "self-consistency")
budget = int(os.environ.get("ITS_BUDGET") or config.get("budget", 8))

if not model_cfg.get("model"):
    print(f"ERROR: Model '{model_key}' not found in config.", file=sys.stderr)
    sys.exit(1)

if algorithm in ("particle-filtering", "beam-search"):
    print(
        f"ERROR: {algorithm} requires process reward models and is experimental in v1.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Read prompts ---

ext = os.path.splitext(input_file)[1].lower()

prompts: list[str] = []
if ext == ".jsonl":
    with open(input_file) as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or json.dumps(obj.get("messages", "")))
elif ext == ".csv":
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
elif ext == ".txt":
    with open(input_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
else:
    print(f"ERROR: Unsupported file format: {ext}", file=sys.stderr)
    sys.exit(1)

print(f"Found {len(prompts)} prompts in {input_file}")

# --- Set up model and algorithm once ---

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
    print(f"ERROR: Unsupported algorithm: {algorithm}", file=sys.stderr)
    sys.exit(1)

# --- Process prompts ---

os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

succeeded = 0
failed = 0
failures: list[dict] = []

with open(output_file, "w") as out:
    for i, prompt in enumerate(prompts, 1):
        try:
            result = alg.infer(lm, prompt, budget=budget, return_response_only=False)
            if hasattr(result, "the_one"):
                selected = result.the_one
            elif isinstance(result, dict):
                selected = result
            else:
                selected = str(result)
            entry = {
                "prompt": prompt,
                "selected_response": selected,
                "algorithm": algorithm,
                "budget": budget,
            }
            out.write(json.dumps(entry, default=str) + "\n")
            succeeded += 1
            print(f"  [{i}/{len(prompts)}] OK")
        except Exception as e:
            entry = {
                "prompt": prompt,
                "error": str(e),
                "algorithm": algorithm,
                "budget": budget,
            }
            out.write(json.dumps(entry, default=str) + "\n")
            failed += 1
            failures.append({"line": i, "error": str(e)})
            print(f"  [{i}/{len(prompts)}] FAILED: {e}")

print(
    json.dumps(
        {
            "status": "complete",
            "total": len(prompts),
            "succeeded": succeeded,
            "failed": failed,
            "failures": failures,
            "output_file": output_file,
        }
    )
)
