"""Re-score existing BFCL detail JSON files using the Schema Validation ORM.

For each task: take the N raw tool calls, score each with the ORM, select
the highest-scored one, compare against ground truth.

Reports: single-shot accuracy vs schema-validated-BoN accuracy
         vs majority-voted accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_results import check_tool_call_against_gt, load_ground_truth
from sampling_harness import build_tool_call_prompt, load_bfcl_tasks
from schema_validation_orm import SchemaValidationORM

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

BFCL_MAP = {
    "BFCL_v4_simple_python": "BFCL_v4_simple_python.json",
    "BFCL_v4_multiple": "BFCL_v4_multiple.json",
    "BFCL_v4_parallel": "BFCL_v4_parallel.json",
    "BFCL_v4_parallel_multiple": "BFCL_v4_parallel_multiple.json",
}


def _parse_tc_args(tc: dict) -> dict:
    args_raw = tc.get("function", {}).get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            return json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return args_raw if isinstance(args_raw, dict) else {}


def _build_task_schemas(bfcl_file: str) -> dict[str, list[dict]]:
    """Load BFCL tasks and build per-task tool definitions."""
    tasks = load_bfcl_tasks(bfcl_file)
    schemas: dict[str, list[dict]] = {}
    for task in tasks:
        task_id = task.get("id", "")
        _, tools = build_tool_call_prompt(task)
        schemas[task_id] = tools
    return schemas


def rescore_detail_file(detail_path: Path, bfcl_file: str, fuzzy: bool = False) -> None:
    """Re-score a detail file using schema validation BoN."""
    with open(detail_path) as f:
        results = json.load(f)

    gt = load_ground_truth(bfcl_file)
    task_schemas = _build_task_schemas(bfcl_file)

    ss_correct = 0
    voted_correct = 0
    bon_correct = 0
    oracle_correct = 0
    scorable = 0

    for entry in results:
        task_id = entry["task_id"]
        fa = entry.get("field_aware")
        if fa is None:
            continue

        gt_entry = gt.get(task_id)
        if gt_entry is None:
            continue

        raw_tcs = fa.get("raw_tool_calls", [])
        if not raw_tcs:
            continue

        scorable += 1
        tools = task_schemas.get(task_id, [])
        orm = SchemaValidationORM(tools)

        # Single-shot: first sample
        first_tc = raw_tcs[0]
        first_name = first_tc.get("function", {}).get("name", "")
        first_args = _parse_tc_args(first_tc)
        ss_result = check_tool_call_against_gt(first_name, first_args, gt_entry, fuzzy=fuzzy)
        if ss_result["args_correct"]:
            ss_correct += 1

        # Majority-voted (field-aware scorer result already in the file)
        voted_result = check_tool_call_against_gt(
            fa["tool_name"], fa["merged_args"], gt_entry, fuzzy=fuzzy
        )
        if voted_result["args_correct"]:
            voted_correct += 1

        # Schema-validated BoN: score each raw TC, pick the best
        best_score = -1.0
        best_tc = raw_tcs[0]
        for tc in raw_tcs:
            conv = [
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": None, "tool_calls": [tc]},
            ]
            s = orm.score(conv)
            if s > best_score:
                best_score = s
                best_tc = tc

        bon_name = best_tc.get("function", {}).get("name", "")
        bon_args = _parse_tc_args(best_tc)
        bon_result = check_tool_call_against_gt(bon_name, bon_args, gt_entry, fuzzy=fuzzy)
        if bon_result["args_correct"]:
            bon_correct += 1

        # Oracle: any sample correct?
        for tc in raw_tcs:
            tc_name = tc.get("function", {}).get("name", "")
            tc_args = _parse_tc_args(tc)
            if check_tool_call_against_gt(tc_name, tc_args, gt_entry, fuzzy=fuzzy)["args_correct"]:
                oracle_correct += 1
                break

    # Report
    print(f"\n{'='*60}")
    print(f"FILE: {detail_path.name}")
    print(f"{'='*60}")
    print(f"Scorable tasks: {scorable}")

    if scorable == 0:
        print("  No scorable tasks found.")
        return

    mode = "FUZZY" if fuzzy else "EXACT"
    print(f"\n--- ACCURACY ({mode} MATCH) ---")
    print(f"  Single-shot (1st sample):  {ss_correct}/{scorable} ({ss_correct/scorable:.1%})")
    print(f"  Majority-voted (N=5):      {voted_correct}/{scorable} ({voted_correct/scorable:.1%})")
    print(f"  Schema-validated BoN:      {bon_correct}/{scorable} ({bon_correct/scorable:.1%})")
    print(f"  Oracle (any-of-N correct): {oracle_correct}/{scorable} ({oracle_correct/scorable:.1%})")

    delta_bon = (bon_correct - ss_correct) / scorable * 100
    delta_voted = (voted_correct - ss_correct) / scorable * 100
    headroom = (oracle_correct - bon_correct) / scorable * 100
    print(f"\n  Delta schema-BoN vs single:  {delta_bon:+.1f}pp")
    print(f"  Delta voted vs single:       {delta_voted:+.1f}pp")
    print(f"  Headroom (oracle - schema):  {headroom:+.1f}pp")


def main() -> None:
    detail_files = sorted(RESULTS_DIR.glob("*_detail.json"))
    if not detail_files:
        print("No detail JSON files found in results/.", file=sys.stderr)
        sys.exit(1)

    fuzzy = "--fuzzy" in sys.argv

    for detail_path in detail_files:
        stem = detail_path.stem
        bfcl_file = None
        for prefix, filename in BFCL_MAP.items():
            if stem.startswith(prefix):
                bfcl_file = filename
                break
        if bfcl_file:
            rescore_detail_file(detail_path, bfcl_file, fuzzy=fuzzy)
        else:
            print(f"Skipping {detail_path.name}: no matching BFCL file")


if __name__ == "__main__":
    main()
