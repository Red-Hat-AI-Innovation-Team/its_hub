"""Analyze tool-call voting checkpoint results.

Probe 1: Inspect forced cases — what do they look like?
Probe 2: Score against BFCL ground truth — does voting improve accuracy?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_ground_truth(bfcl_file: str) -> dict[str, list[dict]]:
    """Load BFCL ground truth keyed by task ID."""
    path = DATA_DIR / "answers" / bfcl_file
    gt = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            gt[entry["id"]] = entry["ground_truth"]
    return gt


def check_tool_call_against_gt(
    tool_name: str,
    args: dict,
    ground_truth: list[dict],
    fuzzy: bool = False,
) -> dict:
    """Check a single tool call against BFCL ground truth.

    Ground truth format: [{func_name: {arg_name: [acceptable_values]}}]

    Returns dict with name_correct, args_correct, per-field details.
    """
    # Try normalized name matching (accounts for sanitizer dot→underscore)
    for gt_call in ground_truth:
        for gt_name, gt_args in gt_call.items():
            import re as _re
            tn = _re.sub(r"[-_.\s]+", "_", tool_name).lower()
            gn = _re.sub(r"[-_.\s]+", "_", gt_name).lower()
            name_match = (tool_name == gt_name) or (tn == gn)
            if not name_match:
                continue

            field_results = {}
            all_correct = True
            for arg_name, acceptable in gt_args.items():
                actual = args.get(arg_name)
                correct = any(
                    _values_match(actual, acc, fuzzy=fuzzy) for acc in acceptable
                )
                field_results[arg_name] = {
                    "correct": correct,
                    "actual": actual,
                    "acceptable": acceptable,
                }
                if not correct:
                    all_correct = False

            return {
                "name_correct": True,
                "args_correct": all_correct,
                "field_results": field_results,
            }

    return {"name_correct": False, "args_correct": False, "field_results": {}}


def _normalize_string(s: str) -> str:
    """Normalize a string for fuzzy comparison."""
    import re
    s = s.strip().lower()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Strip trailing punctuation
    s = s.rstrip(".,;:!?")
    return s


def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-split tokens."""
    tokens_a = set(_normalize_string(a).split())
    tokens_b = set(_normalize_string(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _values_match(actual, expected, fuzzy: bool = False) -> bool:
    """Flexible value comparison for BFCL ground truth."""
    if actual is None and expected == "":
        return True
    if actual == expected:
        return True
    # Numeric comparison (int/float equivalence)
    try:
        if float(actual) == float(expected):
            return True
    except (TypeError, ValueError):
        pass
    # String comparison (case-insensitive, strip whitespace)
    if isinstance(actual, str) and isinstance(expected, str):
        if _normalize_string(actual) == _normalize_string(expected):
            return True
        if fuzzy:
            # Containment: one is a substring of the other after normalization
            na, ne = _normalize_string(actual), _normalize_string(expected)
            if na in ne or ne in na:
                return True
            # High token overlap (>=0.8 Jaccard)
            if _token_overlap(actual, expected) >= 0.8:
                return True
    return False


def analyze_detail_file(detail_path: Path, bfcl_file: str) -> None:
    """Analyze a detailed results JSON file."""
    with open(detail_path) as f:
        results = json.load(f)

    gt = load_ground_truth(bfcl_file)

    forced_cases = []
    high_conf_cases = []

    for r in results:
        fa = r.get("field_aware")
        if fa is None:
            continue

        task_id = r["task_id"]
        confidence = fa["confidence"]

        # Score against ground truth
        gt_entry = gt.get(task_id)
        if gt_entry:
            score = check_tool_call_against_gt(
                fa["tool_name"], fa["merged_args"], gt_entry
            )
        else:
            score = None

        entry = {
            "task_id": task_id,
            "confidence": confidence,
            "tool_name": fa["tool_name"],
            "merged_args": fa["merged_args"],
            "field_votes": fa["field_votes"],
            "tool_name_is_tie": fa.get("tool_name_is_tie", False),
            "raw_tool_calls": fa.get("raw_tool_calls", []),
            "gt_score": score,
        }

        if confidence == "forced":
            forced_cases.append(entry)
        else:
            high_conf_cases.append(entry)

    # --- Probe 1: Inspect forced cases ---
    print(f"\n{'='*60}")
    print(f"FILE: {detail_path.name}")
    print(f"{'='*60}")
    print(f"Total: {len(results)}, High-conf: {len(high_conf_cases)}, Forced: {len(forced_cases)}")

    if forced_cases:
        print(f"\n--- FORCED CASES ({len(forced_cases)}) ---")
        for case in forced_cases:
            print(f"\n  Task: {case['task_id']}")
            print(f"  Tool: {case['tool_name']} (tie={case['tool_name_is_tie']})")
            print(f"  Merged args: {json.dumps(case['merged_args'], default=str)[:200]}")
            for fv in case["field_votes"]:
                if fv["agreement"] < 0.75:
                    print(f"  LOW AGREEMENT: {fv['field_name']} = {fv['winning_value']} "
                          f"({fv['vote_count']}/{fv['total_votes']} = {fv['agreement']:.0%})")
            # Show what the samples actually produced
            if case["raw_tool_calls"]:
                print(f"  Raw samples ({len(case['raw_tool_calls'])}):")
                for i, tc in enumerate(case["raw_tool_calls"]):
                    tc_name = tc.get("function", {}).get("name", "?")
                    tc_args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(tc_args, str):
                        tc_args_str = tc_args[:150]
                    else:
                        tc_args_str = json.dumps(tc_args, default=str)[:150]
                    print(f"    [{i}] {tc_name}({tc_args_str})")
            if case["gt_score"]:
                s = case["gt_score"]
                print(f"  GT: name={'OK' if s['name_correct'] else 'WRONG'}, "
                      f"args={'OK' if s['args_correct'] else 'WRONG'}")
                for fname, fr in s["field_results"].items():
                    if not fr["correct"]:
                        print(f"    WRONG: {fname} = {fr['actual']} "
                              f"(expected one of {fr['acceptable']})")

    # --- Probe 2: Accuracy against ground truth ---
    all_cases = high_conf_cases + forced_cases

    for mode_label, fuzzy in [("EXACT MATCH", False), ("FUZZY MATCH", True)]:
        # Re-score everything under this mode
        for c in all_cases:
            gt_entry = gt.get(c["task_id"])
            if gt_entry:
                c[f"gt_{mode_label}"] = check_tool_call_against_gt(
                    c["tool_name"], c["merged_args"], gt_entry, fuzzy=fuzzy
                )
            else:
                c[f"gt_{mode_label}"] = None

        scored = [c for c in all_cases if c.get(f"gt_{mode_label}") is not None]
        if not scored:
            print(f"\nNo ground truth available for accuracy scoring.")
            continue

        hc_scored = [c for c in scored if c["confidence"] == "high_confidence"]
        fc_scored = [c for c in scored if c["confidence"] == "forced"]

        voted_name = sum(1 for c in scored if c[f"gt_{mode_label}"]["name_correct"])
        voted_args = sum(1 for c in scored if c[f"gt_{mode_label}"]["args_correct"])
        hc_name = sum(1 for c in hc_scored if c[f"gt_{mode_label}"]["name_correct"])
        hc_args = sum(1 for c in hc_scored if c[f"gt_{mode_label}"]["args_correct"])

        # Single-shot: score first sample only
        ss_name = 0
        ss_args = 0
        for c in scored:
            if c["raw_tool_calls"]:
                first_tc = c["raw_tool_calls"][0]
                first_name = first_tc.get("function", {}).get("name", "")
                first_args_raw = first_tc.get("function", {}).get("arguments", "{}")
                if isinstance(first_args_raw, str):
                    try:
                        first_args = json.loads(first_args_raw)
                    except (json.JSONDecodeError, TypeError):
                        first_args = {}
                else:
                    first_args = first_args_raw if isinstance(first_args_raw, dict) else {}

                gt_entry = gt.get(c["task_id"])
                if gt_entry:
                    ss_score = check_tool_call_against_gt(
                        first_name, first_args, gt_entry, fuzzy=fuzzy
                    )
                    if ss_score["name_correct"]:
                        ss_name += 1
                    if ss_score["args_correct"]:
                        ss_args += 1

        # Best-of-5: any single sample correct?
        bo5_name = 0
        bo5_args = 0
        for c in scored:
            gt_entry = gt.get(c["task_id"])
            if not gt_entry or not c["raw_tool_calls"]:
                continue
            any_name = False
            any_args = False
            for tc in c["raw_tool_calls"]:
                tc_name = tc.get("function", {}).get("name", "")
                tc_args_raw = tc.get("function", {}).get("arguments", "{}")
                if isinstance(tc_args_raw, str):
                    try:
                        tc_args = json.loads(tc_args_raw)
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {}
                else:
                    tc_args = tc_args_raw if isinstance(tc_args_raw, dict) else {}
                s = check_tool_call_against_gt(tc_name, tc_args, gt_entry, fuzzy=fuzzy)
                if s["name_correct"]:
                    any_name = True
                if s["args_correct"]:
                    any_args = True
            if any_name:
                bo5_name += 1
            if any_args:
                bo5_args += 1

        print(f"\n--- ACCURACY vs GROUND TRUTH ({mode_label}) ---")
        n = len(scored)
        print(f"  Single-shot (1st sample): name={ss_name}/{n} ({ss_name/n:.1%}), "
              f"args={ss_args}/{n} ({ss_args/n:.1%})")
        print(f"  Voted (N=5):              name={voted_name}/{n} ({voted_name/n:.1%}), "
              f"args={voted_args}/{n} ({voted_args/n:.1%})")
        print(f"  Best-of-5 (oracle):       name={bo5_name}/{n} ({bo5_name/n:.1%}), "
              f"args={bo5_args}/{n} ({bo5_args/n:.1%})")
        if hc_scored:
            print(f"  High-conf voted:          name={hc_name}/{len(hc_scored)} ({hc_name/len(hc_scored):.1%}), "
                  f"args={hc_args}/{len(hc_scored)} ({hc_args/len(hc_scored):.1%})")

        delta_name = (voted_name - ss_name) / n * 100
        delta_args = (voted_args - ss_args) / n * 100
        headroom = (bo5_args - voted_args) / n * 100
        print(f"\n  Delta voted vs single:   name={delta_name:+.1f}pp, args={delta_args:+.1f}pp")
        print(f"  Headroom (oracle - voted): args={headroom:+.1f}pp")


def main() -> None:
    detail_files = sorted(RESULTS_DIR.glob("*_detail.json"))
    if not detail_files:
        print("No detail JSON files found. Run the harness first.", file=sys.stderr)
        sys.exit(1)

    bfcl_map = {
        "BFCL_v4_simple_python": "BFCL_v4_simple_python.json",
        "BFCL_v4_multiple": "BFCL_v4_multiple.json",
        "BFCL_v4_parallel": "BFCL_v4_parallel.json",
        "BFCL_v4_parallel_multiple": "BFCL_v4_parallel_multiple.json",
    }

    for detail_path in detail_files:
        stem = detail_path.stem
        bfcl_file = None
        for prefix, filename in bfcl_map.items():
            if stem.startswith(prefix):
                bfcl_file = filename
                break
        if bfcl_file:
            analyze_detail_file(detail_path, bfcl_file)
        else:
            print(f"Skipping {detail_path.name}: no matching BFCL file")


if __name__ == "__main__":
    main()
