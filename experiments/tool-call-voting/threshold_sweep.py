"""Threshold sweep: accuracy vs coverage at varying confidence thresholds.

For each threshold, tags cases as "accepted" (high-confidence) or "rejected"
(forced), then reports:
  - Coverage: % of cases accepted
  - Accuracy: arg accuracy on accepted cases only
  - Overall: accuracy counting rejected as wrong (conservative)

Produces the accuracy-vs-coverage curve the research doc calls for.
Runs on existing detail JSON files — no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_results import check_tool_call_against_gt, load_ground_truth
from roll_up_scorer import score_tool_calls

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]


def sweep_detail_file(
    detail_path: Path,
    bfcl_file: str,
    equivalence: bool = False,
) -> None:
    with open(detail_path) as f:
        results = json.load(f)

    gt = load_ground_truth(bfcl_file)

    # Pre-compute: re-score each task's raw tool calls at each threshold
    tasks = []
    for r in results:
        fa = r.get("field_aware")
        if not fa or not fa.get("raw_tool_calls"):
            continue
        tcs = fa["raw_tool_calls"]
        task_id = r["task_id"]
        gt_entry = gt.get(task_id)
        if not gt_entry:
            continue

        # Score the voted result's accuracy (same for all thresholds — the
        # voted answer doesn't change, only the confidence tag does)
        scored = score_tool_calls(tcs, threshold=0.5, equivalence=equivalence)
        if scored is None:
            continue
        gt_result = check_tool_call_against_gt(
            scored.tool_name, scored.merged_args, gt_entry, fuzzy=True
        )

        # Also get single-shot accuracy
        first_tc = tcs[0]
        first_name = first_tc.get("function", {}).get("name", "")
        first_args_raw = first_tc.get("function", {}).get("arguments", "{}")
        if isinstance(first_args_raw, str):
            try:
                first_args = json.loads(first_args_raw)
            except (json.JSONDecodeError, TypeError):
                first_args = {}
        else:
            first_args = first_args_raw if isinstance(first_args_raw, dict) else {}
        ss_result = check_tool_call_against_gt(first_name, first_args, gt_entry, fuzzy=True)

        tasks.append({
            "task_id": task_id,
            "tcs": tcs,
            "voted_args_correct": gt_result["args_correct"],
            "ss_args_correct": ss_result["args_correct"],
        })

    ss_accuracy = sum(1 for t in tasks if t["ss_args_correct"]) / len(tasks) * 100

    equiv_label = " +equiv" if equivalence else ""
    print(f"\n{'='*70}")
    print(f"THRESHOLD SWEEP: {detail_path.name}{equiv_label}")
    print(f"Tasks: {len(tasks)}, Single-shot accuracy: {ss_accuracy:.1f}%")
    print(f"{'='*70}")
    print(f"{'Threshold':>10} {'Coverage':>10} {'Accepted':>10} {'AcceptAcc':>10} "
          f"{'Lift':>10} {'Rejected':>10} {'RejAcc':>10}")
    print("-" * 70)

    for thresh in THRESHOLDS:
        accepted = []
        rejected = []
        for t in tasks:
            result = score_tool_calls(t["tcs"], threshold=thresh, equivalence=equivalence)
            if result is not None and result.confidence == "high_confidence":
                accepted.append(t)
            else:
                rejected.append(t)

        n_acc = len(accepted)
        n_rej = len(rejected)
        coverage = n_acc / len(tasks) * 100

        acc_correct = sum(1 for t in accepted if t["voted_args_correct"])
        acc_accuracy = acc_correct / n_acc * 100 if n_acc else 0

        rej_correct = sum(1 for t in rejected if t["voted_args_correct"])
        rej_accuracy = rej_correct / n_rej * 100 if n_rej else 0

        lift = acc_accuracy - ss_accuracy

        print(f"{thresh:>10.2f} {coverage:>9.1f}% {n_acc:>10} {acc_accuracy:>9.1f}% "
              f"{lift:>+9.1f}pp {n_rej:>10} {rej_accuracy:>9.1f}%")

    # Check pre-registered criterion: >=3pp lift at >=50% coverage
    print(f"\nPre-registered criterion: lift >= 3pp AND coverage >= 50%")
    met = False
    for thresh in THRESHOLDS:
        accepted = []
        for t in tasks:
            result = score_tool_calls(t["tcs"], threshold=thresh, equivalence=equivalence)
            if result is not None and result.confidence == "high_confidence":
                accepted.append(t)

        coverage = len(accepted) / len(tasks) * 100
        if not accepted:
            continue
        acc_accuracy = sum(1 for t in accepted if t["voted_args_correct"]) / len(accepted) * 100
        lift = acc_accuracy - ss_accuracy

        if lift >= 3.0 and coverage >= 50.0:
            print(f"  MET at threshold={thresh:.2f}: lift={lift:+.1f}pp, coverage={coverage:.1f}%")
            met = True

    if not met:
        print("  NOT MET at any threshold")


def main() -> None:
    bfcl_map = {
        "BFCL_v4_simple_python": "BFCL_v4_simple_python.json",
        "BFCL_v4_multiple": "BFCL_v4_multiple.json",
        "BFCL_v4_parallel": "BFCL_v4_parallel.json",
        "BFCL_v4_parallel_multiple": "BFCL_v4_parallel_multiple.json",
    }

    detail_files = sorted(RESULTS_DIR.glob("*_detail.json"))
    for detail_path in detail_files:
        stem = detail_path.stem
        bfcl_file = None
        for prefix, filename in bfcl_map.items():
            if stem.startswith(prefix):
                bfcl_file = filename
                break
        if not bfcl_file:
            continue

        # Run without and with equivalence
        sweep_detail_file(detail_path, bfcl_file, equivalence=False)
        sweep_detail_file(detail_path, bfcl_file, equivalence=True)


if __name__ == "__main__":
    main()
