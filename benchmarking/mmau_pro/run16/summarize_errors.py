"""Per-cell rows/errors summary of a diversity_probe JSONL (dedupe keep-last).

Exit code 1 iff any errors remain in the filtered set — run_all.sh uses this
to decide whether to fire the one automatic resume-retry after each stage.
"""

import argparse
import json
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl")
    ap.add_argument("--min-budget", type=int, default=0,
                    help="only report cells with budget >= this (e.g. 64 for the new stages)")
    args = ap.parse_args()

    latest = {}
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            latest[(r["unique_id"], int(r["method"]), r["signal"], int(r["budget"]))] = r

    cells = defaultdict(lambda: [0, 0])
    for (_uid, m, sig, b), r in latest.items():
        if b < args.min_budget:
            continue
        cells[(m, sig, b)][0] += 1
        if r.get("error"):
            cells[(m, sig, b)][1] += 1

    total_err = 0
    print(f"{'cell':>32} {'rows':>7} {'errors':>7}")
    for (m, sig, b) in sorted(cells):
        n, e = cells[(m, sig, b)]
        total_err += e
        print(f"P{m} {sig:>14} b{b:<4} {n:>10} {e:>7}")
    print(f"total errors (budget >= {args.min_budget}): {total_err}")
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
