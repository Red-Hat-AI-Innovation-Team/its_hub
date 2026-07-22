"""Audit BFCL v4 single-turn parameter schemas.

Pulls the function-calling benchmark data and buckets every argument field
into one of three categories by JSON schema type:

  - numeric:          type is "number" or "integer"
  - categorical-enum: type is "string" with an "enum" constraint
  - free-text:        type is "string" without an "enum" constraint

Outputs a count/percentage breakdown to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

BFCL_REPO_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    "main/berkeley-function-call-leaderboard/data/"
)

SINGLE_TURN_FILES_V3 = [
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
]

# BFCL v4 may use different filenames when released; override via
# BFCL_VERSION env var (e.g. "v4") to look for BFCL_v4_*.json instead.
BFCL_VERSION = os.environ.get("BFCL_VERSION", "v3")
SINGLE_TURN_FILES = [
    f.replace("v3", BFCL_VERSION) for f in SINGLE_TURN_FILES_V3
]


def classify_field(schema: dict) -> str:
    """Classify a single parameter field by its JSON schema definition."""
    field_type = schema.get("type", "")

    if isinstance(field_type, list):
        types = set(field_type)
        if types & {"number", "integer"}:
            return "numeric"
        if "string" in types and "enum" in schema:
            return "categorical-enum"
        if "string" in types:
            return "free-text"
        return "free-text"

    if field_type in ("number", "integer"):
        return "numeric"
    if field_type == "string" and "enum" in schema:
        return "categorical-enum"
    if field_type == "string":
        return "free-text"
    if field_type == "boolean":
        return "categorical-enum"
    if field_type == "array":
        items = schema.get("items", {})
        return classify_field(items)

    return "free-text"


def audit_function_schemas(functions: list[dict]) -> Counter:
    """Walk function parameter schemas and classify each field."""
    counts: Counter = Counter()

    for func in functions:
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        for _name, field_schema in properties.items():
            category = classify_field(field_schema)
            counts[category] += 1

    return counts


def load_bfcl_from_file(path: Path) -> list[dict]:
    """Load BFCL data from a local JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_functions_from_entries(entries: list[dict]) -> list[dict]:
    """Extract function schemas from BFCL entries."""
    functions = []
    for entry in entries:
        for func_group in entry.get("function", []):
            if isinstance(func_group, dict):
                functions.append(func_group)
            elif isinstance(func_group, list):
                functions.extend(func_group)
    return functions


def main() -> None:
    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        print(
            f"Data directory not found: {data_dir}\n"
            f"Download BFCL data files to {data_dir}/ first.\n"
            f"Expected files: {', '.join(SINGLE_TURN_FILES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_counts: Counter = Counter()

    for filename in SINGLE_TURN_FILES:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)", file=sys.stderr)
            continue

        entries = load_bfcl_from_file(filepath)
        functions = extract_functions_from_entries(entries)
        counts = audit_function_schemas(functions)
        total_counts += counts
        print(f"{filename}: {dict(counts)}")

    print("\n--- Aggregate ---")
    total = sum(total_counts.values())
    if total == 0:
        print("No fields found.")
        return

    for category in ["numeric", "categorical-enum", "free-text"]:
        count = total_counts[category]
        pct = count / total * 100
        print(f"  {category:20s}: {count:5d}  ({pct:5.1f}%)")
    print(f"  {'total':20s}: {total:5d}")


if __name__ == "__main__":
    main()
