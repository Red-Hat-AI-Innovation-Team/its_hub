"""Dataset loading utilities for e2e tests."""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DATASET_FILES = {
    "math500": os.path.join(DATA_DIR, "math500_subset.jsonl"),
    "aime2024": os.path.join(DATA_DIR, "aime2024_subset.jsonl"),
}


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file.

    Each record is expected to have at least ``unique_id``, ``problem``,
    and ``answer`` keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_datasets(names: list[str]) -> dict[str, list[dict]]:
    """Load one or more named dataset subsets.

    Returns ``{name: [records]}`` for each successfully loaded dataset.
    Unknown names are printed as warnings and skipped.
    """
    loaded: dict[str, list[dict]] = {}
    for name in names:
        if name not in DATASET_FILES:
            print(f"  Warning: unknown dataset '{name}', skipping")
            continue
        path = DATASET_FILES[name]
        print(f"Loading {name} from {path}...")
        loaded[name] = load_jsonl(path)
        print(f"  Loaded {len(loaded[name])} problems")
    return loaded
