#!/usr/bin/env python3
"""ITS-INS-008 — algorithm results must track token usage (GenerationUsage).

Deterministic proxy: every algorithm module under its_hub/core/algorithms/
references token usage. Modules that never mention `usage` are not threading
GenerationUsage through their result — the exact gap this rule guards.
"""
from __future__ import annotations

import pathlib
import sys

PKG = pathlib.Path("its_hub/core/algorithms")
SKIP = {"__init__.py"}
missing: list[str] = []
for p in sorted(PKG.glob("*.py")):
    if p.name in SKIP:
        continue
    if "usage" not in p.read_text():
        missing.append(p.name)

if missing:
    print("algorithm modules with no token-usage tracking (GenerationUsage):")
    print("\n".join(missing))
    sys.exit(1)
print("all algorithm modules reference token usage ✓")
