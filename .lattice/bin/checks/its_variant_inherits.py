#!/usr/bin/env python3
"""ITS-INS-010 — SelfConsistency variants inherit from SelfConsistency.

Any class whose name ends in "SelfConsistency" (other than the base itself)
should reuse the base rather than reimplement voting/orchestration.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

import its_hub.core.algorithms as pkg
from its_hub.core.algorithms.self_consistency import SelfConsistency

bad: list[str] = []
for m in pkgutil.iter_modules(pkg.__path__):
    mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != mod.__name__:
            continue
        if name.endswith("SelfConsistency") and name != "SelfConsistency":
            if not issubclass(cls, SelfConsistency):
                bases = ", ".join(b.__name__ for b in cls.__bases__)
                bad.append(f"{name} does not inherit SelfConsistency (bases: {bases})")

if bad:
    print("SelfConsistency variant inheritance violations:")
    print("\n".join(bad))
    sys.exit(1)
print("all *SelfConsistency variants inherit SelfConsistency ✓")
