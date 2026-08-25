#!/usr/bin/env python3
"""ITS-INS-002 — every scaling algorithm implements async ainfer() + infer().

Runtime introspection: every concrete AbstractScalingAlgorithm subclass defined
under its_hub/core/algorithms/ must have a non-abstract, async ainfer and an
infer (inherited from the base is fine).
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

import its_hub.core.algorithms as pkg
from its_hub.api.algorithm import AbstractScalingAlgorithm

bad: list[str] = []
for m in pkgutil.iter_modules(pkg.__path__):
    mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls is AbstractScalingAlgorithm or not issubclass(cls, AbstractScalingAlgorithm):
            continue
        if cls.__module__ != mod.__name__:
            continue
        ainfer = getattr(cls, "ainfer", None)
        infer = getattr(cls, "infer", None)
        if ainfer is None or getattr(ainfer, "__isabstractmethod__", False):
            bad.append(f"{name}: ainfer() not implemented")
        elif not inspect.iscoroutinefunction(ainfer):
            bad.append(f"{name}: ainfer() is not async")
        if infer is None:
            bad.append(f"{name}: infer() missing")

if bad:
    print("AbstractScalingAlgorithm interface violations:")
    print("\n".join(bad))
    sys.exit(1)
print("all algorithms implement async ainfer() + infer() ✓")
