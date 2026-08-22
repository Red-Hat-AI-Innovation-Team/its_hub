#!/usr/bin/env python3
"""ITS-INS-004 — language models implement async agenerate() + agenerate_single().

Runtime introspection over its_hub/core/lms/: every concrete
AbstractLanguageModel subclass must provide both methods, non-abstract and async.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

import its_hub.core.lms as pkg
from its_hub.api.lm import AbstractLanguageModel

REQUIRED = ("agenerate", "agenerate_single")
bad: list[str] = []
for m in pkgutil.iter_modules(pkg.__path__):
    mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls is AbstractLanguageModel or not issubclass(cls, AbstractLanguageModel):
            continue
        if cls.__module__ != mod.__name__:
            continue
        for meth in REQUIRED:
            fn = getattr(cls, meth, None)
            if fn is None or getattr(fn, "__isabstractmethod__", False):
                bad.append(f"{name}: {meth}() not implemented")
            elif not inspect.iscoroutinefunction(fn):
                bad.append(f"{name}: {meth}() is not async")

if bad:
    print("AbstractLanguageModel interface violations:")
    print("\n".join(bad))
    sys.exit(1)
print("all language models implement async agenerate() + agenerate_single() ✓")
