#!/usr/bin/env python3
"""ITS-INS-012 — no unguarded module-time imports of optional-extra packages.

A module in its_hub/ must not import a dependency that lives only in an optional
extra at module import time; that breaks CI for users who installed only the
base/lm extras. Imports are exempt only when deferred into a function body or
guarded by a try block whose except handles ImportError. Every other
module-executed block (if / for / while / with / class body / try-else /
try-finally) is inspected — an `if flag: import scipy` still runs on import.

This is the defect ITS-INS-012 was mined from (scipy imported unconditionally
but declared only in the experimental extra).
"""
from __future__ import annotations

import ast
import pathlib
import sys

BANNED = {"scipy", "transformers", "reward_hub", "math_verify", "datasets", "matplotlib"}


def _banned_hit(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Import):
        for a in node.names:
            if a.name.split(".")[0] in BANNED:
                return f"import {a.name}"
    elif isinstance(node, ast.ImportFrom):
        # level == 0 only: `from .datasets import x` is a local relative module,
        # not the PyPI `datasets` extra.
        if node.level == 0 and (node.module or "").split(".")[0] in BANNED:
            return f"from {node.module} import ..."
    return None


def _handles_import_error(handlers: list[ast.excepthandler]) -> bool:
    for h in handlers:
        t = h.type
        if t is None:  # bare except
            return True
        names = [t] if not isinstance(t, ast.Tuple) else t.elts
        for n in names:
            if isinstance(n, ast.Name) and n.id in {"ImportError", "ModuleNotFoundError", "Exception"}:
                return True
    return False


def _scan(body: list[ast.stmt], path: pathlib.Path, out: list[str]) -> None:
    """Walk statements executed at module import time; collect banned imports."""
    for node in body:
        hit = _banned_hit(node)
        if hit:
            out.append(f"{path}:{node.lineno}: {hit}")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue  # deferred until called — legitimately lazy
        if isinstance(node, ast.Try):
            if not _handles_import_error(node.handlers):
                _scan(node.body, path, out)  # unguarded try body still runs
            _scan(node.orelse, path, out)
            _scan(node.finalbody, path, out)
            continue
        for attr in ("body", "orelse", "finalbody"):
            block = getattr(node, attr, None)
            if isinstance(block, list):
                _scan(block, path, out)


def main() -> int:
    bad: list[str] = []
    for p in sorted(pathlib.Path("its_hub").rglob("*.py")):
        _scan(ast.parse(p.read_text()).body, p, bad)
    if bad:
        print("unguarded module-time imports of optional-extra packages in its_hub/:")
        print("\n".join(bad))
        return 1
    print("no unguarded optional-extra imports in its_hub/ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
