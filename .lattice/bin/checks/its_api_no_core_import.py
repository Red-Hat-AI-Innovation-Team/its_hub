#!/usr/bin/env python3
"""ITS-INS-001 — api/ defines interfaces; it must not depend on core/.

The crisp reading of "public interfaces in api/, implementations in core/" is
dependency direction: nothing under its_hub/api/ may import its_hub.core, whether
via an absolute import or a relative one (`from ..core import X`,
`from .. import core`, `from ..core.sub import Y`). Implementations depend on
interfaces, never the reverse.
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path("its_hub/api")
TARGET = "its_hub.core"


def _package_of(path: pathlib.Path) -> str:
    """Dotted package for a module file, e.g. its_hub/api/lm.py -> its_hub.api."""
    return ".".join(path.with_suffix("").parts[:-1])


def _resolve(pkg: str, level: int, module: str | None) -> str:
    """Resolve a relative import target to an absolute dotted path."""
    if level == 0:
        return module or ""
    parts = pkg.split(".")
    base = parts[: len(parts) - (level - 1)]
    absmod = ".".join(base)
    if module:
        absmod = f"{absmod}.{module}" if absmod else module
    return absmod


def _hits_core(target: str) -> bool:
    return target == TARGET or target.startswith(TARGET + ".")


def main() -> int:
    bad: list[str] = []
    for p in sorted(ROOT.rglob("*.py")):
        pkg = _package_of(p)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if _hits_core(a.name):
                        bad.append(f"{p}:{node.lineno}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                target = _resolve(pkg, node.level, node.module)
                if _hits_core(target):
                    bad.append(f"{p}:{node.lineno}: from {'.' * node.level}{node.module or ''} import ...")
                elif node.level > 0 and node.module is None:
                    # `from .. import core` — the imported name is the submodule.
                    for a in node.names:
                        if _hits_core(f"{target}.{a.name}" if target else a.name):
                            bad.append(f"{p}:{node.lineno}: from {'.' * node.level} import {a.name}")

    if bad:
        print("api/ must not import core/ (interface must not depend on implementation):")
        print("\n".join(bad))
        return 1
    print("its_hub/api/ has no dependency on its_hub/core/ ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
