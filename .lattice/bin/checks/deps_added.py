#!/usr/bin/env python3
"""CORE-INS-007 — flag newly added dependency *names* in pyproject.toml.

Parses the base (origin/main) and head pyproject.toml, normalizes requirement
names across project.dependencies and every optional-dependency group, and
reports names present in head but not base. Comparing names (not diff text)
means a version bump of an existing dependency is not a false positive, and
forms like "pkg" and "pkg @ https://..." are handled. New deps require human
justification; this surfaces additions, it cannot verify the justification.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib

NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def norm(req: str) -> str | None:
    """Normalize a requirement string to its PEP 503 project name."""
    m = NAME_RE.match(req)
    if not m:
        return None
    return re.sub(r"[-_.]+", "-", m.group(1)).lower()


def names(pyproject_text: str) -> set[str]:
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    reqs: list[str] = list(project.get("dependencies", []) or [])
    for group in (project.get("optional-dependencies", {}) or {}).values():
        reqs.extend(group or [])
    return {n for r in reqs if (n := norm(r))}


def _git_show(ref_path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "show", ref_path], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    if subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "origin/main"],
        capture_output=True,
    ).returncode != 0:
        print("no origin/main baseline — skipping new-dependency comparison")
        return 0

    base_text = _git_show("origin/main:pyproject.toml")
    if base_text is None:
        print("no pyproject.toml on origin/main — nothing to compare")
        return 0
    with open("pyproject.toml") as fh:
        head_text = fh.read()

    added = sorted(names(head_text) - names(base_text))
    if added:
        print("New dependencies added (need justification):")
        for n in added:
            print(f"  {n}")
        return 1
    print("No new dependencies vs origin/main ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
