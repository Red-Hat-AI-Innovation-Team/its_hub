"""Shared lattice helpers: pack loading, versioning, freshness.

Imported by both ``run_check.py`` (runtime) and ``install_checks.py``
(materialization) so the two paths agree on what a check is and whether it is
installable on this recipient.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LATTICE_DIR = REPO_ROOT / ".lattice"
PACKS_DIR = LATTICE_DIR / "packs"
SCOPE_MAP = LATTICE_DIR / "scope-map.yaml"
VERSION_FILE = LATTICE_DIR / "version.yaml"
TRACES_DIR = LATTICE_DIR / "traces"

INSTALLABLE_TIERS = ("inner", "gate", "deep")


def load_version() -> dict[str, Any]:
    """Return the parsed ``version.yaml`` (declared lattice version + notes)."""
    return yaml.safe_load(VERSION_FILE.read_text()) or {}


def lattice_version_hash() -> str:
    """Content hash of composed packs + scope map (SPEC §3.5), truncated.

    Immutable identity of the current lattice state; recorded in every T2 trace
    so events from different lattice versions are never conflated.
    """
    h = hashlib.sha256()
    for path in sorted(PACKS_DIR.glob("*.yaml")):
        h.update(path.read_bytes())
    if SCOPE_MAP.exists():
        h.update(SCOPE_MAP.read_bytes())
    return h.hexdigest()[:12]


def iter_checks() -> list[dict[str, Any]]:
    """Yield every ``kind: check`` rule across all packs, pack context attached."""
    checks: list[dict[str, Any]] = []
    for path in sorted(PACKS_DIR.glob("*.yaml")):
        pack = yaml.safe_load(path.read_text()) or {}
        for rule in pack.get("rules", []) or []:
            if rule.get("kind") != "check":
                continue
            enriched = dict(rule)
            enriched["_pack"] = pack.get("name", path.stem)
            enriched["_layer"] = pack.get("layer")
            checks.append(enriched)
    return checks


def get_check(rule_id: str) -> dict[str, Any] | None:
    """Return the single check rule with ``rule_id``, or ``None``."""
    for check in iter_checks():
        if check.get("id") == rule_id:
            return check
    return None


def scope_prefix_exists(scope: str) -> bool:
    """Freshness for a scope glob (SPEC §5.4).

    ``**`` (whole repo) always passes. Otherwise the static directory prefix
    before the first wildcard must exist in the recipient — this is what parks
    the grafted ``rust/**`` packs and any ``notebooks/**`` check on a repo that
    has neither.
    """
    if not scope or scope == "**":
        return True
    prefix = scope.split("*", 1)[0].rstrip("/")
    if not prefix:
        return True
    return (REPO_ROOT / prefix).exists()


def freshness_reason(check: dict[str, Any]) -> str | None:
    """Return a skip reason if the check is not installable here, else ``None``."""
    if not check.get("check_command"):
        return "no check_command (protocol/action check, not directly installable)"
    scope = check.get("scope", "**")
    if not scope_prefix_exists(scope):
        return f"scope '{scope}' does not exist in this repository"
    return None


def installable_checks() -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Split checks into (installable, skipped-with-reason) for this recipient."""
    keep: list[dict[str, Any]] = []
    skip: list[tuple[dict[str, Any], str]] = []
    for check in iter_checks():
        reason = freshness_reason(check)
        if reason:
            skip.append((check, reason))
        else:
            keep.append(check)
    return keep, skip
