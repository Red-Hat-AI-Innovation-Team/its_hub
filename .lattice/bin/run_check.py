#!/usr/bin/env python3
"""Lattice check runner — the deterministic enforcement primitive (SPEC §5.3/§6.1).

Runs one or more checks by id or by tier, executing each check's
``check_command`` and recording a T2 trace event per execution. Honors rule
state:

- ``warn-only`` — record the outcome, **always exit 0** (never blocks; Axiom A5).
- ``enforced``  — exit non-zero if any check's exit status differs from its
  ``expected_exit`` (blocks per the tier).

Both the generated CI steps (GATE) and the PostToolUse hook (INNER) call this
runner, so every enforcement path emits identical, comparable records.

Usage:
    run_check.py PYTHON-CHK-001 [MORE-IDS ...]
    run_check.py --tier inner
    run_check.py --tier gate --event ci
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    TRACES_DIR,
    freshness_reason,
    get_check,
    installable_checks,
    lattice_version_hash,
    load_version,
)

_OUTPUT_LIMIT = 2000


def _select(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve the requested ids/tier into runnable checks and warnings."""
    keep, _ = installable_checks()
    if args.tier:
        return [c for c in keep if c.get("tier") == args.tier], []
    selected: list[dict[str, Any]] = []
    warnings: list[str] = []
    for rule_id in args.ids:
        check = get_check(rule_id)
        if check is None:
            warnings.append(f"unknown check id: {rule_id}")
            continue
        reason = freshness_reason(check)
        if reason:
            warnings.append(f"{rule_id} not installable: {reason}")
            continue
        selected.append(check)
    return selected, warnings


def _execute(command: str) -> tuple[int, str]:
    """Run a check command once from the repo root, returning (exit, output)."""
    try:
        proc = subprocess.run(  # noqa: S603 - commands come from vetted packs
            shlex.split(command),
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return proc.returncode, (proc.stdout + proc.stderr)[-_OUTPUT_LIMIT:]
    except FileNotFoundError as exc:
        return 127, f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, "check timed out after 600s"


def _run_group(
    command: str,
    checks: list[dict[str, Any]],
    event: str,
    version: str,
    version_hash: str,
) -> int:
    """Run one command once; emit a T2 event per check sharing it.

    Composition can leave two rules with the same command and scope (e.g. the
    repo and its-domain build checks); we execute once but record each rule's
    outcome independently. Returns the count of enforced checks that failed.
    """
    exit_status, output = _execute(command)
    blocking = 0
    for check in checks:
        expected = int(check.get("expected_exit", 0))
        passed = exit_status == expected
        _emit_t2(check, event, version, version_hash, exit_status, expected, passed, output)

        state = check.get("state", "warn-only")
        marker = "PASS" if passed else ("FAIL" if state == "enforced" else "WARN")
        print(f"[{marker}] {check['id']} ({check.get('tier')}/{state}) — {command}")
        if not passed and state == "enforced":
            blocking += 1
    if not exit_status == 0 and output.strip():
        for line in output.strip().splitlines()[-8:]:
            print(f"       {line}")
    return blocking


def _emit_t2(
    check: dict[str, Any],
    event: str,
    version: str,
    version_hash: str,
    exit_status: int,
    expected: int,
    passed: bool,
    output: str,
) -> None:
    """Append one append-only T2 check event (SPEC §6.1). Never raises."""
    record = {
        "rule_id": check["id"],
        "lattice_version": version,
        "lattice_version_hash": version_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": check.get("tier"),
        "state": check.get("state", "warn-only"),
        "scope": check.get("scope", "**"),
        "command": check["check_command"],
        "exit_status": exit_status,
        "expected_exit": expected,
        "passed": passed,
        "parsed_output": output.strip()[-_OUTPUT_LIMIT:],
        "reaction": None,  # filled by the improvement loop from later traces
        "event": event,
    }
    try:
        TRACES_DIR.joinpath("T2").mkdir(parents=True, exist_ok=True)
        day = record["timestamp"][:10]
        with TRACES_DIR.joinpath("T2", f"{day}.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # tracing must never block a run (Axiom A6)
        print(f"       (warning: could not write T2 trace: {exc})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run lattice checks and record T2 traces.")
    parser.add_argument("ids", nargs="*", help="Check ids to run, e.g. PYTHON-CHK-001")
    parser.add_argument("--tier", choices=("inner", "gate", "deep"), help="Run all installable checks of this tier")
    parser.add_argument("--event", default="manual", help="Trace context label (ci, posttooluse, manual)")
    args = parser.parse_args(argv)

    if not args.ids and not args.tier:
        parser.error("provide check id(s) or --tier")

    version = str(load_version().get("version", "unknown"))
    version_hash = lattice_version_hash()

    checks, warnings = _select(args)
    for warning in warnings:
        print(f"[skip] {warning}")
    if not checks:
        print("no runnable checks selected")
        return 0

    grouped: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        grouped.setdefault(check["check_command"], []).append(check)

    blocking_failures = 0
    for command, group in grouped.items():
        blocking_failures += _run_group(command, group, args.event, version, version_hash)

    if blocking_failures:
        print(f"\n{blocking_failures} enforced check(s) failed — blocking.")
        return 1
    return 0  # warn-only outcomes never block


if __name__ == "__main__":
    raise SystemExit(main())
