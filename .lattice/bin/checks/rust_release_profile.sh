#!/bin/sh
# RUST-INS-012 — release profile hardening. Asserts rust/Cargo.toml [profile.release]
# enables lto, single codegen unit, overflow-checks, and strip. NOTE: panic=abort
# is deliberately NOT required — this crate is a PyO3 cdylib extension and PyO3
# relies on unwinding (catch_unwind) to turn Rust panics into Python exceptions,
# so panic=abort would break that. Records the gap when the section is absent.
set -eu
f=rust/Cargo.toml
[ -f "$f" ] || { echo "missing $f"; exit 1; }
sec=$(awk '/^\[profile\.release\]/{p=1;next} /^\[/{p=0} p' "$f")
[ -n "$sec" ] || { echo "rust/Cargo.toml has no [profile.release] section"; exit 1; }
missing=""
echo "$sec" | grep -qE 'lto[[:space:]]*=[[:space:]]*(true|"fat"|"thin")' || missing="$missing lto"
echo "$sec" | grep -qE 'codegen-units[[:space:]]*=[[:space:]]*1'             || missing="$missing codegen-units=1"
echo "$sec" | grep -qE 'overflow-checks[[:space:]]*=[[:space:]]*true'        || missing="$missing overflow-checks"
echo "$sec" | grep -qE 'strip[[:space:]]*=' || missing="$missing strip"
if [ -n "$missing" ]; then
  echo "release profile missing:$missing"
  exit 1
fi
echo "release profile hardened ✓"
