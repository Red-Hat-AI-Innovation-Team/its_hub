#!/bin/sh
# RUST-INS-001 — the rust crate uses edition 2024.
set -eu
f=rust/Cargo.toml
[ -f "$f" ] || { echo "missing $f"; exit 1; }
if grep -qE '^edition[[:space:]]*=[[:space:]]*"2024"' "$f"; then
  echo "edition 2024 ✓"
else
  echo "rust/Cargo.toml does not set edition = \"2024\""
  exit 1
fi
