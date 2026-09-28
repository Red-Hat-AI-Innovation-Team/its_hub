#!/bin/sh
# RUST-INS-015 — PR size limit: max 750 added lines in the review range, excluding
# Cargo manifests/locks, tests, docs, examples, and benchmarks.
set -eu
. "$(dirname "$0")/_range.sh"
added=$(git diff --numstat "$LATTICE_DIFF_BASE" "$LATTICE_HEAD" -- \
  ':(exclude)**/Cargo.toml' ':(exclude)**/Cargo.lock' \
  ':(exclude)tests/**' ':(exclude)**/tests/**' \
  ':(exclude)docs/**' ':(exclude)examples/**' \
  ':(exclude)benches/**' ':(exclude)**/benches/**' 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ {sum += $1} END {print sum + 0}')
echo "added lines (excl Cargo/tests/docs/examples/benches): $added"
if [ "$added" -gt 750 ]; then
  echo "exceeds 750-line PR budget"
  exit 1
fi
echo "within 750-line budget ✓"
