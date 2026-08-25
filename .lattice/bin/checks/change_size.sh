#!/bin/sh
# CORE-INS-005 — change-size budget: prefer <= ~400 changed lines and <= ~10 files
# in the review range, excluding lockfiles and generated files.
set -eu
. "$(dirname "$0")/_range.sh"
stat=$(git diff --numstat "$LATTICE_DIFF_BASE" "$LATTICE_HEAD" -- \
  ':(exclude)**/uv.lock' ':(exclude)**/Cargo.lock' ':(exclude)**/poetry.lock' \
  ':(exclude)its_hub/_version.py' ':(exclude)its_hub/integration/proto/**' 2>/dev/null)
lines=$(echo "$stat" | awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {s += $1 + $2} END {print s + 0}')
files=$(echo "$stat" | grep -c . || true)
echo "changed lines: $lines, files: $files (budget: 400 lines / 10 files)"
if [ "$lines" -gt 400 ] || [ "$files" -gt 10 ]; then
  echo "exceeds change-size budget — consider splitting the PR"
  exit 1
fi
echo "within change-size budget ✓"
