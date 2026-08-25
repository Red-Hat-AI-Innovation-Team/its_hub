#!/bin/sh
# CORE-INS-008 — commit subjects follow the repo's conventional format across the
# review range. Accepts both "type: desc" and this repo's "(type) desc" style.
# The "why, not what" quality of the body is not machine-checkable. Range excludes
# the synthetic PR merge commit (via _range.sh); fails closed on a git error.
set -eu
. "$(dirname "$0")/_range.sh"
types='feat|fix|chore|docs|refactor|test|perf|build|ci|style|revert'
if ! subjects=$(git log "$LATTICE_LOG_RANGE" --format='%s' 2>&1); then
  echo "error: cannot inspect range $LATTICE_LOG_RANGE:"
  echo "$subjects"
  exit 1
fi
bad=$(printf '%s\n' "$subjects" | grep -vE "^(\(($types)\)|($types)(\([a-z0-9_-]+\))?):? " || true)
if [ -n "$bad" ]; then
  echo "Commit subjects not in conventional format:"
  printf '  %s\n' "$bad"
  exit 1
fi
echo "All commit subjects follow conventional format ✓"
