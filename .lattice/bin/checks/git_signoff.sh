#!/bin/sh
# REPO-INS-001 — every commit in the review range carries a DCO Signed-off-by
# trailer. Range excludes the synthetic PR merge commit (via _range.sh) but still
# inspects developer-created merges. Fails closed if git cannot read the range.
set -eu
. "$(dirname "$0")/_range.sh"
if ! missing=$(git log "$LATTICE_LOG_RANGE" --invert-grep --grep='Signed-off-by:' --format='%h %s' 2>&1); then
  echo "error: cannot inspect range $LATTICE_LOG_RANGE:"
  echo "$missing"
  exit 1
fi
if [ -n "$missing" ]; then
  echo "Commits missing Signed-off-by ($LATTICE_LOG_RANGE):"
  echo "$missing"
  exit 1
fi
echo "All commits in $LATTICE_LOG_RANGE are signed off."
