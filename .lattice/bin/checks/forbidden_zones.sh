#!/bin/sh
# CORE-INS-002 — flag changes to forbidden zones (CI workflows, lockfiles, SPEC,
# version/release files) in the review range. These require explicit human
# permission; the check surfaces them for review (it cannot verify permission).
set -eu
. "$(dirname "$0")/_range.sh"
touched=$(git diff --name-only "$LATTICE_DIFF_BASE" "$LATTICE_HEAD" 2>/dev/null | grep -E \
  '^\.github/workflows/|(^|/)uv\.lock$|(^|/)Cargo\.lock$|(^|/)poetry\.lock$|(^|/)SPEC\.md$|its_hub/_version\.py|(^|/)version\.ya?ml$' \
  || true)
if [ -n "$touched" ]; then
  echo "Forbidden-zone files changed (require explicit permission / human review):"
  echo "$touched"
  exit 1
fi
echo "No forbidden-zone changes ✓"
