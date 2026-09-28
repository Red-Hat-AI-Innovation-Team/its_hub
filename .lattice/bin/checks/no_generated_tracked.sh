#!/bin/sh
# REPO-INS-004 — generated / environment files must never be git-tracked.
set -eu
tracked=$(git ls-files \
  | grep -E 'its_hub/_version\.py|its_hub/integration/proto/|(^|/)\.env$|(^|/)\.its-hub/' \
  || true)
if [ -n "$tracked" ]; then
  echo "Generated/ignored files are tracked (should not be):"
  echo "$tracked"
  exit 1
fi
echo "No generated files tracked."
