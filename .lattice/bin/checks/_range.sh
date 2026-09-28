# Shared: resolve a root-safe review base/range for git-based checks.
# Sourced, not executed. Sets:
#   LATTICE_HEAD       — the review tip commit
#   LATTICE_DIFF_BASE  — diff LATTICE_HEAD against this (merge-base, else empty tree)
#   LATTICE_LOG_RANGE  — revision range for git log
#
# On a pull_request event the checked-out HEAD is a synthetic merge commit that
# is neither signed off nor conventionally formatted. Using the PR head SHA as
# the tip excludes exactly that synthetic commit while still inspecting real
# commits (including developer-created merges). The empty-tree fallback keeps
# diffs valid on an initial commit or a depth-one checkout lacking origin/main.

LATTICE_HEAD="HEAD"
if [ -n "${GITHUB_EVENT_PATH:-}" ] && [ -f "$GITHUB_EVENT_PATH" ]; then
  _sha=$(python3 -c "import json,os; d=json.load(open(os.environ['GITHUB_EVENT_PATH'])); print((d.get('pull_request') or {}).get('head',{}).get('sha') or '')" 2>/dev/null || true)
  if [ -n "$_sha" ] && git rev-parse --verify -q "${_sha}^{commit}" >/dev/null 2>&1; then
    LATTICE_HEAD="$_sha"
  fi
fi

if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
  # merge-base, not origin/main: if main advances after the branch started, a
  # plain origin/main..HEAD diff would fold in main's own changes.
  LATTICE_DIFF_BASE="$(git merge-base origin/main "$LATTICE_HEAD" 2>/dev/null || echo origin/main)"
  LATTICE_LOG_RANGE="origin/main..$LATTICE_HEAD"
else
  LATTICE_DIFF_BASE="$(git hash-object -t tree /dev/null)"
  LATTICE_LOG_RANGE="$LATTICE_HEAD"
fi
