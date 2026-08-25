#!/bin/sh
# REPO-INS-005 — lint/format scope is its_hub/ only. Asserts the CI config keeps
# ruff pointed at its_hub/ and not at tests/, scripts/, or benchmarking/.
set -eu
f=.github/workflows/lint.yaml
[ -f "$f" ] || { echo "missing $f"; exit 1; }
grep -q 'ruff check its_hub/' "$f" || { echo "lint.yaml: 'ruff check its_hub/' not found"; exit 1; }
grep -q 'ruff format --check its_hub/' "$f" || { echo "lint.yaml: 'ruff format --check its_hub/' not found"; exit 1; }
if grep -nE 'ruff (check|format)[^#]*\b(tests|scripts|benchmarking)\b' "$f"; then
  echo "lint.yaml applies ruff outside its_hub/"
  exit 1
fi
echo "Lint/format scope is its_hub/ only."
