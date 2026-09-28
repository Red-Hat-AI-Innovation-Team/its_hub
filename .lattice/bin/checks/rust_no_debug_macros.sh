#!/bin/sh
# RUST-INS-003 (partial) — no println!/eprintln!/dbg! in rust production sources.
# Runtime narration belongs in tracing. NOTE: this greps all of rust/src, so it
# also flags any debug macro inside #[cfg(test)] modules — acceptable, documented.
set -eu
d=rust/src
[ -d "$d" ] || { echo "no $d (nothing to check)"; exit 0; }
hits=$(grep -rnE '\b(println!|eprintln!|dbg!)' "$d" || true)
if [ -n "$hits" ]; then
  echo "Debug macros in production Rust source (use tracing instead):"
  echo "$hits"
  exit 1
fi
echo "No debug macros in $d ✓"
