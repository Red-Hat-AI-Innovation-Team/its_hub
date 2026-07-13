"""Copy the testmini-referenced audio clips out of the full audio tree.

The 1,099 clips referenced by the testmini parquet (1,088 wav + 11 mp3 — never
glob '*.wav') are a strict filename-subset of the full test set's data.zip.
The loader resolves subsets 'full'/'le30s' relative to the testmini root, so
these files must exist beside the parquets.

--check mode copies nothing and exits 1 if any referenced clip is missing at
either the source (audio root) or the destination (testmini root).
"""

import argparse
import os
import shutil
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testmini-root", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--check", action="store_true", help="verify only, copy nothing")
    args = ap.parse_args()

    parquet = os.path.join(args.testmini_root, "data", "testmini-00000-of-00001.parquet")
    df = pd.read_parquet(parquet)
    rel = sorted({p for row in df["audio_path"] if row is not None for p in list(row)})

    missing, present, copied = [], 0, 0
    for r in rel:
        src = os.path.join(args.audio_root, r)
        dst = os.path.join(args.testmini_root, r)
        if not os.path.exists(src):
            missing.append(f"(source missing) {r}")
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            present += 1
            continue
        if args.check:
            missing.append(f"(destination missing) {r}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"testmini audio: {len(rel)} referenced, {present} already present, {copied} copied")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for m in missing[:20]:
            print("  ", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
