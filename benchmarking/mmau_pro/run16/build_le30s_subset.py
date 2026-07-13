"""Rebuild the <=30 s subset parquets (provenance for run16/data/*.parquet).

Keeps parquet rows whose clips ALL have duration <= 30.0 s (soundfile header
read) — the fit rule for Qwen2-Audio's Whisper-style 30 s encoder window
(RESULTS.md §16). The two committed parquets used slightly different scopes
(verified by id-set equality against the originals):

  test_le30s (2,256 rows incl. open-ended -> 2,190 MCQ via the loader):
      --src-parquet .../mmau_pro_testmini/data/test-00000-of-00001.parquet \
      --audio-root  .../mmau_pro_audio
  testmini_le30s (411 rows, MCQ-only at build time):
      --src-parquet .../mmau_pro_testmini/data/testmini-00000-of-00001.parquet \
      --audio-root  .../mmau_pro_testmini --mcq-only

--verify-against compares row count + id set with an existing parquet (content
equality; byte equality is not expected — parquet metadata differs per write).
"""

import argparse
import os

import pandas as pd
import soundfile as sf


def fits(row, audio_root, cache):
    paths = [] if row is None else list(row)
    if not paths:
        return False
    for p in paths:
        fp = os.path.join(audio_root, p)
        if fp not in cache:
            try:
                info = sf.info(fp)
                cache[fp] = info.frames / info.samplerate
            except Exception:
                cache[fp] = None
        dur = cache[fp]
        if dur is None or dur > 30.0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-parquet", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", help="write the filtered parquet here")
    ap.add_argument("--verify-against", help="compare row count + id set with this parquet")
    ap.add_argument("--mcq-only", action="store_true",
                    help="also drop open-ended rows (empty choices) — the testmini_le30s scope")
    args = ap.parse_args()

    df = pd.read_parquet(args.src_parquet)
    if args.mcq_only:
        df = df[df["choices"].map(lambda c: c is not None and len(list(c)) > 0)]
    cache = {}
    out = df[df["audio_path"].map(lambda r: fits(r, args.audio_root, cache))].reset_index(drop=True)
    print(f"{len(df)} rows -> {len(out)} with all clips <= 30 s"
          + (" (MCQ-only)" if args.mcq_only else ""))

    if args.out:
        out.to_parquet(args.out, index=False)
        print(f"wrote {args.out}")
    if args.verify_against:
        ref = pd.read_parquet(args.verify_against)
        assert len(ref) == len(out), f"row count {len(out)} != committed {len(ref)}"
        assert set(ref["id"]) == set(out["id"]), "id sets differ"
        print(f"VERIFY OK vs {args.verify_against}")


if __name__ == "__main__":
    main()
