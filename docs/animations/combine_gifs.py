#!/usr/bin/env python3
"""Combine algorithm GIFs into a single cycling GIF with brief fade transitions.

Pipeline:
  1. Convert each algorithm GIF to MP4 with fade-out at end and fade-in at start
  2. Insert a short white gap between segments
  3. Concatenate, then convert to optimized GIF via two-pass palettegen
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).parent
GIF_DIR = SCRIPT_DIR / "gifs"
OUTPUT = GIF_DIR / "its_hub_algorithms.gif"

WIDTH, HEIGHT = 960, 540
FPS = 15
BG_HEX = "F8F8F8"

ALGORITHMS = [
    "self_consistency.gif",
    "best_of_n.gif",
    "particle_filtering.gif",
]

FADE_DURATION_S = 0.3
PAUSE_AFTER_S = 1.0
GAP_DURATION_S = 0.3


def ffmpeg(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "warning", *args],
        check=True,
    )


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def make_bg_frame(out_path: Path) -> None:
    r, g, b = int(BG_HEX[0:2], 16), int(BG_HEX[2:4], 16), int(BG_HEX[4:6], 16)
    img = Image.new("RGB", (WIDTH, HEIGHT), (r, g, b))
    img.save(out_path)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        segments: list[Path] = []

        bg_png = tmp / "bg.png"
        make_bg_frame(bg_png)

        for i, filename in enumerate(ALGORITHMS):
            gif_path = GIF_DIR / filename
            name = filename.removesuffix(".gif")
            print(f"Processing {name}...")

            # GIF -> MP4 with fade-in at start and fade-out at end
            raw_mp4 = tmp / f"{name}_raw.mp4"
            ffmpeg(
                "-i", str(gif_path),
                "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
                "-pix_fmt", "yuv420p",
                str(raw_mp4),
            )

            duration = get_duration(raw_mp4)
            fade_out_start = duration - FADE_DURATION_S

            faded_mp4 = tmp / f"{name}_faded.mp4"
            fade_filter = (
                f"fade=t=in:st=0:d={FADE_DURATION_S}:color=0x{BG_HEX},"
                f"fade=t=out:st={fade_out_start:.2f}:d={FADE_DURATION_S}:color=0x{BG_HEX}"
            )
            ffmpeg(
                "-i", str(raw_mp4),
                "-vf", fade_filter,
                "-pix_fmt", "yuv420p",
                str(faded_mp4),
            )
            segments.append(faded_mp4)

            # Short background gap between algorithms
            gap_mp4 = tmp / f"{name}_gap.mp4"
            ffmpeg(
                "-loop", "1", "-i", str(bg_png),
                "-t", str(GAP_DURATION_S),
                "-vf", f"scale={WIDTH}:{HEIGHT},fps={FPS}",
                "-pix_fmt", "yuv420p",
                str(gap_mp4),
            )
            segments.append(gap_mp4)

        # Concat all segments
        concat_list = tmp / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{s}'" for s in segments)
        )
        concat_mp4 = tmp / "combined.mp4"
        ffmpeg(
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(concat_mp4),
        )

        # Two-pass GIF encoding
        palette = tmp / "palette.png"
        print("\nGenerating optimized palette...")
        ffmpeg(
            "-i", str(concat_mp4),
            "-vf", f"fps={FPS},palettegen=stats_mode=diff",
            str(palette),
        )

        print("Encoding final GIF...")
        ffmpeg(
            "-i", str(concat_mp4),
            "-i", str(palette),
            "-filter_complex",
            f"[0:v] fps={FPS} [v]; [v][1:v] paletteuse=dither=bayer:bayer_scale=3",
            str(OUTPUT),
        )

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
