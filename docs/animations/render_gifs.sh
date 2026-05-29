#!/usr/bin/env bash
# Renders all ITS algorithm animations to GIF.
# Usage: ./render_gifs.sh
# Requirements: manimgl (manimlib), ffmpeg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MANIM_PATH="${MANIM_PATH:-/workspace/home/lab/gcg/its_plugin/manim}"
export PYTHONPATH="${MANIM_PATH}:${PYTHONPATH:-}"

OUTPUT_DIR="output"
GIF_DIR="gifs"
mkdir -p "$GIF_DIR"

SCENES=(
    "self_consistency.py SelfConsistencyScene self_consistency"
    "best_of_n.py BestOfNScene best_of_n"
    "beam_search.py BeamSearchScene beam_search"
    "particle_filtering.py ParticleFilteringScene particle_filtering"
)

for entry in "${SCENES[@]}"; do
    read -r file scene name <<< "$entry"
    echo "=== Rendering $scene ==="

    if [ -z "${DISPLAY:-}" ]; then
        xvfb-run -a manimgl "$file" "$scene" -w 2>/dev/null || \
            manimgl "$file" "$scene" -w
    else
        manimgl "$file" "$scene" -w
    fi

    mp4=$(find "$OUTPUT_DIR" -name "*.mp4" -newer "$file" | head -1)
    if [ -z "$mp4" ]; then
        echo "ERROR: No MP4 found for $scene"
        continue
    fi

    echo "=== Converting $name to GIF ==="
    ffmpeg -y -i "$mp4" \
        -filter_complex "[0:v] split [a][b]; [a] palettegen=stats_mode=diff [p]; [b][p] paletteuse=dither=bayer" \
        "$GIF_DIR/${name}.gif"

    echo "=== Done: $GIF_DIR/${name}.gif ==="
done

echo ""
echo "All GIFs rendered to $GIF_DIR/"
ls -lh "$GIF_DIR/"*.gif
