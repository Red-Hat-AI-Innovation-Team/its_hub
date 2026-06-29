#!/usr/bin/env bash
# Combine Self-Consistency, Best-of-N, and Particle Filtering MP4s
# into a single video with title cards, progress bar, and section indicators.
#
# Usage: conda run -n manim_render bash docs/animations/combine_video.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
VIDEO_DIR="${SCRIPT_DIR}/videos"
OUT="${VIDEO_DIR}/its_hub_algorithms.mp4"

W=960; H=540; FPS=15
BG="0xF8F8F8"; TEXT_COLOR="0x333333"
TITLE_DURATION=2
FONT="/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf"
FONT_LIGHT="/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-ExtraLight.ttf"

BAR_H=5
BAR_MARGIN=20
BAR_GAP=4
BAR_INACTIVE="0xDDDDDD@1"
BAR_ACTIVE="0x4A90D9@1"

mkdir -p "$VIDEO_DIR"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

TITLES=("Self-Consistency" "Best-of-N" "Particle Filtering")
SUBTITLES=("Consensus voting over N samples" "Reward-ranked selection" "Sequential Monte Carlo resampling")
FILES=("SelfConsistencyScene.mp4" "BestOfNScene.mp4" "ParticleFilteringScene.mp4")

TOTAL=${#TITLES[@]}
CONCAT_LIST="${TMPDIR}/concat.txt"
> "$CONCAT_LIST"

# ── Phase 1: build title cards + concat list, track segment durations ──
declare -a SEG_DURATIONS
cumulative=0

for i in "${!TITLES[@]}"; do
    title="${TITLES[$i]}"
    subtitle="${SUBTITLES[$i]}"
    src="${OUTPUT_DIR}/${FILES[$i]}"
    card="${TMPDIR}/card_${i}.mp4"
    n=$((i + 1))

    echo "Creating title card: ${title} (${n}/${TOTAL})"

    ffmpeg -y -loglevel warning \
        -f lavfi -i "color=c=${BG}:s=${W}x${H}:d=${TITLE_DURATION}:r=${FPS}" \
        -vf "\
            drawtext=fontfile=${FONT}:text='${title}':fontcolor=${TEXT_COLOR}:fontsize=42:x=(w-tw)/2:y=(h-th)/2-20,\
            drawtext=fontfile=${FONT_LIGHT}:text='${subtitle}':fontcolor=0x888888:fontsize=22:x=(w-tw)/2:y=(h-th)/2+30,\
            drawtext=fontfile=${FONT_LIGHT}:text='${n} / ${TOTAL}':fontcolor=0xAAAAAA:fontsize=18:x=(w-tw)/2:y=h-40,\
            fade=t=in:st=0:d=0.4:color=${BG},\
            fade=t=out:st=$((TITLE_DURATION - 1)).0:d=0.6:color=${BG}" \
        -pix_fmt yuv420p -c:v libx264 -profile:v baseline \
        "$card"

    clip_dur=$(ffprobe -v quiet -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$src")
    seg_dur=$(awk "BEGIN{printf \"%.6f\", $TITLE_DURATION + $clip_dur}")
    SEG_DURATIONS+=("$seg_dur")

    echo "file '${card}'" >> "$CONCAT_LIST"
    echo "file '${src}'"  >> "$CONCAT_LIST"
done

# ── Phase 2: concat without progress bar ──
RAW="${TMPDIR}/raw.mp4"
echo ""
echo "Concatenating segments..."
ffmpeg -y -loglevel warning \
    -f concat -safe 0 -i "$CONCAT_LIST" \
    -c:v libx264 -profile:v baseline -pix_fmt yuv420p \
    "$RAW"

TOTAL_DUR=$(ffprobe -v quiet -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$RAW")

# ── Phase 3: build progress bar filter ──
# Bar spans full width minus margins, split into N segments with small gaps
BAR_TOTAL_W=$((W - 2 * BAR_MARGIN))
BAR_Y=$((H - BAR_H - 10))

# Compute pixel widths proportional to duration
declare -a SEG_WIDTHS
remaining_px=$BAR_TOTAL_W
for i in "${!SEG_DURATIONS[@]}"; do
    if [ "$i" -eq $((TOTAL - 1)) ]; then
        SEG_WIDTHS+=("$remaining_px")
    else
        px=$(awk "BEGIN{printf \"%d\", $BAR_TOTAL_W * ${SEG_DURATIONS[$i]} / $TOTAL_DUR}")
        px=$((px - BAR_GAP))
        SEG_WIDTHS+=("$px")
        remaining_px=$((remaining_px - px - BAR_GAP))
    fi
done

# Build drawbox filters: inactive (gray) for all segments, active (blue) for current
FILTER=""
x_offset=$BAR_MARGIN
cumulative="0"

for i in "${!TITLES[@]}"; do
    seg_w=${SEG_WIDTHS[$i]}
    seg_start="$cumulative"
    seg_end=$(awk "BEGIN{printf \"%.6f\", $cumulative + ${SEG_DURATIONS[$i]}}")

    # Inactive segment (always drawn)
    FILTER+="drawbox=x=${x_offset}:y=${BAR_Y}:w=${seg_w}:h=${BAR_H}:color=${BAR_INACTIVE}:t=fill,"

    # Active highlight (drawn only during this segment's time range)
    FILTER+="drawbox=x=${x_offset}:y=${BAR_Y}:w=${seg_w}:h=${BAR_H}:color=${BAR_ACTIVE}:t=fill:enable='between(t,${seg_start},${seg_end})',"

    x_offset=$((x_offset + seg_w + BAR_GAP))
    cumulative="$seg_end"
done

# Remove trailing comma
FILTER="${FILTER%,}"

echo "Adding progress bar..."
ffmpeg -y -loglevel warning \
    -i "$RAW" \
    -vf "$FILTER" \
    -c:v libx264 -profile:v baseline -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "Saved: ${OUT} (${SIZE})"
