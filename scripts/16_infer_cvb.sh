#!/bin/bash
# 16_infer_cvb.sh
# Run the Phase 9 inference pipeline on all CVB clips.
# Each clip directory under data/raw/cvb/raw_frames/ is processed independently.
#
# Run from project root: bash scripts/16_infer_cvb.sh
#
# Optional flags:
#   --output_dir  <path>    Root output directory (default: results/inference/cvb)
#   --conf_thresh <float>   Detection confidence threshold (overrides config)
#   --cleanup               Delete intermediate files after each run
#   --resume                Skip clips whose output directory already exists
#   --limit       <int>     Process at most N clips (useful for a test run)
#
# Each clip gets its own subdirectory under --output_dir named after the clip.
# A shared timing.csv is appended under --output_dir/timing.csv.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer

cd "$REPO_ROOT"

FRAMES_ROOT="data/raw/cvb/raw_frames"
OUTPUT_DIR="results/inference/cvb"
CONF_THRESH=""
CLEANUP=""
RESUME=0
LIMIT=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --conf_thresh) CONF_THRESH="$2"; EXTRA_ARGS+=(--conf_thresh "$2"); shift 2 ;;
        --cleanup)     CLEANUP="--cleanup"; EXTRA_ARGS+=(--cleanup); shift ;;
        --resume)      RESUME=1; shift ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mapfile -t CLIPS < <(ls -1 "$FRAMES_ROOT")
TOTAL=${#CLIPS[@]}
echo "CVB inference — $TOTAL clips found under $FRAMES_ROOT"
echo "Output root : $OUTPUT_DIR"
echo ""

DONE=0
SKIPPED=0
FAILED=0
COUNT=0

for CLIP in "${CLIPS[@]}"; do
    if [[ $LIMIT -gt 0 && $COUNT -ge $LIMIT ]]; then
        echo "Reached --limit $LIMIT, stopping."
        break
    fi

    CLIP_DIR="$FRAMES_ROOT/$CLIP"
    OUT_CLIP="$OUTPUT_DIR/$CLIP"

    if [[ $RESUME -eq 1 && -d "$OUT_CLIP" && -f "$OUT_CLIP/activity_budget.csv" ]]; then
        echo "[$((COUNT+1))/$TOTAL] SKIP (already done): $CLIP"
        SKIPPED=$((SKIPPED+1))
        COUNT=$((COUNT+1))
        continue
    fi

    echo "[$((COUNT+1))/$TOTAL] Processing: $CLIP"

    if python src/inference/run_inference.py \
        --frames_dir "$CLIP_DIR" \
        --frames_fps 25.0 \
        --output_dir "$OUTPUT_DIR" \
        --config configs/inference/default.yaml \
        "${EXTRA_ARGS[@]}"; then
        DONE=$((DONE+1))
    else
        echo "  [WARN] Pipeline failed for $CLIP — continuing"
        FAILED=$((FAILED+1))
    fi

    COUNT=$((COUNT+1))
done

echo ""
echo "CVB inference complete."
echo "  Done   : $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed : $FAILED"
echo "  Total  : $COUNT / $TOTAL"
