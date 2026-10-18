#!/bin/bash
# 17_infer_cbvd5.sh
# Run the Phase 9 inference pipeline on all CBVD-5 videos.
# Each .mp4 under data/raw/cbvd5/videos/videos/ is processed independently.
#
# Run from project root: bash scripts/17_infer_cbvd5.sh
#
# Optional flags:
#   --output_dir  <path>    Root output directory (default: results/inference/cbvd5)
#   --conf_thresh <float>   Detection confidence threshold (overrides config)
#   --cleanup               Delete intermediate files after each run
#   --resume                Skip videos whose output directory already exists
#   --limit       <int>     Process at most N videos (useful for a test run)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer

cd "$REPO_ROOT"

VIDEOS_ROOT="data/raw/cbvd5/videos/videos"
OUTPUT_DIR="results/inference/cbvd5"
EXTRA_ARGS=()
RESUME=0
LIMIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --conf_thresh) EXTRA_ARGS+=(--conf_thresh "$2"); shift 2 ;;
        --cleanup)     EXTRA_ARGS+=(--cleanup); shift ;;
        --resume)      RESUME=1; shift ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mapfile -t VIDEOS < <(ls -1 "$VIDEOS_ROOT"/*.mp4 2>/dev/null | sort)
TOTAL=${#VIDEOS[@]}
echo "CBVD-5 inference — $TOTAL videos found under $VIDEOS_ROOT"
echo "Output root : $OUTPUT_DIR"
echo ""

DONE=0
SKIPPED=0
FAILED=0
COUNT=0

for VIDEO in "${VIDEOS[@]}"; do
    if [[ $LIMIT -gt 0 && $COUNT -ge $LIMIT ]]; then
        echo "Reached --limit $LIMIT, stopping."
        break
    fi

    STEM="$(basename "$VIDEO" .mp4)"
    OUT_DIR="$OUTPUT_DIR/$STEM"

    if [[ $RESUME -eq 1 && -d "$OUT_DIR" && -f "$OUT_DIR/activity_budget.csv" ]]; then
        echo "[$((COUNT+1))/$TOTAL] SKIP (already done): $STEM"
        SKIPPED=$((SKIPPED+1))
        COUNT=$((COUNT+1))
        continue
    fi

    echo "[$((COUNT+1))/$TOTAL] Processing: $STEM"

    if python src/inference/run_inference.py \
        --video "$VIDEO" \
        --output_dir "$OUTPUT_DIR" \
        --config configs/inference/default.yaml \
        "${EXTRA_ARGS[@]}"; then
        DONE=$((DONE+1))
    else
        echo "  [WARN] Pipeline failed for $STEM — continuing"
        FAILED=$((FAILED+1))
    fi

    COUNT=$((COUNT+1))
done

echo ""
echo "CBVD-5 inference complete."
echo "  Done   : $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed : $FAILED"
echo "  Total  : $COUNT / $TOTAL"
