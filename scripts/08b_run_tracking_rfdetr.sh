#!/usr/bin/env bash
# scripts/08b_run_tracking_rfdetr.sh
# Phase 4 — OC-SORT Tracking on RF-DETR-Seg outputs
#
# Reads:  data/processed/segmentation_rfdetr/{cbvd5,cvb}/*_masks.json
# Writes: data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/*_tracks.json
#
# USAGE:
#   bash scripts/08b_run_tracking_rfdetr.sh                  # both datasets
#   bash scripts/08b_run_tracking_rfdetr.sh --sanity          # 3 videos per dataset
#   bash scripts/08b_run_tracking_rfdetr.sh --dataset cbvd5
#   bash scripts/08b_run_tracking_rfdetr.sh --dataset cvb
#   bash scripts/08b_run_tracking_rfdetr.sh --video_id 618    # single video
#
# Run from project root:
#   bash scripts/08b_run_tracking_rfdetr.sh

set -e

SEG_BASE="data/processed/segmentation_rfdetr"
OUT_BASE="data/processed/tracking_v2_rfdetr"
OCSORT_ROOT="third_party/OC_SORT"

# ── Parse args ────────────────────────────────────────────────────────────────
DATASET=""
VIDEO_ID=""
SANITY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2";   shift 2 ;;
        --video_id)  VIDEO_ID="$2";  shift 2 ;;
        --sanity)    SANITY=1;        shift   ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Prereq checks ─────────────────────────────────────────────────────────────
if [ ! -d "$OCSORT_ROOT" ]; then
    echo "[ERROR] $OCSORT_ROOT not found."
    echo "  Run: git clone https://github.com/noahcao/OC_SORT.git $OCSORT_ROOT"
    exit 1
fi

if [ ! -d "$SEG_BASE/cbvd5" ] && [ ! -d "$SEG_BASE/cvb" ]; then
    echo "[ERROR] No RF-DETR-Seg outputs found in $SEG_BASE"
    echo "  Run Step 07b first: bash scripts/07b_run_rfdetr_seg.sh"
    exit 1
fi

# ── Conda activation ──────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cattletransformer

# ── Dataset list ──────────────────────────────────────────────────────────────
if [ -n "$DATASET" ]; then
    DATASETS=("$DATASET")
else
    DATASETS=(cbvd5 cvb)
fi

echo "========================================"
echo "Step 08b: RF-DETR-Seg OC-SORT Tracking"
echo "  Seg input : $SEG_BASE"
echo "  Output    : $OUT_BASE"
echo "========================================"

for DS in "${DATASETS[@]}"; do
    if [ ! -d "$SEG_BASE/$DS" ]; then
        echo "  [SKIP] $DS — no segmentation output found in $SEG_BASE/$DS"
        continue
    fi

    echo ""
    echo "  Tracking $DS ..."

    EXTRA_ARGS=""
    if [ -n "$VIDEO_ID" ]; then
        EXTRA_ARGS="--video_id $VIDEO_ID"
    fi
    if [ "$SANITY" -eq 1 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --limit 3"
    fi

    python src/tracking/track.py \
        --dataset "$DS" \
        --seg_dir "$SEG_BASE/$DS" \
        --output_dir "$OUT_BASE/$DS" \
        $EXTRA_ARGS
done

echo ""
echo "========================================"
echo "Step 08b complete. Tracking JSONs saved to $OUT_BASE"
echo "  Verify: ls $OUT_BASE/cbvd5/ | head -5"
echo "========================================"
