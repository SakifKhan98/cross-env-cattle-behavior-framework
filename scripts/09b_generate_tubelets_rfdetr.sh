#!/usr/bin/env bash
# scripts/09b_generate_tubelets_rfdetr.sh
# Phase 5 — Tubelet generation from RF-DETR-Seg tracking outputs
#
# Reads:  data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/*_tracks.json
#         data/raw/{cbvd5,cvb}/
# Writes: data/processed/tubelets_rfdetr/{cbvd5,cvb}/
#         data/processed/tubelets_rfdetr/labels.csv
#
# USAGE:
#   bash scripts/09b_generate_tubelets_rfdetr.sh              # both datasets
#   bash scripts/09b_generate_tubelets_rfdetr.sh --cbvd5_only
#   bash scripts/09b_generate_tubelets_rfdetr.sh --cvb_only
#   bash scripts/09b_generate_tubelets_rfdetr.sh --sanity     # 3 videos per dataset
#
# Runtime: several hours (CPU-only, I/O bound)
# Run from project root.

set -e

TRACKING_BASE="data/processed/tracking_v2_rfdetr"
OUTPUT_DIR="data/processed/tubelets_rfdetr"

# ── Parse args ────────────────────────────────────────────────────────────────
CVB_ONLY=0
SANITY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cvb_only)   CVB_ONLY=1; shift ;;
        --sanity)     SANITY=1;   shift ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Prereq checks ─────────────────────────────────────────────────────────────
if [ ! -d "$TRACKING_BASE/cbvd5" ] && [ ! -d "$TRACKING_BASE/cvb" ]; then
    echo "[ERROR] No tracking_v2_rfdetr outputs found in $TRACKING_BASE"
    echo "  Run Step 08b first: bash scripts/08b_run_tracking_rfdetr.sh"
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

echo "========================================"
echo "Step 09b: Generating tubelets (RF-DETR-Seg path)"
echo "  Input : $TRACKING_BASE"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

EXTRA_ARGS=""
if [ "$CVB_ONLY" -eq 1 ]; then
    EXTRA_ARGS="--cvb_only"
fi
if [ "$SANITY" -eq 1 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_cvb_videos 3 --max_cbvd5_videos 3"
fi

# Pre-create tracking dirs so os.listdir() returns [] instead of crashing
# when a dataset hasn't been run through 08b yet.
mkdir -p "$TRACKING_BASE/cvb" "$TRACKING_BASE/cbvd5"

python src/data/export_tubelets.py \
    --output "$OUTPUT_DIR" \
    --cvb_tracking "$TRACKING_BASE/cvb" \
    --cbvd5_tracking "$TRACKING_BASE/cbvd5" \
    $EXTRA_ARGS

echo ""
echo "========================================"
echo "Step 09b complete. Tubelets saved to $OUTPUT_DIR"
echo "  Verify: wc -l $OUTPUT_DIR/labels.csv"
echo "========================================"
