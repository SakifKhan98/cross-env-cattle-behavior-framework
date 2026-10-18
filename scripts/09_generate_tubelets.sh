#!/bin/bash
# 09_generate_tubelets.sh
# Extract fixed-length tubelet clips from tracking_v2 outputs for behavior classification.
# Run from project root: bash scripts/09_generate_tubelets.sh
#
# Prerequisites:
#   - Step 7 or 8 complete: data/processed/tracking_v2/{cbvd5,cvb}/*_tracks.json
#   - Raw videos/frames present: data/raw/{cbvd5,cvb}/
#
# Output:
#   data/processed/tubelets/{cbvd5,cvb}/   (125,586 clips)
#   data/processed/tubelets/labels.csv
#
# Runtime: several hours (CPU-only, I/O bound)

set -e

TRACKING_DIR="data/processed/tracking_v2"
OUTPUT_DIR="data/processed/tubelets"

if [ ! -d "$TRACKING_DIR/cbvd5" ] && [ ! -d "$TRACKING_DIR/cvb" ]; then
    echo "[ERROR] No tracking_v2 outputs found in $TRACKING_DIR"
    echo "  Run Step 7 (segmentation) or Step 8 (box tracking) first."
    echo "  Or download pre-computed tracking_v2 from HuggingFace (see docs/setup.md)."
    exit 1
fi

echo "========================================"
echo "Step 9: Generating tubelets"
echo "  Input : $TRACKING_DIR"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

python src/data/export_tubelets.py \
    --output "$OUTPUT_DIR" \
    --cvb_tracking "$TRACKING_DIR/cvb" \
    --cbvd5_tracking "$TRACKING_DIR/cbvd5"

echo ""
echo "========================================"
echo "Step 9 complete. Tubelets saved to $OUTPUT_DIR"
echo "  Verify: wc -l $OUTPUT_DIR/labels.csv"
echo "========================================"
