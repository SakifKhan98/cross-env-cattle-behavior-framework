#!/bin/bash
# 13_prepare_freeman.sh
# Convert Freeman Center (CMB 2024) YOLO annotations to COCO detection format.
# Run from project root: bash scripts/13_prepare_freeman.sh
#
# Input:  data/raw/freeman-cmb-2024/CMB_dataset/CMB_dataset/{train,val,test}/
# Output: data/processed/detection/freeman/{train,valid,test}/_annotations.coco.json

set -e

RAW_DIR="data/raw/freeman-cmb-2024"
OUT_DIR="data/processed/detection/freeman"

if [ ! -d "$RAW_DIR/CMB_dataset/CMB_dataset" ]; then
    echo "[ERROR] Freeman Center dataset not found: $RAW_DIR/CMB_dataset/CMB_dataset"
    echo "  Download from: https://universe.roboflow.com/..."
    exit 1
fi

echo "================================================"
echo "Step 22: Preparing Freeman Center dataset"
echo "  Input : $RAW_DIR"
echo "  Output: $OUT_DIR"
echo "================================================"

python src/data/convert_freeman.py \
    --raw_dir "$RAW_DIR" \
    --out_dir "$OUT_DIR"

echo "Done. COCO annotations saved to $OUT_DIR"
