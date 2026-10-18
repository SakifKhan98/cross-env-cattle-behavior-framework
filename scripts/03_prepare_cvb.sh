#!/bin/bash
# 03_prepare_cvb.sh
# Convert CVB AVA annotations to COCO detection format
# Run from project root: bash scripts/03_prepare_cvb.sh

set -e
echo "================================================"
echo "Step 3: Preparing CVB detection dataset"
echo "================================================"

python src/data/convert_cvb.py \
    --raw_dir data/raw/cvb \
    --out_dir data/processed/detection/cvb

echo "Done. Check data/processed/detection/cvb/"