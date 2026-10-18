#!/bin/bash
# 02_prepare_cbvd5.sh
# Convert CBVD-5 AVA annotations to COCO detection format
# Run from project root: bash scripts/02_prepare_cbvd5.sh

set -e
echo "================================================"
echo "Step 2: Preparing CBVD-5 detection dataset"
echo "================================================"

python src/data/convert_cbvd5.py \
    --raw_dir data/raw/cbvd5 \
    --out_dir data/processed/detection/cbvd5

echo "Done. Check data/processed/detection/cbvd5/"