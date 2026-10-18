#!/bin/bash
# 04_merge_datasets.sh
# Merge CBVD-5 and CVB into combined dataset, then create mini subset
# Run from project root: bash scripts/04_merge_datasets.sh

set -e
echo "================================================"
echo "Step 4: Merging datasets + creating mini subset"
echo "================================================"

python src/data/merge_coco.py \
    --det_dir data/processed/detection \
    --out_dir data/processed/detection/combined

python src/data/make_mini.py \
    --src_dir data/processed/detection/combined \
    --out_dir data/processed/detection/combined_mini \
    --n_images 50

echo "Done. Check data/processed/detection/combined/"
echo "      Mini dataset: data/processed/detection/combined_mini/"