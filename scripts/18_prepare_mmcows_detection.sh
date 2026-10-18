#!/bin/bash
# 18_prepare_mmcows_detection.sh
# Convert mmCows cam_4 YOLO annotations to COCO format for OOD detection eval.
#
# Run from project root: bash scripts/18_prepare_mmcows_detection.sh
#
# Inputs:  data/raw/mmCows/labels/combined/0725/cam_4/  (5040 .txt files)
#          data/raw/mmCows/images/0725/cam_4/           (5040 .jpg files)
# Output:  data/processed/detection/mmcows/_annotations.coco.json
#          data/processed/detection/mmcows/*.jpg         (symlinks)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer

cd "$REPO_ROOT"

python src/data/convert_mmcows_detection.py \
    --raw_dir data/raw/mmCows \
    --out_dir data/processed/detection/mmcows \
    --date    0725 \
    --cam     cam_4

echo ""
echo "COCO dataset written to data/processed/detection/mmcows/"
echo "Run scripts/19_eval_mmcows_detection.sh to compute mAP@50."
