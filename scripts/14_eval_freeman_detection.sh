#!/bin/bash
# 14_eval_freeman_detection.sh
# Evaluate RF-DETR detector (OOD) on Freeman Center test split.
# Run from project root: bash scripts/14_eval_freeman_detection.sh
#
# Prerequisite: bash scripts/13_prepare_freeman.sh must have been run.
#
# Output: results/detection/freeman_detection_eval.json

set -e

CHECKPOINT="runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth"
DATASET_DIR="data/processed/detection/freeman/test"
OUTPUT="results/detection/freeman_detection_eval.json"
THRESHOLD=0.3

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$DATASET_DIR/_annotations.coco.json" ]; then
    echo "[ERROR] Freeman Center test split not found."
    echo "  Run Step 13 first: bash scripts/13_prepare_freeman.sh"
    exit 1
fi

echo "================================================"
echo "Step 14: Freeman Center detection evaluation"
echo "  Checkpoint : $CHECKPOINT"
echo "  Dataset    : $DATASET_DIR"
echo "  Threshold  : $THRESHOLD"
echo "  Output     : $OUTPUT"
echo "================================================"

python src/tools/eval_detection_ood.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --output "$OUTPUT" \
    --dataset_name freeman \
    --threshold $THRESHOLD

echo "Done. Results saved to $OUTPUT"
