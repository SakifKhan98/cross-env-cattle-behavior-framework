#!/bin/bash
# 06_run_detection.sh
# Run trained detector on all videos from both datasets
# Run from project root: bash scripts/06_run_detection.sh
#
# Prerequisite: Step 5 must be complete and checkpoint must exist at:
#   runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth

set -e

CHECKPOINT="runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth"
OUT_DIR="data/processed/tracking"
CONF_THRESH=0.5

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "  Run Step 5 first: bash scripts/05_train_detector.sh"
    exit 1
fi

echo "================================================"
echo "Step 6: Running detection on all dataset videos"
echo "  Checkpoint : $CHECKPOINT"
echo "  Output     : $OUT_DIR"
echo "  Conf thresh: $CONF_THRESH"
echo "================================================"

python src/detection/infer_dataset.py \
    --checkpoint $CHECKPOINT \
    --dataset both \
    --out_dir $OUT_DIR \
    --conf_thresh $CONF_THRESH

echo "Done. Detection JSONs saved to $OUT_DIR"