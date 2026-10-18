#!/bin/bash
# 10_train_behavior.sh
# Train VideoMAE behavior classifier for one or all configs.
# Run from project root: bash scripts/10_train_behavior.sh [CONFIG]
#
# Prerequisites:
#   - Step 9 complete: data/processed/tubelets/ with labels.csv
#   - GPU with ≥12 GB VRAM (RTX 3060 local or HiPE1 V100)
#   - Docker (for HiPE1 path) or conda env (local path)
#
# Usage:
#   bash scripts/10_train_behavior.sh                          # train all 5 configs
#   bash scripts/10_train_behavior.sh configs/behavior/videomae_combined.yaml
#
# NOTE: Download pretrained checkpoints from HuggingFace to skip training.
#       See docs/setup.md for huggingface-cli download commands.

set -e

TUBELETS="data/processed/tubelets"
CONFIG_DIR="configs/behavior"

if [ ! -f "$TUBELETS/labels.csv" ]; then
    echo "[ERROR] $TUBELETS/labels.csv not found."
    echo "  Run Step 9 first: bash scripts/09_generate_tubelets.sh"
    exit 1
fi

if [ -n "$1" ]; then
    CONFIGS=("$1")
else
    CONFIGS=(
        "$CONFIG_DIR/videomae_cbvd5.yaml"
        "$CONFIG_DIR/videomae_cvb.yaml"
        "$CONFIG_DIR/videomae_combined.yaml"
        "$CONFIG_DIR/videomae_cbvd5_to_cvb.yaml"
        "$CONFIG_DIR/videomae_cvb_to_cbvd5.yaml"
    )
fi

echo "========================================"
echo "Step 10: Training VideoMAE behavior classifier"
echo "  Configs: ${#CONFIGS[@]}"
echo "========================================"

for CFG in "${CONFIGS[@]}"; do
    if [ ! -f "$CFG" ]; then
        echo "[ERROR] Config not found: $CFG"
        exit 1
    fi
    echo ""
    echo "  Training with $CFG ..."
    python src/behavior/train.py --config "$CFG"
done

echo ""
echo "========================================"
echo "Step 10 complete. Checkpoints saved to runs/behavior/"
echo "========================================"
