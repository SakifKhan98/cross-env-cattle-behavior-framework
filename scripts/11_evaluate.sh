#!/bin/bash
# 11_evaluate.sh
# Evaluate all 5 VideoMAE configs and write per-class F1, predictions, confusion matrices.
# Run from project root: bash scripts/11_evaluate.sh [CONFIG] [CHECKPOINT]
#
# Prerequisites:
#   - Step 9 complete: data/processed/tubelets/ with labels.csv
#   - Checkpoints in runs/behavior/ (train Step 10 or download from HuggingFace)
#
# Usage:
#   bash scripts/11_evaluate.sh                                       # all 5 configs
#   bash scripts/11_evaluate.sh configs/behavior/videomae_combined.yaml \
#        runs/behavior/videomae_combined_v1/checkpoint_best.pt

set -e

CONFIG_DIR="configs/behavior"
RUNS_DIR="runs/behavior"
RESULTS_DIR="results/behavior"

# Config → run directory mapping
declare -A RUN_MAP
RUN_MAP["videomae_cbvd5"]="videomae_cbvd5_v1"
RUN_MAP["videomae_cvb"]="videomae_cvb_v1"
RUN_MAP["videomae_combined"]="videomae_combined_v1"
RUN_MAP["videomae_cbvd5_to_cvb"]="videomae_cbvd5_to_cvb_v1"
RUN_MAP["videomae_cvb_to_cbvd5"]="videomae_cvb_to_cbvd5_v1"

if [ ! -f "data/processed/tubelets/labels.csv" ]; then
    echo "[ERROR] data/processed/tubelets/labels.csv not found."
    echo "  Run Step 9 first: bash scripts/09_generate_tubelets.sh"
    exit 1
fi

echo "========================================"
echo "Step 11: Evaluating VideoMAE configs"
echo "========================================"

if [ -n "$1" ] && [ -n "$2" ]; then
    echo "  Evaluating single config: $1"
    python src/behavior/evaluate.py --config "$1" --checkpoint "$2"
else
    for KEY in "${!RUN_MAP[@]}"; do
        CFG="$CONFIG_DIR/${KEY}.yaml"
        CKPT="$RUNS_DIR/${RUN_MAP[$KEY]}/checkpoint_best.pt"
        if [ ! -f "$CFG" ]; then
            echo "  [SKIP] Config not found: $CFG"
            continue
        fi
        if [ ! -f "$CKPT" ]; then
            echo "  [SKIP] Checkpoint not found: $CKPT"
            echo "         Download from HuggingFace (see docs/setup.md) or run Step 10."
            continue
        fi
        echo ""
        echo "  Evaluating $KEY ..."
        python src/behavior/evaluate.py --config "$CFG" --checkpoint "$CKPT"
    done
fi

echo ""
echo "========================================"
echo "Step 11 complete. Results written to $RESULTS_DIR/"
echo "  predictions/    confusion_matrices/    f1_per_class.csv"
echo "========================================"
