#!/bin/bash
# 05_train_detector.sh
# Train RF-DETR cattle detector
# Run from project root: bash scripts/05_train_detector.sh [--sanity] [--resume]

set -e

CONFIG="configs/detection/rfdetr_combined.yaml"
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --sanity)  EXTRA_ARGS="$EXTRA_ARGS --sanity" ;;
        --resume)  EXTRA_ARGS="$EXTRA_ARGS --resume" ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash scripts/05_train_detector.sh [--sanity] [--resume]"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "Step 5: Training RF-DETR cattle detector"
echo "  Config : $CONFIG"
echo "  Args   : $EXTRA_ARGS"
echo "================================================"

python src/detection/train.py --config $CONFIG $EXTRA_ARGS