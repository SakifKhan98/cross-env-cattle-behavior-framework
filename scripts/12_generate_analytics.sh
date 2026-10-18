#!/bin/bash
# 12_generate_analytics.sh
# Generate activity timelines, budget, transition matrix, and behavioral deviation from predictions.
# Run from project root: bash scripts/12_generate_analytics.sh
#
# Prerequisites:
#   - Step 11 complete: results/behavior/predictions_sam2_tubelets/*.csv
#   - data/processed/tracking_v2/{cbvd5,cvb}/ present (for timeline frame alignment)
#
# Output (CPU-only, fast):
#   results/behavior_analytics/timelines/          (gitignored — large per-video CSVs)
#   results/behavior_analytics/activity_budget.csv
#   results/behavior_analytics/transition_matrix.csv
#   results/behavior_analytics/behavior_deviation.csv

set -e

PREDICTIONS_DIR="results/behavior/predictions_sam2_tubelets"
TRACKING_DIR="data/processed/tracking_v2"
ANALYTICS_DIR="results/behavior_analytics"

if [ ! -d "$PREDICTIONS_DIR" ] || [ -z "$(ls -A $PREDICTIONS_DIR 2>/dev/null)" ]; then
    echo "[ERROR] No prediction CSVs found in $PREDICTIONS_DIR"
    echo "  Run Step 11 first: bash scripts/11_evaluate.sh"
    exit 1
fi

mkdir -p "$ANALYTICS_DIR/timelines"

echo "========================================"
echo "Step 12: Generating analytics"
echo "  Predictions : $PREDICTIONS_DIR"
echo "  Tracking    : $TRACKING_DIR"
echo "  Output      : $ANALYTICS_DIR"
echo "========================================"

echo ""
echo "  Building per-animal timelines ..."
python -m src.analytics.timeline \
    --predictions_dir "$PREDICTIONS_DIR" \
    --tracking_dir "$TRACKING_DIR" \
    --out_dir "$ANALYTICS_DIR/timelines"

echo ""
echo "  Computing activity budget, transitions, and behavioral deviation ..."
python -m src.analytics.budget \
    --timelines_dir "$ANALYTICS_DIR/timelines" \
    --out_dir "$ANALYTICS_DIR"

echo ""
echo "========================================"
echo "Step 12 complete."
echo "  Committed outputs:"
echo "    $ANALYTICS_DIR/activity_budget.csv"
echo "    $ANALYTICS_DIR/transition_matrix.csv"
echo "    $ANALYTICS_DIR/behavior_deviation.csv"
echo "========================================"
