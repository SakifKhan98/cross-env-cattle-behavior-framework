#!/usr/bin/env bash
# scripts/12b_generate_analytics_rfdetr.sh
# Phase 7 — Analytics from RF-DETR-Seg pipeline predictions
#
# Reads:  results/behavior/predictions_rfdetr_tubelets/*.csv
#         data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/
# Writes: results/behavior_analytics_rfdetr/timelines/
#         results/behavior_analytics_rfdetr/activity_budget.csv
#         results/behavior_analytics_rfdetr/transition_matrix.csv
#         results/behavior_analytics_rfdetr/behavior_deviation.csv
#
# USAGE:
#   bash scripts/12b_generate_analytics_rfdetr.sh
#
# Run from project root.

set -e

PREDICTIONS_DIR="results/behavior/predictions_rfdetr_tubelets"
TRACKING_DIR="data/processed/tracking_v2_rfdetr"
ANALYTICS_DIR="results/behavior_analytics_rfdetr"
# Default to combined v2 for analytics; override with --run <name> if needed
RUN_NAME="videomae_combined_v2"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN_NAME="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Prereq checks ─────────────────────────────────────────────────────────────
if [ ! -d "$PREDICTIONS_DIR" ] || [ -z "$(ls -A "$PREDICTIONS_DIR" 2>/dev/null)" ]; then
    echo "[ERROR] No prediction CSVs found in $PREDICTIONS_DIR"
    echo "  Run Step 11b first: bash scripts/11b_evaluate_rfdetr.sh"
    exit 1
fi

if [ ! -d "$TRACKING_DIR/cbvd5" ] && [ ! -d "$TRACKING_DIR/cvb" ]; then
    echo "[ERROR] No tracking_v2_rfdetr outputs found in $TRACKING_DIR"
    echo "  Run Step 08b first: bash scripts/08b_run_tracking_rfdetr.sh"
    exit 1
fi

# ── Conda activation ──────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cattletransformer

mkdir -p "$ANALYTICS_DIR/timelines"

echo "========================================"
echo "Step 12b: Generating analytics (RF-DETR-Seg path)"
echo "  Predictions : $PREDICTIONS_DIR"
echo "  Tracking    : $TRACKING_DIR"
echo "  Output      : $ANALYTICS_DIR"
echo "========================================"

echo ""
echo "  Building per-animal timelines (run: $RUN_NAME) ..."
python -m src.analytics.timeline \
    --predictions_dir "$PREDICTIONS_DIR" \
    --tracking_dir "$TRACKING_DIR" \
    --run "$RUN_NAME" \
    --out_dir "$ANALYTICS_DIR/timelines"

echo ""
echo "  Computing activity budget, transitions, and behavioral deviation ..."
python -m src.analytics.budget \
    --timelines_dir "$ANALYTICS_DIR/timelines" \
    --out_dir "$ANALYTICS_DIR"

echo ""
echo "========================================"
echo "Step 12b complete."
echo "  $ANALYTICS_DIR/activity_budget.csv"
echo "  $ANALYTICS_DIR/transition_matrix.csv"
echo "  $ANALYTICS_DIR/behavior_deviation.csv"
echo "========================================"
