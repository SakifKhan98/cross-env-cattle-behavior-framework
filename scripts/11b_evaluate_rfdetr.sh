#!/usr/bin/env bash
# scripts/11b_evaluate_rfdetr.sh
# Phase 6 — Evaluate VideoMAE v2 checkpoints on RF-DETR-Seg tubelets
#
# Reads:  data/processed/tubelets_rfdetr/labels.csv
#         runs/behavior/videomae_*_v2/checkpoint_best.pt
# Writes: results/behavior/predictions_sam2_tubelets/videomae_*_v2_val.csv
#         results/behavior/predictions_rfdetr_tubelets/               (copies moved here)
#         results/behavior/confusion_matrices/*_v2_*.png
#         results/behavior/f1_per_class.csv                  (v2 rows appended)
#
# USAGE:
#   bash scripts/11b_evaluate_rfdetr.sh                                # all 5 v2 configs
#   bash scripts/11b_evaluate_rfdetr.sh configs/behavior/videomae_combined_v2.yaml \
#        runs/behavior/videomae_combined_v2/checkpoint_best.pt
#
# Run from project root.

set -e

CONFIG_DIR="configs/behavior"
RUNS_DIR="runs/behavior"
TUBELETS="data/processed/tubelets_rfdetr"
PREDS_RFDETR="results/behavior/predictions_rfdetr_tubelets"

# Config basename → v2 run directory name
declare -A RUN_MAP
RUN_MAP["videomae_cbvd5_v2"]="videomae_cbvd5_v2"
RUN_MAP["videomae_cvb_v2"]="videomae_cvb_v2"
RUN_MAP["videomae_combined_v2"]="videomae_combined_v2"
RUN_MAP["videomae_cbvd5_to_cvb_v2"]="videomae_cbvd5_to_cvb_v2"
RUN_MAP["videomae_cvb_to_cbvd5_v2"]="videomae_cvb_to_cbvd5_v2"

# ── Prereq check ──────────────────────────────────────────────────────────────
if [ ! -f "$TUBELETS/labels.csv" ]; then
    echo "[ERROR] $TUBELETS/labels.csv not found."
    echo "  Run Step 09b first: bash scripts/09b_generate_tubelets_rfdetr.sh"
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

mkdir -p "$PREDS_RFDETR"

echo "========================================"
echo "Step 11b: Evaluating VideoMAE v2 configs"
echo "========================================"

# ── Single config mode ────────────────────────────────────────────────────────
if [ -n "$1" ] && [ -n "$2" ]; then
    echo "  Evaluating single config: $1"
    python src/behavior/evaluate.py --config "$1" --checkpoint "$2"

    # Move the resulting prediction CSV into predictions_rfdetr_tubelets/
    EXP=$(python -c "import yaml; d=yaml.safe_load(open('$1')); print(d.get('experiment_name', '$1'.split('/')[-1].replace('.yaml','')))")
    for F in results/behavior/predictions_sam2_tubelets/${EXP}_*.csv; do
        [ -f "$F" ] && cp "$F" "$PREDS_RFDETR/" && echo "  Copied $F → $PREDS_RFDETR/"
    done
    echo "========================================"
    echo "Step 11b complete."
    echo "========================================"
    exit 0
fi

# ── All 5 v2 configs ──────────────────────────────────────────────────────────
for KEY in "${!RUN_MAP[@]}"; do
    CFG="$CONFIG_DIR/${KEY}.yaml"
    CKPT="$RUNS_DIR/${RUN_MAP[$KEY]}/checkpoint_best.pt"

    if [ ! -f "$CFG" ]; then
        echo "  [SKIP] Config not found: $CFG"
        continue
    fi
    if [ ! -f "$CKPT" ]; then
        echo "  [SKIP] Checkpoint not found: $CKPT"
        echo "         Run Step 10b or fetch from HiPE1:"
        echo "         rsync -avz hipe1:~/cattle_behavior/runs/behavior/ runs/behavior/"
        continue
    fi

    echo ""
    echo "  Evaluating $KEY ..."
    python src/behavior/evaluate.py --config "$CFG" --checkpoint "$CKPT"

    # Copy v2 prediction CSV into predictions_rfdetr_tubelets/ for clean analytics separation
    for F in results/behavior/predictions_sam2_tubelets/${KEY}_*.csv; do
        [ -f "$F" ] && cp "$F" "$PREDS_RFDETR/" && echo "  Copied $(basename $F) → $PREDS_RFDETR/"
    done
done

echo ""
echo "========================================"
echo "Step 11b complete."
echo "  Predictions: $PREDS_RFDETR/"
echo "  F1 summary : results/behavior/f1_per_class.csv  (v2 rows appended)"
echo "========================================"
