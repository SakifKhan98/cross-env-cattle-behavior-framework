#!/usr/bin/env bash
# scripts/10b_train_behavior_rfdetr.sh
# Phase 6 — VideoMAE behavior training on RF-DETR-Seg tubelets (v2 configs)
#
# Two modes:
#   Local  — conda env on RTX 3060 (smoke test only, batch=4)
#   HiPE1  — rsync tubelets + src to HiPE1, then Docker run for each v2 config
#
# USAGE:
#   bash scripts/10b_train_behavior_rfdetr.sh                            # local, all 5 configs
#   bash scripts/10b_train_behavior_rfdetr.sh configs/behavior/videomae_combined_v2.yaml
#   bash scripts/10b_train_behavior_rfdetr.sh --hipe1                    # HiPE1, all 5 configs
#   bash scripts/10b_train_behavior_rfdetr.sh --hipe1 configs/behavior/videomae_combined_v2.yaml
#
# Prerequisites:
#   - Step 09b complete: data/processed/tubelets_rfdetr/labels.csv
#   - HiPE1 mode: cattle-behavior Docker image loaded on HiPE1 (see docs/hipe_ops.md §4)
#                 SSH alias 'hipe1' configured in ~/.ssh/config
#
# Run from project root.

set -e

CONFIG_DIR="configs/behavior"
TUBELETS="data/processed/tubelets_rfdetr"
HIPE1_HOME="~/cattle_behavior"

V2_CONFIGS=(
    "$CONFIG_DIR/videomae_cbvd5_v2.yaml"
    "$CONFIG_DIR/videomae_cvb_v2.yaml"
    "$CONFIG_DIR/videomae_combined_v2.yaml"
    "$CONFIG_DIR/videomae_cbvd5_to_cvb_v2.yaml"
    "$CONFIG_DIR/videomae_cvb_to_cbvd5_v2.yaml"
)

# ── Parse args ────────────────────────────────────────────────────────────────
HIPE1_MODE=0
SINGLE_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hipe1) HIPE1_MODE=1; shift ;;
        *.yaml)  SINGLE_CONFIG="$1"; shift ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -n "$SINGLE_CONFIG" ]; then
    CONFIGS=("$SINGLE_CONFIG")
else
    CONFIGS=("${V2_CONFIGS[@]}")
fi

# ── Prereq check ──────────────────────────────────────────────────────────────
if [ ! -f "$TUBELETS/labels.csv" ]; then
    echo "[ERROR] $TUBELETS/labels.csv not found."
    echo "  Run Step 09b first: bash scripts/09b_generate_tubelets_rfdetr.sh"
    exit 1
fi

for CFG in "${CONFIGS[@]}"; do
    if [ ! -f "$CFG" ]; then
        echo "[ERROR] Config not found: $CFG"
        exit 1
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# HiPE1 MODE
# ══════════════════════════════════════════════════════════════════════════════
if [ "$HIPE1_MODE" -eq 1 ]; then
    echo "========================================"
    echo "Step 10b: VideoMAE training — HiPE1 mode"
    echo "  Configs: ${#CONFIGS[@]}"
    echo "  Remote : hipe1:$HIPE1_HOME"
    echo "========================================"

    echo ""
    echo "  [1/3] Syncing tubelets to HiPE1 ..."
    ssh hipe1 "mkdir -p $HIPE1_HOME/data/processed"
    rsync -avz --progress \
        "$TUBELETS/" \
        "hipe1:$HIPE1_HOME/data/processed/tubelets_rfdetr/"

    echo ""
    echo "  [2/3] Syncing src/ and configs/ to HiPE1 ..."
    rsync -avz --progress \
        src/ configs/ \
        "hipe1:$HIPE1_HOME/"

    echo ""
    echo "  [3/3] Launching Docker training runs on HiPE1 ..."
    for CFG in "${CONFIGS[@]}"; do
        echo ""
        echo "  Training: $CFG"
        # Run in background on HiPE1 — each config is independent
        ssh hipe1 "cd $HIPE1_HOME && docker run --rm --gpus all --shm-size=16g \
            -v \$(pwd):/workspace \
            cattle-behavior \
            python src/behavior/train.py --config $CFG"
        echo "  Done: $CFG"
    done

    echo ""
    echo "========================================"
    echo "Step 10b complete (HiPE1)."
    echo "  Fetch checkpoints when done:"
    echo "  rsync -avz hipe1:$HIPE1_HOME/runs/behavior/ runs/behavior/"
    echo "========================================"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# LOCAL MODE
# ══════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "Step 10b: VideoMAE training — local mode"
echo "  Configs: ${#CONFIGS[@]}"
echo "  NOTE: RTX 3060 (12 GB) — for smoke tests only"
echo "========================================"

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cattletransformer

for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "  Training: $CFG"
    python src/behavior/train.py --config "$CFG"
    echo "  Done: $CFG"
done

echo ""
echo "========================================"
echo "Step 10b complete (local). Checkpoints in runs/behavior/"
echo "========================================"
