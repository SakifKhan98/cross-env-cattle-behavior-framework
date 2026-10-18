#!/usr/bin/env bash
# scripts/07b_run_rfdetr_seg.sh
# Phase 3 — RF-DETR-Seg Instance Segmentation (replaces SAM2)
#
# Runs the fine-tuned RF-DETR-Seg model (Config B EMA, epoch 59) on
# raw video frames to produce _masks.json for downstream tracking.
#
# USAGE:
#   bash scripts/07b_run_rfdetr_seg.sh                        # Both datasets
#   bash scripts/07b_run_rfdetr_seg.sh --sanity               # Quick test, 3 videos each
#   bash scripts/07b_run_rfdetr_seg.sh --dataset cbvd5        # CBVD-5 only
#   bash scripts/07b_run_rfdetr_seg.sh --dataset cvb          # CVB only
#   bash scripts/07b_run_rfdetr_seg.sh --video_id 618         # Single video
#
# Run from the project root:
#   cd ~/TXST/Thesis/cattle-vision-framework
#   bash scripts/07b_run_rfdetr_seg.sh --sanity

set -e

PYTHON_ARGS="$@"

echo "[07b] Activating conda environment: cattletransformer"

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cattletransformer

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[07b] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ ! -f "configs/segmentation/rfdetr_seg.yaml" ]; then
    echo "[ERROR] Config not found: configs/segmentation/rfdetr_seg.yaml"
    exit 1
fi

echo "[07b] Checking RF-DETR installation..."
python3 -c "from rfdetr import RFDETRSegMedium; print('[07b] RF-DETR OK')" || {
    echo "[ERROR] rfdetr is not installed."
    echo "  Fix: pip install rfdetr"
    exit 1
}

python3 -c "from pycocotools import mask; print('[07b] pycocotools OK')" || {
    echo "[ERROR] pycocotools is not installed."
    echo "  Fix: pip install pycocotools"
    exit 1
}

CHECKPOINT="runs/seg_medium_lr5e5/checkpoint_best_ema.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "[07b] Checkpoint: $CHECKPOINT"

mkdir -p data/processed/segmentation_rfdetr/cbvd5
mkdir -p data/processed/segmentation_rfdetr/cvb
echo "[07b] Output directories ready."

echo "[07b] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

echo ""
echo "[07b] Starting Phase 3 — RF-DETR-Seg Segmentation"
echo "[07b] Args: $PYTHON_ARGS"
echo ""

python3 src/segmentation/rfdetr_seg_infer.py \
    --config configs/segmentation/rfdetr_seg.yaml \
    $PYTHON_ARGS

echo ""
echo "[07b] Phase 3 complete."
