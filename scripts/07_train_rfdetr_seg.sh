#!/bin/bash
# scripts/07_train_rfdetr_seg.sh
# Phase 3b — Train RF-DETR-Seg on SAM2 pseudo-labels
#
# NOTE: This script is provided for local reproduction.
# The actual thesis training ran on HiPE1 via scripts/hipe/train_rfdetr_seg_medium.py;
# the resulting checkpoint is at runs/seg_medium_lr5e5/checkpoint_best_ema.pth.
#
# WHAT THIS DOES:
#   1. Converts SAM2 mask JSONs → COCO segmentation dataset (if not already done)
#   2. Fine-tunes RF-DETR-Seg-Medium starting from COCO pretrained weights
#   3. Saves checkpoints to runs/segmentation/rfdetr_seg_cattle_v1/
#
# USAGE:
#   bash scripts/07_train_rfdetr_seg.sh              # full run, both datasets
#   bash scripts/07_train_rfdetr_seg.sh --cbvd5_only # CBVD-5 only
#   bash scripts/07_train_rfdetr_seg.sh --skip_conversion # if dataset already built

set -e
cd "$(dirname "$0")/.."

echo "[07] Starting Phase 3b — RF-DETR-Seg Training"
echo "[07] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR="data/rfdetr_seg/cattle"
OUTPUT_DIR="runs/segmentation/rfdetr_seg_cattle_v1"
EPOCHS=100
BATCH_SIZE=2
GRAD_ACCUM=8        # effective batch = 16
LR="1e-4"
CVB_STRIDE=15       # one frame per SAM2 re-prompt window
VAL_RATIO=0.2
SKIP_CONVERSION=false
CBVD5_ONLY=false

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --skip_conversion) SKIP_CONVERSION=true ;;
        --cbvd5_only)      CBVD5_ONLY=true ;;
        --epochs=*)        EPOCHS="${arg#*=}" ;;
        --lr=*)            LR="${arg#*=}" ;;
    esac
done

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "[07] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── Step 1: Convert SAM2 masks → COCO dataset ─────────────────────────────────
if [ "$SKIP_CONVERSION" = false ]; then
    echo ""
    echo "[07] Step 1: Converting SAM2 masks → COCO segmentation dataset"
    echo "[07] CVB stride: $CVB_STRIDE (one frame per re-prompt window)"

    EXTRA_ARGS=""
    if [ "$CBVD5_ONLY" = true ]; then
        EXTRA_ARGS="--cbvd5_only"
        echo "[07] Mode: CBVD-5 only"
    fi

    python src/tools/sam2_to_coco_seg.py \
        --output_dir "$DATASET_DIR" \
        --cvb_stride "$CVB_STRIDE" \
        --val_ratio  "$VAL_RATIO" \
        --min_area   100 \
        $EXTRA_ARGS

    echo "[07] Dataset conversion complete."
else
    echo "[07] Skipping conversion (--skip_conversion set)"
    if [ ! -d "$DATASET_DIR/train" ]; then
        echo "[ERROR] Dataset not found at $DATASET_DIR — remove --skip_conversion"
        exit 1
    fi
fi

# ── Check dataset stats ───────────────────────────────────────────────────────
echo ""
echo "[07] Dataset summary:"
python3 -c "
import json
for split in ['train', 'valid']:
    path = '$DATASET_DIR/' + split + '/_annotations.coco.json'
    try:
        d = json.load(open(path))
        n_img = len(d['images'])
        n_ann = len(d['annotations'])
        print(f'  {split:5s}: {n_img:6,} images, {n_ann:7,} annotations ({n_ann/n_img:.1f} avg/image)')
    except FileNotFoundError:
        print(f'  {split:5s}: not found')
"

# ── Step 2: Train RF-DETR-Seg ─────────────────────────────────────────────────
echo ""
echo "[07] Step 2: Fine-tuning RF-DETR-Seg-Medium"
echo "[07] Epochs: $EPOCHS | Batch: $BATCH_SIZE | Grad accum: $GRAD_ACCUM | LR: $LR"
echo "[07] Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

python3 - << PYEOF
from rfdetr import RFDETRSegMedium
import os

model = RFDETRSegMedium()

model.train(
    dataset_dir="$DATASET_DIR",
    epochs=$EPOCHS,
    batch_size=$BATCH_SIZE,
    grad_accum_steps=$GRAD_ACCUM,
    lr=$LR,
    output_dir="$OUTPUT_DIR",
    use_ema=True,
    gradient_checkpointing=True,   # ~30% VRAM reduction
)

print("[07] Training complete.")
print(f"[07] Best weights: $OUTPUT_DIR/checkpoint_best_total.pth")
PYEOF

echo ""
echo "[07] Phase 3b complete."
echo "[07] Best checkpoint: $OUTPUT_DIR/checkpoint_best_total.pth"
echo "[07] To evaluate: python src/tools/eval_rfdetr_seg.py --checkpoint $OUTPUT_DIR/checkpoint_best_total.pth"