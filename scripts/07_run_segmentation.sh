#!/usr/bin/env bash
# scripts/07_run_segmentation.sh
# Phase 3 — SAM2 Segmentation
#
# USAGE:
#   bash scripts/07_run_segmentation.sh                        # Both datasets, full run
#   bash scripts/07_run_segmentation.sh --sanity               # Quick test, first 3 videos
#   bash scripts/07_run_segmentation.sh --dataset cbvd5        # CBVD-5 only
#   bash scripts/07_run_segmentation.sh --dataset cvb          # CVB only
#   bash scripts/07_run_segmentation.sh --video_id 618         # Single video debug
#
# Run from the project root:
#   cd ~/TXST/Thesis/cattle-vision-framework
#   bash scripts/07_run_segmentation.sh --sanity

set -e  # Exit immediately if any command fails

# ---------------------------------------------------------------------------
# Pass all arguments through to the Python script
# ---------------------------------------------------------------------------
PYTHON_ARGS="$@"

# ---------------------------------------------------------------------------
# Activate conda environment
# ---------------------------------------------------------------------------
echo "[07] Activating conda environment: cattletransformer"

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "[ERROR] conda not found. Is Anaconda/Miniconda installed?"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cattletransformer

# ---------------------------------------------------------------------------
# VRAM memory allocator setting — reduces fragmentation on RTX 3060
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[07] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# ---------------------------------------------------------------------------
# Confirm we are in the correct project root
# ---------------------------------------------------------------------------
if [ ! -f "configs/segmentation/sam2.yaml" ]; then
    echo "[ERROR] Config not found: configs/segmentation/sam2.yaml"
    echo "  Make sure you are running from the project root:"
    echo "    cd ~/TXST/Thesis/cattle-vision-framework"
    echo "    bash scripts/07_run_segmentation.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Check SAM2 is installed
# ---------------------------------------------------------------------------
echo "[07] Checking SAM2 installation..."
python3 -c "from sam2.build_sam import build_sam2; print('[07] SAM2 OK')" || {
    echo ""
    echo "[ERROR] SAM2 is not installed."
    echo "  Fix: pip install sam2"
    exit 1
}

# ---------------------------------------------------------------------------
# Check pycocotools is installed (needed for RLE mask encoding)
# ---------------------------------------------------------------------------
python3 -c "from pycocotools import mask; print('[07] pycocotools OK')" || {
    echo ""
    echo "[ERROR] pycocotools is not installed."
    echo "  Fix: pip install pycocotools"
    exit 1
}

# ---------------------------------------------------------------------------
# Check SAM2 checkpoint exists
# ---------------------------------------------------------------------------
CHECKPOINT="models/sam2/sam2.1_hiera_large.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo ""
    echo "[ERROR] SAM2 checkpoint not found: $CHECKPOINT"
    echo ""
    echo "  Download it with:"
    echo "    mkdir -p models/sam2"
    echo "    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \\"
    echo "         -O models/sam2/sam2.1_hiera_large.pt"
    echo ""
    exit 1
fi
echo "[07] Checkpoint found: $CHECKPOINT"

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
mkdir -p data/processed/segmentation/cbvd5
mkdir -p data/processed/segmentation/cvb
echo "[07] Output directories ready."

# ---------------------------------------------------------------------------
# Print GPU info
# ---------------------------------------------------------------------------
echo "[07] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available)"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo ""
echo "[07] Starting Phase 3 — SAM2 Segmentation"
echo "[07] Args passed to Python: $PYTHON_ARGS"
echo ""

python3 src/segmentation/segment.py \
    --config configs/segmentation/sam2.yaml \
    $PYTHON_ARGS

echo ""
echo "[07] Phase 3 complete."