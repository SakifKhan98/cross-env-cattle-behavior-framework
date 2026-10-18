#!/bin/bash
# 15_infer_video.sh
# Run the Phase 9 single-video inference pipeline.
# Run from project root: bash scripts/15_infer_video.sh --video path/to/video.avi
#                    or: bash scripts/15_infer_video.sh --frames_dir path/to/frames/
#
# Flags:
#   --video       <path>    Input video file (mutually exclusive with --frames_dir)
#   --frames_dir  <path>    Directory of sorted JPEG frames (mutually exclusive with --video)
#   --frames_fps  <float>   FPS to use when building video from frames (default: 25.0)
#   --output_dir  <path>    Root output directory (default: results/inference)
#   --config      <path>    YAML config (default: configs/inference/default.yaml)
#   --conf_thresh <float>   Detection confidence threshold (overrides config)
#   --cleanup               Delete intermediate files after the run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer

cd "$REPO_ROOT"
python src/inference/run_inference.py "$@"
