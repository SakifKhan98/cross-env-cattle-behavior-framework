#!/usr/bin/env bash
# run_pipeline.sh — End-to-end pipeline runner
#
# Usage:
#   ./scripts/run_pipeline.sh                # run all stages (01–12)
#   ./scripts/run_pipeline.sh --from 5       # resume from stage 5
#   ./scripts/run_pipeline.sh --stage 11     # run just stage 11
#
# Each stage calls the corresponding numbered script in scripts/.
# GPU stages require NVIDIA runtime (docker/docker-compose.yml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FROM_STAGE=1
ONLY_STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)   FROM_STAGE="$2"; shift 2 ;;
    --stage)  ONLY_STAGE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

run_stage() {
  local num="$1"
  local script="$2"
  local desc="$3"

  if [[ -n "$ONLY_STAGE" && "$num" -ne "$ONLY_STAGE" ]]; then return; fi
  if [[ -z "$ONLY_STAGE" && "$num" -lt "$FROM_STAGE" ]]; then return; fi

  echo ""
  echo "═══════════════════════════════════════════════════════════"
  echo "  Stage $num: $desc"
  echo "═══════════════════════════════════════════════════════════"
  bash "$REPO_ROOT/scripts/$script"
}

run_stage  1  "01_inspect_data.sh"       "Inspect raw data"
run_stage  2  "02_prepare_cbvd5.sh"      "Convert CBVD-5 to COCO"
run_stage  3  "03_prepare_cvb.sh"        "Convert CVB to COCO"
run_stage  4  "04_merge_datasets.sh"     "Merge datasets"
run_stage  5  "05_train_detector.sh"     "Train RF-DETR detector (GPU)"
run_stage  6  "06_run_detection.sh"      "Run detection inference (GPU)"
run_stage  7  "07_run_segmentation.sh"   "Run SAM2 segmentation + OC-SORT tracking (GPU)"
run_stage  9  "09_generate_tubelets.sh"  "Export tubelets"
run_stage 10  "10_train_behavior.sh"     "Train VideoMAE behavior classifier (GPU)"
run_stage 11  "11_evaluate.sh"           "Evaluate behavior classifier"
run_stage 12  "12_generate_analytics.sh" "Generate analytics"

echo ""
echo "Pipeline complete."
