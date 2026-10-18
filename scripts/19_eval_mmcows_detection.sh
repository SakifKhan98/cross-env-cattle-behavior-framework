#!/bin/bash
# 19_eval_mmcows_detection.sh
# Run RF-DETR-Seg OOD detection evaluation on the mmCows cam_4 COCO dataset.
#
# Run from project root: bash scripts/19_eval_mmcows_detection.sh
#
# Prerequisites:
#   - bash scripts/18_prepare_mmcows_detection.sh  (builds COCO dataset)
#   - runs/seg_medium_lr5e5/checkpoint_best_ema.pth  (RF-DETR-Seg checkpoint)
#
# Output:
#   results/detection/mmcows_eval.json
#   results/generalization/ood_summary.csv  (mmcows row appended)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer

cd "$REPO_ROOT"

CHECKPOINT="runs/seg_medium_lr5e5/checkpoint_best_ema.pth"
DATASET_DIR="data/processed/detection/mmcows"
OUTPUT="results/detection/mmcows_eval.json"
THRESHOLD=0.3

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "        Download from HuggingFace: sakifkhan/cattle-vision-framework"
    exit 1
fi

if [[ ! -f "$DATASET_DIR/_annotations.coco.json" ]]; then
    echo "[ERROR] COCO dataset not found: $DATASET_DIR/_annotations.coco.json"
    echo "        Run 18_prepare_mmcows_detection.sh first."
    exit 1
fi

echo "Running RF-DETR-Seg OOD eval on mmCows cam_4 ..."
python src/tools/eval_detection_ood.py \
    --checkpoint   "$CHECKPOINT" \
    --dataset_dir  "$DATASET_DIR" \
    --output       "$OUTPUT" \
    --dataset_name mmcows \
    --threshold    "$THRESHOLD" \
    --seg

echo ""
echo "Detection eval complete.  Results → $OUTPUT"
echo ""

# ── Append mmcows row to ood_summary.csv ─────────────────────────────────────
python - <<'EOF'
import json, csv, sys
from pathlib import Path

result_file = Path("results/detection/mmcows_eval.json")
summary_file = Path("results/generalization/ood_summary.csv")

if not result_file.exists():
    print(f"[WARN] {result_file} not found — skipping ood_summary.csv update")
    sys.exit(0)

with open(result_file) as f:
    r = json.load(f)

def _blank_if_neg(v):
    """Return empty string for COCO sentinel -1.0 (no objects in size bin)."""
    try:
        return "" if float(v) < 0 else v
    except (TypeError, ValueError):
        return v

new_row = {
    "dataset":                   "mmcows",
    "n_images":                  r.get("n_images", ""),
    "mAP50":                     r.get("mAP50", ""),
    "mAP50_95":                  r.get("mAP", ""),        # eval JSON key is "mAP"
    "AR100":                     r.get("AR100", ""),
    "mAP_s":                     _blank_if_neg(r.get("mAP_s", "")),
    "mAP_m":                     _blank_if_neg(r.get("mAP_m", "")),
    "mAP_l":                     _blank_if_neg(r.get("mAP_l", "")),
    "mean_mask_iou":             "",
    "behavior_macro_f1_4class":  "",
    "domain_shift_note":         "indoor isometric corner-mounted 4-camera barn",
}

rows = []
fieldnames = []
if summary_file.exists():
    with open(summary_file, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

# Add new columns if not already present
for col in ("behavior_macro_f1_4class",):
    if col not in fieldnames:
        fieldnames.append(col)

# Remove existing mmcows row if re-running
rows = [row for row in rows if row.get("dataset") != "mmcows"]
rows.append(new_row)

with open(summary_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"ood_summary.csv updated — mmcows row added.")
EOF
