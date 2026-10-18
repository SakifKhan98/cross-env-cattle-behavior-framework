"""
Aggregate OOD detection evaluation JSONs into results/generalization/ood_summary.csv.

Reads:
  results/detection/{opencows2020,cows2021,cattleeyeview,freeman}_eval.json
  results/segmentation/cattleeyeview_maskiou.json

Writes:
  results/generalization/ood_summary.csv

Run:
  python src/tools/aggregate_ood_summary.py
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DETECTION_JSONS = [
    ("opencows2020", "opencows2020_eval.json", "aerial top-down UAV"),
    ("cows2021",     "cows2021_eval.json",      "indoor UK barn"),
    ("cattleeyeview","cattleeyeview_eval.json",  "top-down outdoor polygon masks"),
    ("freeman",      "freeman_detection_eval.json", "angled real ranch"),
]

MASKIOU_JSON = REPO_ROOT / "results/segmentation/cattleeyeview_maskiou.json"

COLUMNS = [
    "dataset", "n_images",
    "mAP50", "mAP50_95", "AR100",
    "mAP_s", "mAP_m", "mAP_l",
    "mean_mask_iou", "domain_shift_note",
]

OUT_CSV = REPO_ROOT / "results/generalization/ood_summary.csv"


def load_maskiou():
    with open(MASKIOU_JSON) as f:
        return json.load(f)["mean_mask_iou"]


def build_rows():
    maskiou_val = load_maskiou()
    rows = []
    for dataset_key, fname, note in DETECTION_JSONS:
        path = REPO_ROOT / "results/detection" / fname
        with open(path) as f:
            d = json.load(f)

        # mAP_s / mAP_m are -1 when no instances in that size bucket; store as empty
        def clean(v):
            return "" if v == -1.0 else v

        rows.append({
            "dataset":          dataset_key,
            "n_images":         d["n_images"],
            "mAP50":            d["mAP50"],
            "mAP50_95":         d["mAP"],
            "AR100":            d["AR100"],
            "mAP_s":            clean(d["mAP_s"]),
            "mAP_m":            clean(d["mAP_m"]),
            "mAP_l":            clean(d["mAP_l"]),
            "mean_mask_iou":    maskiou_val if dataset_key == "cattleeyeview" else "",
            "domain_shift_note": note,
        })
    return rows


def main():
    rows = build_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV.relative_to(REPO_ROOT)}")
    print()
    header = f"{'dataset':<18} {'mAP50':>7} {'mAP50_95':>9} {'AR100':>7} {'mean_mask_iou':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        mask = f"{r['mean_mask_iou']:>14}" if r["mean_mask_iou"] != "" else f"{'—':>14}"
        print(f"{r['dataset']:<18} {r['mAP50']:>7.4f} {r['mAP50_95']:>9.4f} {r['AR100']:>7.4f} {mask}")


if __name__ == "__main__":
    main()
