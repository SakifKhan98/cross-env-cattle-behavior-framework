"""
src/tools/aggregate_perturbation.py

Reads all intermediate per-condition JSONs under
results/generalization/perturbation_runs/ and assembles them into
results/generalization/perturbation_delta.csv.

Idempotent: re-running after a single condition is re-run picks up the
updated intermediate JSON without touching the others.

Usage:
    python src/tools/aggregate_perturbation.py
    python src/tools/aggregate_perturbation.py --runs_dir results/generalization/perturbation_runs
    python src/tools/aggregate_perturbation.py --out results/generalization/perturbation_delta.csv
"""

import argparse
import csv
import json
from pathlib import Path

DATASET_ORDER = ["cbvd5", "cvb", "opencows2020", "cows2021", "cattleeyeview", "freeman"]
PERTURBATION_ORDER = ["brightness", "gaussian_noise", "motion_blur", "fog", "rain"]
SEVERITY_ORDER = ["low", "high"]

COLUMNS = [
    "dataset", "perturbation_type", "severity", "n_images",
    "mAP50_clean", "mAP50_perturbed", "delta_mAP50",
    "mAP_clean", "mAP_perturbed",
    "AR100_clean", "AR100_perturbed",
]


def sort_key(row):
    d = DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 99
    p = PERTURBATION_ORDER.index(row["perturbation_type"]) if row["perturbation_type"] in PERTURBATION_ORDER else 99
    s = SEVERITY_ORDER.index(row["severity"]) if row["severity"] in SEVERITY_ORDER else 99
    return (d, p, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="results/generalization/perturbation_runs")
    p.add_argument("--out", default="results/generalization/perturbation_delta.csv")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)

    json_files = sorted(runs_dir.glob("*.json"))
    if not json_files:
        print(f"[aggregate] No JSON files found in {runs_dir}")
        return

    rows = []
    for jf in json_files:
        with open(jf) as f:
            d = json.load(f)
        row = {col: d.get(col, "") for col in COLUMNS}
        rows.append(row)

    rows.sort(key=sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aggregate] Wrote {len(rows)} rows → {out_path}")

    # Summary table
    print(f"\n{'─'*72}")
    print(f"  {'Dataset':<16} {'Perturbation':<16} {'Sev':<5} {'Clean':>7} {'Perturb':>8} {'Delta':>8}")
    print(f"{'─'*72}")
    for r in rows:
        print(f"  {r['dataset']:<16} {r['perturbation_type']:<16} {r['severity']:<5} "
              f"{float(r['mAP50_clean']):>7.4f} {float(r['mAP50_perturbed']):>8.4f} "
              f"{float(r['delta_mAP50']):>+8.4f}")
    print(f"{'─'*72}")
    print(f"\n  Total conditions: {len(rows)}  (expected: 40)")


if __name__ == "__main__":
    main()
