"""
Aggregates per-min_hits tracking summary JSONs into minhits_ablation.csv.

Reads:
  results/tracking/cvb_mh{1,2,3,5}_summary.json

Writes:
  results/tracking/minhits_ablation.csv

Columns: min_hits, idf1, mota, motp, total_id_switches, n_videos
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "tracking"

MH_VALUES = [1, 2, 3, 5]
FIELDNAMES = ["min_hits", "idf1", "mota", "motp", "total_id_switches", "n_videos"]


def main():
    rows = []
    for mh in MH_VALUES:
        src = RESULTS_DIR / f"cvb_mh{mh}_summary.json"
        with open(src) as f:
            d = json.load(f)
        r = d["results"]
        rows.append(
            {
                "min_hits": mh,
                "idf1": r["IDF1"],
                "mota": r["MOTA"],
                "motp": r["MOTP"],
                "total_id_switches": r["total_id_switches"],
                "n_videos": r["videos_evaluated"],
            }
        )

    out_path = RESULTS_DIR / "minhits_ablation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {out_path}")
    for row in rows:
        print(
            f"  mh={row['min_hits']:2d}  IDF1={row['idf1']:.2f}  "
            f"MOTA={row['mota']:.2f}  MOTP={row['motp']:.2f}  "
            f"IDS={row['total_id_switches']}  n={row['n_videos']}"
        )


if __name__ == "__main__":
    main()
