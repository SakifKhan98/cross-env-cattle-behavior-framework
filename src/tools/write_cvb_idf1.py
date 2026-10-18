"""
Write CVB and CBVD-5 tracking metric JSON files.

Reads:
  results/tracking/tracking_summary_all.json   (already CVB-only)
  results/tracking/tracking_per_video_all.csv

Writes:
  results/tracking/cvb_idf1.json     — CVB IDF1/MOTA/etc with dataset + n_videos fields
  results/tracking/cbvd5_idf1.json   — documented stub (metrics not computable)

Run:
  python scripts/write_cvb_idf1.py
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_JSON = REPO_ROOT / "results/tracking/tracking_summary_all.json"
PER_VIDEO_CSV = REPO_ROOT / "results/tracking/tracking_per_video_all.csv"
CVB_OUT = REPO_ROOT / "results/tracking/cvb_idf1.json"
CBVD5_OUT = REPO_ROOT / "results/tracking/cbvd5_idf1.json"


def count_videos():
    with open(PER_VIDEO_CSV) as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)

    n_videos = count_videos()

    cvb = {
        "dataset": "cvb",
        "n_videos": n_videos,
        "config": summary["config"],
        "results": summary["results"],
        "note": (
            "All videos in tracking_per_video_all.csv belong to the CVB dataset. "
            "tracking_summary_all.json is therefore a CVB-only aggregate."
        ),
    }
    with open(CVB_OUT, "w") as f:
        json.dump(cvb, f, indent=2)
    print(f"Wrote {CVB_OUT.relative_to(REPO_ROOT)}")
    print(f"  IDF1={cvb['results']['IDF1']}  MOTA={cvb['results']['MOTA']}  n_videos={n_videos}")

    cbvd5 = {
        "dataset": "cbvd5",
        "computable": False,
        "note": (
            "CBVD-5 annotations do not provide persistent track IDs across frames. "
            "Standard MOT metrics (IDF1, MOTA) require ground-truth trajectory IDs "
            "to match predicted tracks over time. Because CBVD-5 annotates per-frame "
            "bounding boxes without cross-frame identity, MOT evaluation is not applicable. "
            "OC-SORT was run on CBVD-5 solely to generate tubelets for VideoMAE training; "
            "tracking quality on CBVD-5 cannot be quantified."
        ),
    }
    with open(CBVD5_OUT, "w") as f:
        json.dump(cbvd5, f, indent=2)
    print(f"Wrote {CBVD5_OUT.relative_to(REPO_ROOT)}  (stub, computable=false)")


if __name__ == "__main__":
    main()
