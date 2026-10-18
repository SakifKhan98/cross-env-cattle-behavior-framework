"""
Build per-animal behavior timelines from VideoMAE tubelet predictions.

Groups tubelet predictions by (dataset, video_id, track_id), resolves
overlapping windows by averaging logits per frame, applies a median filter,
and merges consecutive same-label frames into timeline segments.

Output: one CSV per track under out_dir/{dataset}/{video_id}/{track_id}.csv
Schema:  track_id, label_id, label_name, start_frame, end_frame,
         start_sec, end_sec, duration_sec

Usage:
    python -m src.analytics.timeline \\
        --predictions_dir results/behavior/predictions \\
        --out_dir results/behavior_analytics/timelines
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = {
    0: "Standing", 1: "Lying", 2: "Foraging", 3: "Drinking",
    4: "Ruminating", 5: "Grooming", 6: "Other",
}
FPS = {"cbvd5": 25.0, "cvb": 30.0}
LOGIT_COLS = [f"logit_{i}" for i in range(7)]
MEDIAN_WINDOW = 5


def _extract_track_id(row: pd.Series) -> str:
    parts = row["tubelet_dir"].split("/")
    if row["dataset"] == "cvb":
        return parts[-2]   # .../track_0007/tubelet_0003 → track_0007
    return parts[-1]       # .../kf6_instc85ac7 → kf6_instc85ac7


def _median_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    result = arr.copy()
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = int(np.median(arr[lo:hi]))
    return result


def build_timeline(group: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    For all tubelets belonging to one (dataset, video_id, track_id):
      1. Collect logit vectors for every frame index covered by any tubelet.
      2. For frames covered by multiple tubelets, average their logits.
      3. Argmax → frame-level predicted label.
      4. Apply median filter (window=5) to smooth prediction noise.
      5. Merge consecutive same-label frames into segments.
    Returns a DataFrame of segments, or an empty DataFrame on failure.
    """
    frame_logits: dict[int, list[np.ndarray]] = {}
    for _, row in group.iterrows():
        logits = row[LOGIT_COLS].values.astype(np.float32)
        for f in range(int(row["start_frame"]), int(row["end_frame"])):
            frame_logits.setdefault(f, []).append(logits)

    if not frame_logits:
        return pd.DataFrame()

    frames = sorted(frame_logits.keys())
    labels = np.array(
        [np.mean(frame_logits[f], axis=0).argmax() for f in frames],
        dtype=np.int32,
    )

    win = min(MEDIAN_WINDOW, len(labels))
    if win % 2 == 0:
        win -= 1
    if win >= 3:
        labels = _median_filter_1d(labels, win)

    # Merge consecutive same-label frames into segments.
    # A gap (non-consecutive frame index) also starts a new segment.
    segments: list[tuple[int, int, int]] = []  # (label, start_f, end_f_excl)
    seg_start = frames[0]
    seg_label = int(labels[0])
    for i in range(1, len(frames)):
        new_segment = int(labels[i]) != seg_label or frames[i] != frames[i - 1] + 1
        if new_segment:
            segments.append((seg_label, seg_start, frames[i - 1] + 1))
            seg_start = frames[i]
            seg_label = int(labels[i])
    segments.append((seg_label, seg_start, frames[-1] + 1))

    track_id = group["track_id"].iloc[0]
    rows = []
    for label_id, start_f, end_f in segments:
        rows.append({
            "track_id": track_id,
            "label_id": label_id,
            "label_name": LABEL_NAMES[label_id],
            "start_frame": start_f,
            "end_frame": end_f,
            "start_sec": round(start_f / fps, 3),
            "end_sec": round(end_f / fps, 3),
            "duration_sec": round((end_f - start_f) / fps, 3),
        })
    return pd.DataFrame(rows)


def run_timeline_analysis(
    predictions_csv: "Path | str",
    output_dir: "Path | str",
    fps: float,
) -> int:
    """Build per-track behavior timelines from a single predictions CSV.

    Args:
        predictions_csv: Path to predictions.csv from classify_tubelets.
        output_dir:      Root for timeline CSVs.
                         Structure: {output_dir}/{dataset}/{video_id}/{track_id}.csv
        fps:             Video FPS from the VideoIngestor header — never hardcoded.

    Returns:
        Number of timeline CSVs written.
    """
    import re as _re

    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)

    df = pd.read_csv(predictions_csv)
    if df.empty:
        return 0

    def _extract_tid(tdir: str) -> str:
        m = _re.search(r"track_(\d+)", tdir)
        return f"track_{m.group(1)}" if m else tdir.split("/")[-2]

    df["track_id"] = df["tubelet_dir"].apply(_extract_tid)

    n_written = 0
    for (dataset, video_id, track_id), group in df.groupby(
        ["dataset", "video_id", "track_id"], sort=False
    ):
        timeline = build_timeline(group, fps)
        if timeline.empty:
            continue
        out_path = output_dir / str(dataset) / str(video_id) / f"{track_id}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timeline.to_csv(out_path, index=False)
        n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_dir", required=True,
        help="Directory containing *_val.csv prediction files",
    )
    parser.add_argument(
        "--tracking_dir", default=None,
        help="Tracking directory (reserved for future dataset support)",
    )
    parser.add_argument(
        "--run", default="videomae_combined_v1",
        help="Prediction run name to use (default: videomae_combined_v1)",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    csv_path = pred_dir / f"{args.run}_val.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Reading predictions: {csv_path}")
    df = pd.read_csv(csv_path)
    df["track_id"] = df.apply(_extract_track_id, axis=1)

    n_timelines = 0
    groups = df.groupby(["dataset", "video_id", "track_id"], sort=False)
    for (dataset, video_id, track_id), group in groups:
        fps = FPS.get(str(dataset), 25.0)
        timeline = build_timeline(group, fps)
        if timeline.empty:
            continue
        out_path = out_dir / str(dataset) / str(video_id) / f"{track_id}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timeline.to_csv(out_path, index=False)
        n_timelines += 1

    print(f"  Wrote {n_timelines} timeline CSVs → {out_dir}")


if __name__ == "__main__":
    main()
