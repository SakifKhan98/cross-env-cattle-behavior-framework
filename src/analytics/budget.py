"""
Compute activity budgets, behavior transition matrices, and behavioral
deviation analysis from per-animal timeline CSVs produced by timeline.py.

Inputs:
    timelines_dir/  {dataset}/{video_id}/{track_id}.csv

Outputs (written to out_dir/):
    activity_budget.csv      — % time per behavior per (dataset, video_id, track_id)
    transition_matrix.csv    — behavior-to-behavior transition counts & probabilities
    behavior_deviation.csv   — per-track deviation from dataset-median budget; IQR outlier flag

Schema — activity_budget.csv:
    dataset, video_id, track_id, label_id, behavior, duration_sec, pct_time

Schema — transition_matrix.csv:
    dataset, from_label, from_behavior, to_label, to_behavior, count, probability

Schema — behavior_deviation.csv:
    dataset, video_id, track_id, label_id, behavior,
    pct_time, baseline_median, deviation, is_outlier

Usage:
    python -m src.analytics.budget \\
        --timelines_dir results/behavior_analytics/timelines \\
        --out_dir results/behavior_analytics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BEHAVIORS = {
    0: "Standing", 1: "Lying", 2: "Foraging", 3: "Drinking",
    4: "Ruminating", 5: "Grooming", 6: "Other",
}


def load_all_timelines(timelines_dir: Path) -> pd.DataFrame:
    """
    Recursively loads all timeline CSVs.
    Expects path structure: {timelines_dir}/{dataset}/{video_id}/{track_id}.csv
    """
    frames = []
    for csv_path in sorted(timelines_dir.rglob("*.csv")):
        parts = csv_path.relative_to(timelines_dir).parts
        if len(parts) < 3:
            continue
        dataset, video_id = parts[0], parts[1]
        df = pd.read_csv(csv_path)
        df.insert(0, "dataset", dataset)
        df.insert(1, "video_id", video_id)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No timeline CSVs found under {timelines_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_activity_budget(timelines: pd.DataFrame) -> pd.DataFrame:
    """
    Per (dataset, video_id, track_id): compute % time in each behavior.
    All 7 behavior classes are always present; behaviors with 0 duration get pct_time=0.
    """
    records = []
    groups = timelines.groupby(["dataset", "video_id", "track_id"], sort=False)
    for (dataset, video_id, track_id), grp in groups:
        total_sec = grp["duration_sec"].sum()
        if total_sec == 0:
            continue
        dur_by_label = grp.groupby("label_id")["duration_sec"].sum()
        for label_id, label_name in BEHAVIORS.items():
            dur = float(dur_by_label.get(label_id, 0.0))
            records.append({
                "dataset": dataset,
                "video_id": video_id,
                "track_id": track_id,
                "label_id": label_id,
                "behavior": label_name,
                "duration_sec": round(dur, 3),
                "pct_time": round(100.0 * dur / total_sec, 4),
            })
    return pd.DataFrame(records)


def compute_transition_matrix(timelines: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset, count behavior-to-behavior transitions across all tracks
    and convert to per-row probabilities (row = from_label).
    """
    transition_rows = []
    groups = timelines.groupby(["dataset", "video_id", "track_id"], sort=False)
    for (dataset, video_id, track_id), grp in groups:
        segs = grp.sort_values("start_frame")["label_id"].tolist()
        for a, b in zip(segs[:-1], segs[1:]):
            transition_rows.append({"dataset": dataset, "from_label": int(a), "to_label": int(b)})

    if not transition_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(transition_rows)
    result_parts = []
    for dataset, ds_grp in raw.groupby("dataset"):
        counts = (
            ds_grp.groupby(["from_label", "to_label"])
            .size()
            .reset_index(name="count")
        )
        row_totals = counts.groupby("from_label")["count"].transform("sum")
        counts["probability"] = (counts["count"] / row_totals).round(4)
        counts["from_behavior"] = counts["from_label"].map(BEHAVIORS)
        counts["to_behavior"] = counts["to_label"].map(BEHAVIORS)
        counts["dataset"] = dataset
        result_parts.append(
            counts[["dataset", "from_label", "from_behavior",
                     "to_label", "to_behavior", "count", "probability"]]
        )
    return pd.concat(result_parts, ignore_index=True)


def compute_behavioral_deviation(budget: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, behavior): compute the dataset-level median pct_time
    and IQR. Per track, report deviation from that median and flag as outlier
    if |deviation| > 1.5 × IQR (only when IQR > 0).

    Per §4.6.3 of the approved thesis proposal: this is descriptive deviation
    analysis, not clinical welfare thresholding.
    """
    records = []
    groups = budget.groupby(["dataset", "label_id", "behavior"], sort=False)
    for (dataset, label_id, behavior), grp in groups:
        median = grp["pct_time"].median()
        q1 = grp["pct_time"].quantile(0.25)
        q3 = grp["pct_time"].quantile(0.75)
        iqr = q3 - q1
        threshold = 1.5 * iqr

        for _, row in grp.iterrows():
            dev = abs(float(row["pct_time"]) - median)
            records.append({
                "dataset": dataset,
                "video_id": row["video_id"],
                "track_id": row["track_id"],
                "label_id": int(label_id),
                "behavior": behavior,
                "pct_time": row["pct_time"],
                "baseline_median": round(float(median), 4),
                "deviation": round(dev, 4),
                "is_outlier": bool(dev > threshold and iqr > 0),
            })
    return pd.DataFrame(records)


def run_budget_analysis(
    timelines_dir: "Path | str",
    output_dir: "Path | str",
) -> None:
    """Compute activity budgets and behavioral deviation from timeline CSVs.

    Writes activity_budget.csv, transition_matrix.csv, and
    behavior_deviation.csv to output_dir.
    """
    timelines_dir = Path(timelines_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        timelines = load_all_timelines(timelines_dir)
    except RuntimeError:
        return  # no timeline CSVs available yet

    budget = compute_activity_budget(timelines)
    budget.to_csv(output_dir / "activity_budget.csv", index=False)

    transitions = compute_transition_matrix(timelines)
    transitions.to_csv(output_dir / "transition_matrix.csv", index=False)

    deviation = compute_behavioral_deviation(budget)
    deviation.to_csv(output_dir / "behavior_deviation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines_dir", required=True,
                        help="Directory of timeline CSVs from timeline.py")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for analytics CSVs")
    args = parser.parse_args()

    timelines_dir = Path(args.timelines_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading timeline CSVs ...")
    timelines = load_all_timelines(timelines_dir)
    print(f"    {len(timelines):,} segments across "
          f"{timelines.groupby(['dataset','video_id','track_id']).ngroups:,} tracks")

    print("  Computing activity budgets ...")
    budget = compute_activity_budget(timelines)
    budget.to_csv(out_dir / "activity_budget.csv", index=False)
    print(f"    → {len(budget):,} rows → {out_dir / 'activity_budget.csv'}")

    print("  Computing transition matrix ...")
    transitions = compute_transition_matrix(timelines)
    transitions.to_csv(out_dir / "transition_matrix.csv", index=False)
    print(f"    → {len(transitions):,} rows → {out_dir / 'transition_matrix.csv'}")

    print("  Computing behavioral deviation ...")
    deviation = compute_behavioral_deviation(budget)
    deviation.to_csv(out_dir / "behavior_deviation.csv", index=False)
    n_outliers = deviation["is_outlier"].sum()
    print(f"    → {len(deviation):,} rows, {n_outliers:,} outlier flags "
          f"→ {out_dir / 'behavior_deviation.csv'}")


if __name__ == "__main__":
    main()
