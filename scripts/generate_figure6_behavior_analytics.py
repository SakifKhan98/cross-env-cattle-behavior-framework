#!/usr/bin/env python
"""Figure 6 — Per-animal behavior analytics (representative CVB clip).

Regenerates the thesis figure ``results/figures/graphs/activity-budget.png`` as a
clean, publication-quality two-panel figure for the journal manuscript
(Computers and Electronics in Agriculture), Results Section 3.6.

Panel A: predicted per-track behavior timeline (Gantt).
Panel B: predicted clip-level activity budget (horizontal stacked bar).

Scientific notes
----------------
* All values are VideoMAE-predicted / model-derived behavior outputs — NOT
  ground truth.
* This is a single representative clip, NOT the manuscript-wide CVB aggregate
  over 543 tracks. The percentages here are the duration-weighted budget of the
  seven tracks in this one clip only.
* Nothing is invented: every segment and percentage is read straight from the
  committed per-track timeline CSVs.

Source of truth (never modified by this script)
-----------------------------------------------
    results/behavior_analytics/timelines/cvb/
        0258_arm01_gopro1_20200323_034430_beh4_ani6_ins1_cut_3/track_*.csv

(The same per-track budget also appears, identically, in
 results/behavior_analytics/activity_budget.csv.)

Run
---
    conda activate cattletransformer
    python scripts/generate_figure6_behavior_analytics.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
)

VIDEO_ID = "0258_arm01_gopro1_20200323_034430_beh4_ani6_ins1_cut_3"
TIMELINE_DIR = (
    REPO_ROOT / "results" / "behavior_analytics" / "timelines" / "cvb" / VIDEO_ID
)

OUT_DIR = REPO_ROOT / "results" / "figures" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = "Figure_6_behavior_analytics"

# --------------------------------------------------------------------------- #
# Canonical 7-class behavior taxonomy (data/label_map.json, CLAUDE.md §5)
# Fixed, colorblind-accessible palette — Okabe-Ito, plus a neutral grey for
# "Standing". Identical mapping is used in both panels and the legend.
# --------------------------------------------------------------------------- #
BEHAVIOR_ORDER = [
    "Standing",
    "Lying",
    "Foraging",
    "Drinking",
    "Ruminating",
    "Grooming",
    "Other",
]
BEHAVIOR_COLORS = {
    "Standing":   "#999999",  # neutral grey
    "Lying":      "#56B4E9",  # sky blue
    "Foraging":   "#009E73",  # bluish green
    "Drinking":   "#0072B2",  # blue
    "Ruminating": "#E69F00",  # orange
    "Grooming":   "#CC79A7",  # reddish purple
    "Other":      "#D55E00",  # vermillion
}

# Label the budget segment only when it is wide enough to hold text
BUDGET_LABEL_MIN_PCT = 4.0


def _text_color(hex_color: str) -> str:
    """Black or white text depending on background luminance."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.6 else "white"


# --------------------------------------------------------------------------- #
# Load + validate source data
# --------------------------------------------------------------------------- #
def load_tracks() -> dict[str, pd.DataFrame]:
    if not TIMELINE_DIR.is_dir():
        sys.exit(
            f"STOP: representative-clip timeline directory not found:\n  {TIMELINE_DIR}\n"
            "Cannot regenerate Figure 6 without the source data."
        )
    track_files = sorted(TIMELINE_DIR.glob("track_*.csv"))
    if not track_files:
        sys.exit(f"STOP: no track_*.csv files in {TIMELINE_DIR}")

    expected_cols = {
        "track_id", "label_id", "label_name",
        "start_frame", "end_frame", "start_sec", "end_sec", "duration_sec",
    }
    tracks: dict[str, pd.DataFrame] = {}
    for tf in track_files:
        df = pd.read_csv(tf)
        if df.empty:
            continue
        missing = expected_cols - set(df.columns)
        if missing:
            sys.exit(f"STOP: {tf.name} missing columns {sorted(missing)}")
        bad = set(df["label_name"]) - set(BEHAVIOR_ORDER)
        if bad:
            sys.exit(f"STOP: {tf.name} has non-canonical behavior label(s): {sorted(bad)}")
        tracks[tf.stem] = df.reset_index(drop=True)

    if not tracks:
        sys.exit(f"STOP: every track_*.csv in {TIMELINE_DIR} was empty")
    return tracks


def clip_budget(tracks: dict[str, pd.DataFrame]) -> tuple[pd.Series, float]:
    """Duration-weighted activity budget for THIS clip (percent per behavior)."""
    all_segs = pd.concat(tracks.values(), ignore_index=True)
    total_sec = float(all_segs["duration_sec"].sum())
    by_beh = all_segs.groupby("label_name")["duration_sec"].sum()
    pct = (by_beh / total_sec * 100.0).reindex(BEHAVIOR_ORDER).fillna(0.0)
    return pct, total_sec


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(tracks: dict[str, pd.DataFrame], budget_pct: pd.Series):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    track_ids = sorted(tracks.keys())  # track_0001 .. track_0012
    n = len(track_ids)
    row_labels = [f"Track {int(t.split('_')[1])}" for t in track_ids]

    clip_end = max(float(df["end_sec"].max()) for df in tracks.values())

    fig = plt.figure(figsize=(7.5, 4.4))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[3.4, 1.0], hspace=0.75,
        left=0.13, right=0.975, top=0.9, bottom=0.19,
    )
    ax_tl = fig.add_subplot(gs[0])
    ax_bud = fig.add_subplot(gs[1])

    # ----- Panel A : per-track behavior timeline -------------------------- #
    bar_h = 0.62
    for i, tid in enumerate(track_ids):
        y = n - 1 - i  # Track 1 at the top
        for _, seg in tracks[tid].iterrows():
            ax_tl.broken_barh(
                [(seg["start_sec"], seg["duration_sec"])],
                (y - bar_h / 2, bar_h),
                facecolors=BEHAVIOR_COLORS[seg["label_name"]],
                edgecolors="white",
                linewidth=0.5,
            )
    ax_tl.set_yticks(range(n))
    ax_tl.set_yticklabels(list(reversed(row_labels)))
    ax_tl.set_ylim(-0.6, n - 0.4)
    ax_tl.set_xlim(0, clip_end)
    ax_tl.set_xlabel("Time (s)")
    ax_tl.set_ylabel("Tracked animal identity")
    ax_tl.set_title("Predicted behavior timeline", fontweight="bold", pad=6)
    ax_tl.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#dddddd")
    ax_tl.set_axisbelow(True)
    for s in ("top", "right"):
        ax_tl.spines[s].set_visible(False)
    ax_tl.tick_params(length=0)

    # ----- Panel B : clip-level activity budget -------------------------- #
    left = 0.0
    for beh in BEHAVIOR_ORDER:
        pct = float(budget_pct[beh])
        if pct <= 0:
            continue
        ax_bud.barh(
            0, pct, left=left, height=0.85,
            color=BEHAVIOR_COLORS[beh], edgecolor="white", linewidth=0.6,
        )
        if pct >= BUDGET_LABEL_MIN_PCT:
            ax_bud.text(
                left + pct / 2, 0, f"{pct:.0f}%",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=_text_color(BEHAVIOR_COLORS[beh]),
            )
        left += pct
    ax_bud.set_xlim(0, 100)
    ax_bud.set_ylim(-0.55, 0.55)
    ax_bud.set_yticks([])
    ax_bud.set_xlabel("Time (%)")
    ax_bud.set_title("Predicted activity budget", fontweight="bold", pad=6)
    for s in ("top", "right", "left"):
        ax_bud.spines[s].set_visible(False)
    ax_bud.tick_params(axis="y", length=0)

    # ----- Panel labels ------------------------------------------------- #
    ax_tl.text(-0.135, 1.14, "A", transform=ax_tl.transAxes,
               fontsize=12, fontweight="bold", va="top", ha="left")
    ax_bud.text(-0.135, 1.55, "B", transform=ax_bud.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

    # ----- Shared legend : all seven canonical classes ---------------- #
    handles = [
        mpatches.Patch(facecolor=BEHAVIOR_COLORS[b], edgecolor="#666666",
                       linewidth=0.4, label=b)
        for b in BEHAVIOR_ORDER
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=7,
        fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.015),
        handlelength=1.3, columnspacing=1.1, handletextpad=0.5,
    )

    return fig, track_ids, row_labels, clip_end


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    tracks = load_tracks()
    budget_pct, total_sec = clip_budget(tracks)

    fig, track_ids, row_labels, clip_end = build_figure(tracks, budget_pct)

    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=400)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Console report / audit trail
    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("Figure 6 — Per-animal behavior analytics (representative CVB clip)")
    print("=" * 78)
    print("Source data (unmodified):")
    print(f"  {TIMELINE_DIR.relative_to(REPO_ROOT)}/track_*.csv")
    print("  (identical per-track budget also in "
          "results/behavior_analytics/activity_budget.csv)")
    print()
    print(f"Representative clip : {VIDEO_ID}  (dataset: CVB)")
    print(f"Tracks plotted     : {', '.join(track_ids)}")
    print(f"Row labels         : {', '.join(row_labels)}")
    print(f"Time span plotted  : 0.00 - {clip_end:.3f} s  "
          f"(clip total track-time = {total_sec:.3f} s)")
    print()
    print("Behavior -> color mapping (both panels + legend):")
    for b in BEHAVIOR_ORDER:
        print(f"  {b:11s} {BEHAVIOR_COLORS[b]}")
    print()
    print("Predicted clip-level activity budget (duration-weighted over the 7 tracks):")
    for b in BEHAVIOR_ORDER:
        p = float(budget_pct[b])
        shown = f"  labelled '{p:.0f}%'" if p >= BUDGET_LABEL_MIN_PCT else (
            "  (segment too small for in-bar label)" if p > 0 else "  (absent in clip)")
        print(f"  {b:11s} {p:7.3f} %{shown}")
    print(f"  {'TOTAL':11s} {float(budget_pct.sum()):7.3f} %")
    print()

    print("-" * 78)
    print("VERIFICATION — timeline segments actually plotted")
    print("-" * 78)
    for tid in track_ids:
        df = tracks[tid]
        print(f"\n{tid}  ({f'Track {int(tid.split(chr(95))[1])}'})")
        print(df[["label_name", "start_sec", "end_sec", "duration_sec"]]
              .to_string(index=False))
        # note gaps
        covered = df[["start_sec", "end_sec"]].values.tolist()
        covered.sort()
        gaps = []
        if covered[0][0] > 0.05:
            gaps.append(f"0.000-{covered[0][0]:.3f}")
        for a, b in zip(covered, covered[1:]):
            if b[0] - a[1] > 0.05:
                gaps.append(f"{a[1]:.3f}-{b[0]:.3f}")
        if covered[-1][1] < clip_end - 0.05:
            gaps.append(f"{covered[-1][1]:.3f}-{clip_end:.3f}")
        print(f"  absent (gap) intervals: {gaps if gaps else 'none'}")

    print()
    print("-" * 78)
    print(f"Script : scripts/generate_figure6_behavior_analytics.py")
    print(f"Outputs: {pdf_path.relative_to(REPO_ROOT)}")
    print(f"         {png_path.relative_to(REPO_ROOT)}  (400 dpi)")


if __name__ == "__main__":
    main()
