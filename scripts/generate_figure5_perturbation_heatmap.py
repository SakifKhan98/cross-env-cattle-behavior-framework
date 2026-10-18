#!/usr/bin/env python
"""Figure 5 — Robustness to controlled visual degradation.

Generates a two-panel diverging heatmap (low / high severity) of the change in
detection mAP@50 relative to each dataset's clean baseline, under five synthetic
visual perturbations.

Source of truth (never modified by this script):
    results/generalization/perturbation_delta.csv

Column `delta_mAP50` is stored as a fraction (perturbed - clean); this script
multiplies by 100 to express the change in percentage points (pp).

Supports manuscript Results Section 3.5 (Computers and Electronics in Agriculture).
mmCows is intentionally excluded — it had no perturbation experiment executed.

Run:
    conda activate cattletransformer
    python scripts/generate_figure5_perturbation_heatmap.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
)
SRC_CSV = REPO_ROOT / "results" / "generalization" / "perturbation_delta.csv"
OUT_DIR = REPO_ROOT / "results" / "figures" / "manuscript"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = "Figure_5_perturbation_robustness"

# --------------------------------------------------------------------------- #
# Repository -> manuscript label mapping (confirmed against
# notebooks/figure_charts.ipynb and CLAUDE.md)
# --------------------------------------------------------------------------- #
DATASET_ORDER = ["cbvd5", "cvb", "opencows2020", "cows2021", "cattleeyeview", "freeman"]
DATASET_LABELS = {
    "cbvd5": "CBVD-5",
    "cvb": "CVB",
    "opencows2020": "OpenCows2020",
    "cows2021": "Cows2021",
    "cattleeyeview": "CattleEyeView",
    "freeman": "Freeman Center",
}

# Manuscript row ordering for Figure 5.
PERT_ORDER = ["brightness", "fog", "rain", "gaussian_noise", "motion_blur"]
PERT_LABELS = {
    "brightness": "Brightness",
    "fog": "Fog",
    "rain": "Rain",
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
}

SEVERITY_ORDER = ["low", "high"]
SEVERITY_TITLES = {"low": "Low severity", "high": "High severity"}


# --------------------------------------------------------------------------- #
# Load + validate
# --------------------------------------------------------------------------- #
def load_delta_pp() -> pd.DataFrame:
    df = pd.read_csv(SRC_CSV)
    required = {"dataset", "perturbation_type", "severity", "delta_mAP50"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise SystemExit(f"CSV missing expected columns: {sorted(missing_cols)}")

    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    df = df[df["perturbation_type"].isin(PERT_ORDER)].copy()
    df = df[df["severity"].isin(SEVERITY_ORDER)].copy()

    # change in mAP@50, percentage points
    df["delta_pp"] = df["delta_mAP50"] * 100.0

    # Completeness check: every dataset x perturbation x severity cell must exist.
    missing_cells = []
    for sev in SEVERITY_ORDER:
        for pert in PERT_ORDER:
            for ds in DATASET_ORDER:
                hit = df[
                    (df["severity"] == sev)
                    & (df["perturbation_type"] == pert)
                    & (df["dataset"] == ds)
                ]
                if len(hit) == 0:
                    missing_cells.append((sev, pert, ds))
                elif len(hit) > 1:
                    raise SystemExit(
                        f"Duplicate rows for ({sev}, {pert}, {ds}) — refusing to plot."
                    )
    if missing_cells:
        raise SystemExit(
            "STOP: missing dataset/perturbation/severity combinations in CSV:\n"
            + "\n".join(f"  severity={s}, perturbation={p}, dataset={d}" for s, p, d in missing_cells)
        )
    return df


def pivot_for(df: pd.DataFrame, severity: str) -> pd.DataFrame:
    sub = df[df["severity"] == severity]
    piv = sub.pivot(index="perturbation_type", columns="dataset", values="delta_pp")
    piv = piv.loc[PERT_ORDER, DATASET_ORDER]
    piv.index = [PERT_LABELS[p] for p in PERT_ORDER]
    piv.columns = [DATASET_LABELS[d] for d in DATASET_ORDER]
    return piv


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def build_figure(piv_low: pd.DataFrame, piv_high: pd.DataFrame, vmin: float, vmax: float):
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

    # Diverging, muted, centred at 0. Negative -> warm (orange/red),
    # positive -> muted blue. Same scale for both panels.
    cmap = sns.color_palette("vlag_r", as_cmap=True)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.5, 3.5), sharey=True, constrained_layout=True
    )

    panels = [("A", "low", piv_low), ("B", "high", piv_high)]
    meshes = []
    for ax, (letter, sev, piv) in zip(axes, panels):
        # pcolormesh draws row 0 at the bottom; reverse so the manuscript row
        # order (Brightness, Fog, Rain, Gaussian noise, Motion blur) reads
        # top-to-bottom.
        piv = piv.iloc[::-1]
        data = piv.to_numpy()
        mesh = ax.pcolormesh(
            data,
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=0.6,
        )
        meshes.append(mesh)

        ax.set_title(SEVERITY_TITLES[sev], pad=8, fontweight="bold")
        ax.set_aspect("equal")

        ax.set_xticks(np.arange(data.shape[1]) + 0.5)
        ax.set_xticklabels(piv.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(data.shape[0]) + 0.5)
        ax.set_yticklabels(piv.index)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        # Cell annotations — exact delta values, one decimal place (pp).
        rgba = mesh.cmap(mesh.norm(data))
        luminance = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                txt = f"{val:+.1f}"
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if luminance[r, c] > 0.6 else "white",
                )

        # Panel label in the upper-left corner.
        ax.text(
            -0.14,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    axes[0].set_ylabel("Perturbation")

    cbar = fig.colorbar(
        meshes[0],
        ax=axes,
        orientation="vertical",
        fraction=0.046,
        pad=0.02,
        shrink=0.9,
        aspect=28,
        extend="neither",
    )
    cbar.set_label("Change in mAP@50 (percentage points)")
    cbar.outline.set_visible(False)

    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_delta_pp()

    piv_low = pivot_for(df, "low")
    piv_high = pivot_for(df, "high")

    all_vals = np.concatenate([piv_low.to_numpy().ravel(), piv_high.to_numpy().ravel()])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())

    fig = build_figure(piv_low, piv_high, vmin, vmax)

    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=400)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Console report / audit trail
    # ------------------------------------------------------------------ #
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=" * 78)
    print("Figure 5 — Robustness to controlled visual degradation")
    print("=" * 78)
    print(f"Source CSV : {SRC_CSV.relative_to(REPO_ROOT)}  (unmodified)")
    print(f"Script     : scripts/generate_figure5_perturbation_heatmap.py")
    print(f"Outputs    : {pdf_path.relative_to(REPO_ROOT)}")
    print(f"             {png_path.relative_to(REPO_ROOT)} (400 dpi)")
    print()
    print(f"Datasets plotted     : {', '.join(DATASET_LABELS[d] for d in DATASET_ORDER)}")
    print(f"Perturbations plotted : {', '.join(PERT_LABELS[p] for p in PERT_ORDER)}")
    print(f"Severity mapping      : low -> 'Low severity' (Panel A), "
          f"high -> 'High severity' (Panel B)")
    print(f"Cell metric           : delta_mAP50 (fraction in CSV) x 100 = percentage points")
    print()
    print(f"Shared colour scale   : vmin = {vmin:+.2f} pp, vcenter = 0.00 pp, "
          f"vmax = {vmax:+.2f} pp  (TwoSlopeNorm, cmap='vlag_r')")

    pos = df[df["delta_pp"] > 0]
    print()
    if len(pos):
        print(f"Positive deltas present: YES ({len(pos)} cells, preserved as-is)")
        print(
            pos.assign(delta_pp=pos["delta_pp"].round(2))[
                ["dataset", "perturbation_type", "severity", "delta_mAP50", "delta_pp"]
            ].to_string(index=False)
        )
    else:
        print("Positive deltas present: NO")

    print()
    print("-" * 78)
    print("VERIFICATION TABLE — values actually plotted (percentage points, 1 dp)")
    print("-" * 78)
    print("\nPanel A — Low severity")
    print(piv_low.round(1).to_string())
    print("\nPanel B — High severity")
    print(piv_high.round(1).to_string())

    print()
    print("-" * 78)
    print("Long-form audit (rounded to 2 dp)")
    print("-" * 78)
    audit = (
        df[["severity", "perturbation_type", "dataset", "delta_mAP50", "delta_pp"]]
        .assign(
            severity=lambda d: pd.Categorical(d["severity"], SEVERITY_ORDER, ordered=True),
            perturbation_type=lambda d: pd.Categorical(d["perturbation_type"], PERT_ORDER, ordered=True),
            dataset=lambda d: pd.Categorical(d["dataset"], DATASET_ORDER, ordered=True),
        )
        .sort_values(["severity", "perturbation_type", "dataset"])
        .reset_index(drop=True)
    )
    audit["delta_pp"] = audit["delta_pp"].round(2)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
