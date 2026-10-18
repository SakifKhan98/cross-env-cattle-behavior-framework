"""
src/tools/plot_training_curves.py

PURPOSE:
    Parse metrics_log.json files saved during RF-DETR-Seg training and
    generate publication-quality figures for the thesis.

FIGURES PRODUCED:
    1. training_loss.png         — train loss per epoch, both runs overlaid
    2. validation_map.png        — val mAP@50 and mAP@50:95 per epoch
    3. learning_rate.png         — LR schedule per epoch
    4. combined_dashboard.png    — all metrics in one 2×2 figure (thesis main fig)
    5. comparison_table.csv      — best epoch stats for each run (thesis table)

USAGE:
    python src/tools/plot_training_curves.py \\
        --runs \\
            runs/segmentation/seg_medium_lr1e4_baseline \\
            runs/segmentation/seg_large_lr5e5_conservative \\
        --output_dir results/rfdetr_seg/comparison
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]  # blue, orange, green, purple
LABELS = {
    "seg_medium_lr1e4_baseline": "Seg-Medium  lr=1e-4",
    "seg_large_lr5e5_conservative": "Seg-Large   lr=5e-5",
}


# ── Metrics parser ────────────────────────────────────────────────────────────


def load_metrics(run_dir: str) -> dict:
    """
    Load per-epoch metrics from a run directory.
    Tries metrics_log.json first, then falls back to parsing the training log.

    Returns dict with keys: epochs, train_loss, val_map50, val_map5095, lr
    """
    run_path = Path(run_dir)
    run_name = run_path.name

    # Try structured JSON first (written by train_hipe1.py / train_hipe2.py)
    json_path = run_path / "metrics_log.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        print(f"  [{run_name}] Loaded metrics_log.json ({len(data['epochs'])} epochs)")
        return data

    # Fall back to parsing the training log file
    log_candidates = list(run_path.glob("*.log")) + list(run_path.glob("../logs/*.log"))
    if not log_candidates:
        print(f"  [{run_name}] WARNING: No metrics_log.json or .log file found")
        return None

    log_path = log_candidates[0]
    print(f"  [{run_name}] Parsing log file: {log_path.name}")
    return parse_log_file(str(log_path), run_name)


def parse_log_file(log_path: str, run_name: str) -> dict:
    """
    Parse RF-DETR training log output into structured metrics.
    RF-DETR logs lines like:
        Epoch [42/100]  train_loss: 0.4321  val_loss: 0.5012  mAP@50: 0.7234  mAP@50:95: 0.4891  lr: 1.00e-04
    """
    epochs = []
    train_loss = []
    val_loss = []
    map50 = []
    map5095 = []
    lr_vals = []

    # Patterns to try — RF-DETR may format slightly differently across versions
    epoch_pat = re.compile(r"[Ee]poch\s*[\[#]?(\d+)")
    tloss_pat = re.compile(r"train[_\s]loss[:\s]+([0-9.]+)")
    vloss_pat = re.compile(r"val[_\s]loss[:\s]+([0-9.]+)")
    map50_pat = re.compile(r"mAP[@_]?50(?::95)?(?!\S)[:\s]+([0-9.]+)")
    map5095_pat = re.compile(r"mAP[@_]?50:95[:\s]+([0-9.]+)")
    lr_pat = re.compile(r"\blr[:\s]+([0-9.e+\-]+)")

    with open(log_path) as f:
        for line in f:
            ep_m = epoch_pat.search(line)
            if not ep_m:
                continue

            ep = int(ep_m.group(1))
            tl = (
                float(tloss_pat.search(line).group(1))
                if tloss_pat.search(line)
                else None
            )
            vl = (
                float(vloss_pat.search(line).group(1))
                if vloss_pat.search(line)
                else None
            )

            # mAP@50:95 must be matched before mAP@50 (more specific pattern first)
            m5095 = (
                float(map5095_pat.search(line).group(1))
                if map5095_pat.search(line)
                else None
            )
            m50_m = map50_pat.search(line)
            m50 = float(m50_m.group(1)) if m50_m else None
            lr_m = lr_pat.search(line)
            lr = float(lr_m.group(1)) if lr_m else None

            if tl is not None or m50 is not None:
                epochs.append(ep)
                train_loss.append(tl)
                val_loss.append(vl)
                map50.append(m50)
                map5095.append(m5095)
                lr_vals.append(lr)

    if not epochs:
        print(f"  WARNING: Could not parse any epoch metrics from {log_path}")
        return None

    print(f"  [{run_name}] Parsed {len(epochs)} epochs from log")
    return {
        "run_name": run_name,
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "map50": map50,
        "map5095": map5095,
        "lr": lr_vals,
    }


def best_epoch(metrics: dict) -> dict:
    """Find the epoch with highest val mAP@50."""
    map50 = [v for v in metrics["map50"] if v is not None]
    if not map50:
        return {}
    best_idx = map50.index(max(map50))
    ep = metrics["epochs"][best_idx]
    return {
        "run_name": metrics["run_name"],
        "best_epoch": ep,
        "train_loss": metrics["train_loss"][best_idx],
        "val_loss": metrics.get("val_loss", [None] * len(metrics["epochs"]))[best_idx],
        "map50": metrics["map50"][best_idx],
        "map5095": metrics["map5095"][best_idx] if metrics.get("map5095") else None,
        "total_epochs": max(metrics["epochs"]),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_metric(ax, all_metrics, key, ylabel, title, ymin=None):
    for i, (metrics, label) in enumerate(all_metrics):
        vals = [v for v in metrics.get(key, []) if v is not None]
        eps = [
            e for e, v in zip(metrics["epochs"], metrics.get(key, [])) if v is not None
        ]
        if vals:
            ax.plot(eps, vals, color=COLORS[i], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.legend()


def generate_plots(all_metrics: list, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labeled = [(m, LABELS.get(m["run_name"], m["run_name"])) for m in all_metrics]

    # ── 1. Training loss ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric(
        ax, labeled, "train_loss", "Training Loss", "Training Loss per Epoch", ymin=0
    )
    fig.tight_layout()
    fig.savefig(out / "training_loss.png")
    plt.close(fig)
    print(f"  Saved: training_loss.png")

    # ── 2. Validation mAP ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_metric(axes[0], labeled, "map50", "mAP@50", "Validation mAP@50", ymin=0)
    plot_metric(
        axes[1], labeled, "map5095", "mAP@50:95", "Validation mAP@50:95", ymin=0
    )
    fig.tight_layout()
    fig.savefig(out / "validation_map.png")
    plt.close(fig)
    print(f"  Saved: validation_map.png")

    # ── 3. Learning rate schedule ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_metric(ax, labeled, "lr", "Learning Rate", "Learning Rate Schedule")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "learning_rate.png")
    plt.close(fig)
    print(f"  Saved: learning_rate.png")

    # ── 4. Combined dashboard (thesis main figure) ────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax_tl = fig.add_subplot(gs[0, 0])
    ax_vl = fig.add_subplot(gs[0, 1])
    ax_m50 = fig.add_subplot(gs[1, 0])
    ax_m95 = fig.add_subplot(gs[1, 1])

    plot_metric(ax_tl, labeled, "train_loss", "Loss", "Training Loss", ymin=0)
    plot_metric(ax_vl, labeled, "val_loss", "Loss", "Validation Loss", ymin=0)
    plot_metric(ax_m50, labeled, "map50", "mAP@50", "Val mAP@50", ymin=0)
    plot_metric(ax_m95, labeled, "map5095", "mAP@50:95", "Val mAP@50:95", ymin=0)

    fig.suptitle(
        "RF-DETR-Seg Fine-Tuning on Cattle — Hyperparameter Comparison",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out / "combined_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: combined_dashboard.png")

    # ── 5. Comparison table CSV ───────────────────────────────────────────────
    rows = [best_epoch(m) for m in all_metrics]

    # Also add pretrained COCO baseline row (from RF-DETR paper)
    rows.insert(
        0,
        {
            "run_name": "RF-DETR-Seg-M (COCO pretrained, no fine-tune)",
            "best_epoch": "—",
            "train_loss": "—",
            "val_loss": "—",
            "map50": 68.4,
            "map5095": 45.3,
            "total_epochs": "—",
        },
    )

    csv_lines = ["run_name,best_epoch,train_loss,val_loss,map50,map5095,total_epochs"]
    for r in rows:
        csv_lines.append(
            f"{r.get('run_name','')},{r.get('best_epoch','')},{r.get('train_loss','')},{r.get('val_loss','')},{r.get('map50','')},{r.get('map5095','')},{r.get('total_epochs','')}"
        )

    csv_path = out / "comparison_table.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"  Saved: comparison_table.csv")

    # Pretty print table
    print()
    print("  Thesis comparison table:")
    print(f"  {'Run':<45} {'mAP@50':>8} {'mAP@50:95':>10} {'BestEp':>7}")
    print("  " + "-" * 75)
    for r in rows:
        m50 = (
            f"{r['map50']:.1f}"
            if isinstance(r.get("map50"), float)
            else str(r.get("map50", "—"))
        )
        m5095 = (
            f"{r['map5095']:.1f}"
            if isinstance(r.get("map5095"), float)
            else str(r.get("map5095", "—"))
        )
        ep = str(r.get("best_epoch", "—"))
        name = LABELS.get(str(r.get("run_name", "")), str(r.get("run_name", "")))
        print(f"  {name:<45} {m50:>8} {m5095:>10} {ep:>7}")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Paths to run directories containing metrics_log.json or *.log",
    )
    p.add_argument("--output_dir", default="results/rfdetr_seg/comparison")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[plot] Loading metrics from {len(args.runs)} runs...")

    all_metrics = []
    for run_dir in args.runs:
        m = load_metrics(run_dir)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        print("[ERROR] No metrics loaded. Check run directory paths.")
        sys.exit(1)

    print(f"\n[plot] Generating figures → {args.output_dir}")
    generate_plots(all_metrics, args.output_dir)
    print(f"\n[plot] Done. Open: {args.output_dir}/combined_dashboard.png")


if __name__ == "__main__":
    main()
