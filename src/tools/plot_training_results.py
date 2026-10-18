"""
Cattle Vision Framework — Training Results Visualizer
RF-DETR-Seg Hyperparameter Comparison: Config A vs Config B
Texas State University — Master's Thesis, Sakif Khan

Usage:
    python plot_training_results.py

    Place config_A_log.txt and config_B_log.txt in the same directory.
    All charts will be saved to ./plots/
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
LOG_A = "config_A_log.txt"  # Medium, lr=1e-4
LOG_B = "config_B_log.txt"  # Medium, lr=5e-5
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Color palette ───────────────────────────
C_A = "#2563EB"  # Config A: blue
C_A_LIGHT = "#93C5FD"
C_B = "#DC2626"  # Config B: red
C_B_LIGHT = "#FCA5A5"
C_EMA_A = "#1D4ED8"  # EMA variants (darker)
C_EMA_B = "#991B1B"
BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"
TEXT_COLOR = "#1E293B"
ACCENT = "#F59E0B"

# ─── Typography ──────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": BG,
        "figure.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CBD5E1",
    }
)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_log(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def extract(records, key, default=None):
    return [r.get(key, default) for r in records]


def extract_nested(records, outer, inner, default=None):
    results = []
    for r in records:
        obj = r.get(outer, {})
        if isinstance(obj, dict):
            results.append(obj.get(inner, default))
        else:
            results.append(default)
    return results


def extract_map(records, source="ema_test_results_json"):
    """Extract mAP@50 and mAP@50:95 from results_json field."""
    map50, map5095 = [], []
    for r in records:
        obj = r.get(source, {})
        if obj:
            cm = obj.get("class_map", [])
            cattle = next((c for c in cm if c["class"] == "cattle"), None)
            if cattle:
                map50.append(cattle["map@50"] * 100)
                map5095.append(cattle["map@50:95"] * 100)
            else:
                map50.append(None)
                map5095.append(None)
        else:
            map50.append(None)
            map5095.append(None)
    return map50, map5095


def extract_prf(records, source="ema_test_results_json"):
    precision, recall, f1 = [], [], []
    for r in records:
        obj = r.get(source, {})
        if obj:
            cm = obj.get("class_map", [])
            cattle = next((c for c in cm if c["class"] == "cattle"), None)
            if cattle:
                precision.append(cattle["precision"] * 100)
                recall.append(cattle["recall"] * 100)
                f1.append(cattle["f1_score"] * 100)
            else:
                precision.append(None)
                recall.append(None)
                f1.append(None)
        else:
            precision.append(None)
            recall.append(None)
            f1.append(None)
    return precision, recall, f1


print("Loading logs...")
recs_A = load_log(LOG_A)
recs_B = load_log(LOG_B)
epochs_A = [r["epoch"] + 1 for r in recs_A]  # 1-indexed for display
epochs_B = [r["epoch"] + 1 for r in recs_B]

print(f"  Config A: {len(recs_A)} epochs  ({epochs_A})")
print(f"  Config B: {len(recs_B)} epochs  ({epochs_B})")

# ─── Loss data ───────────────────────────────
train_loss_A = extract(recs_A, "train_loss")
test_loss_A = extract(recs_A, "test_loss")
ema_loss_A = extract(recs_A, "ema_test_loss")

train_loss_B = extract(recs_B, "train_loss")
test_loss_B = extract(recs_B, "test_loss")
ema_loss_B = extract(recs_B, "ema_test_loss")

# ─── Component losses ────────────────────────
components = {
    "CE Loss": ("train_loss_ce", "test_loss_ce"),
    "BBox Loss": ("train_loss_bbox", "test_loss_bbox"),
    "GIoU Loss": ("train_loss_giou", "test_loss_giou"),
    "Mask CE Loss": ("train_loss_mask_ce", "test_loss_mask_ce"),
    "Mask Dice Loss": ("train_loss_mask_dice", "test_loss_mask_dice"),
}

# ─── Detection mAP ───────────────────────────
det_map50_A, det_map5095_A = extract_map(recs_A, "test_results_json")
det_map50_A_ema, det_map5095_A_ema = extract_map(recs_A, "ema_test_results_json")
det_map50_B, det_map5095_B = extract_map(recs_B, "test_results_json")
det_map50_B_ema, det_map5095_B_ema = extract_map(recs_B, "ema_test_results_json")

# ─── Segmentation mAP ────────────────────────
seg_map50_A, seg_map5095_A = extract_map(recs_A, "test_results_json_masks")
seg_map50_A_ema, seg_map5095_A_ema = extract_map(recs_A, "ema_test_results_json_masks")
seg_map50_B, seg_map5095_B = extract_map(recs_B, "test_results_json_masks")
seg_map50_B_ema, seg_map5095_B_ema = extract_map(recs_B, "ema_test_results_json_masks")

# ─── Precision / Recall / F1 ─────────────────
prec_A, rec_A, f1_A = extract_prf(recs_A, "ema_test_results_json")
prec_B, rec_B, f1_B = extract_prf(recs_B, "ema_test_results_json")

seg_prec_A, seg_rec_A, seg_f1_A = extract_prf(recs_A, "ema_test_results_json_masks")
seg_prec_B, seg_rec_B, seg_f1_B = extract_prf(recs_B, "ema_test_results_json_masks")

# ─── Epoch timing ────────────────────────────
train_times_A = extract(recs_A, "train_epoch_time")
train_times_B = extract(recs_B, "train_epoch_time")


def parse_time_seconds(t):
    if t is None:
        return None
    parts = t.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


train_sec_A = [parse_time_seconds(t) / 60 for t in train_times_A]  # minutes
train_sec_B = [parse_time_seconds(t) / 60 for t in train_times_B]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def style_ax(ax, title, xlabel="Epoch", ylabel=None, legend=True):
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    if legend:
        ax.legend(loc="best")


def annotate_last(ax, x, y, color, fmt="{:.2f}"):
    if y and y[-1] is not None:
        ax.annotate(
            fmt.format(y[-1]),
            xy=(x[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="bold",
            va="center",
        )


def add_best_marker(ax, epochs, values, color, label="best"):
    valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
    if not valid:
        return
    best_e, best_v = max(valid, key=lambda x: x[1])
    ax.scatter(
        [best_e],
        [best_v],
        s=120,
        color=color,
        zorder=5,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
    )
    ax.annotate(
        f" ★ {best_v:.2f}%",
        xy=(best_e, best_v),
        fontsize=8,
        color=color,
        fontweight="bold",
        xytext=(5, 3),
        textcoords="offset points",
    )


# ─────────────────────────────────────────────
#  CHART 1 — Total Loss Overview (Train + Val)
# ─────────────────────────────────────────────
print("Plotting Chart 1: Total Loss Overview...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle(
    "Training & Validation Loss — Config A vs Config B",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

for ax, epochs, tr_A, te_A, em_A, tr_B, te_B, em_B, title in [
    (
        axes[0],
        epochs_A,
        train_loss_A,
        test_loss_A,
        ema_loss_A,
        train_loss_B,
        test_loss_B,
        ema_loss_B,
        "All Epochs",
    ),
]:
    ax.plot(epochs_A, train_loss_A, color=C_A, lw=2, marker="o", ms=5, label="A Train")
    ax.plot(
        epochs_A,
        test_loss_A,
        color=C_A,
        lw=2,
        marker="s",
        ms=5,
        linestyle="--",
        label="A Val",
    )
    ax.plot(epochs_B, train_loss_B, color=C_B, lw=2, marker="o", ms=5, label="B Train")
    ax.plot(
        epochs_B,
        test_loss_B,
        color=C_B,
        lw=2,
        marker="s",
        ms=5,
        linestyle="--",
        label="B Val",
    )
    ax.fill_between(epochs_A, train_loss_A, test_loss_A, alpha=0.06, color=C_A)
    ax.fill_between(epochs_B, train_loss_B, test_loss_B, alpha=0.06, color=C_B)
    style_ax(ax, "Total Loss per Epoch", ylabel="Loss")
    annotate_last(ax, epochs_A, train_loss_A, C_A)
    annotate_last(ax, epochs_B, train_loss_B, C_B)

# Right panel: EMA val loss comparison only
axes[1].plot(
    epochs_A,
    ema_loss_A,
    color=C_EMA_A,
    lw=2.5,
    marker="D",
    ms=5,
    label="A — EMA Val Loss",
)
axes[1].plot(
    epochs_B,
    ema_loss_B,
    color=C_EMA_B,
    lw=2.5,
    marker="D",
    ms=5,
    label="B — EMA Val Loss",
)
axes[1].fill_between(epochs_A, ema_loss_A, alpha=0.12, color=C_A)
axes[1].fill_between(epochs_B, ema_loss_B, alpha=0.12, color=C_B)
style_ax(axes[1], "EMA Validation Loss Comparison", ylabel="EMA Loss")
annotate_last(axes[1], epochs_A, ema_loss_A, C_EMA_A)
annotate_last(axes[1], epochs_B, ema_loss_B, C_EMA_B)

# Config labels
for ax in axes:
    ax.text(
        0.02,
        0.97,
        "Config A: RF-DETR-Seg-Medium  lr=1e-4\nConfig B: RF-DETR-Seg-Medium  lr=5e-5",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.8
        ),
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_total_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_total_loss.png")


# ─────────────────────────────────────────────
#  CHART 2 — Component Loss Breakdown
# ─────────────────────────────────────────────
print("Plotting Chart 2: Component Loss Breakdown...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Loss Component Breakdown — Train vs Validation", fontsize=15, fontweight="bold"
)

comp_list = list(components.items())

for idx, (name, (train_key, test_key)) in enumerate(comp_list):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    tr_A = extract(recs_A, train_key)
    te_A = extract(recs_A, test_key)
    tr_B = extract(recs_B, train_key)
    te_B = extract(recs_B, test_key)

    ax.plot(epochs_A, tr_A, color=C_A, lw=2, marker="o", ms=4, label="A Train")
    ax.plot(
        epochs_A,
        te_A,
        color=C_A,
        lw=1.5,
        marker="s",
        ms=4,
        linestyle="--",
        alpha=0.7,
        label="A Val",
    )
    ax.plot(epochs_B, tr_B, color=C_B, lw=2, marker="o", ms=4, label="B Train")
    ax.plot(
        epochs_B,
        te_B,
        color=C_B,
        lw=1.5,
        marker="s",
        ms=4,
        linestyle="--",
        alpha=0.7,
        label="B Val",
    )

    style_ax(ax, name, ylabel="Loss")
    annotate_last(ax, epochs_A, tr_A, C_A)
    annotate_last(ax, epochs_B, tr_B, C_B)

# Hide unused subplot
axes[1][2].set_visible(False)

# Add unified legend in the hidden panel space
legend_ax = axes[1][2]
legend_ax.set_visible(True)
legend_ax.axis("off")
handles = [
    mpatches.Patch(color=C_A, label="Config A — lr=1e-4"),
    mpatches.Patch(color=C_B, label="Config B — lr=5e-5"),
    plt.Line2D([0], [0], color="gray", lw=2, label="Train"),
    plt.Line2D([0], [0], color="gray", lw=1.5, linestyle="--", label="Validation"),
]
legend_ax.legend(
    handles=handles,
    loc="center",
    fontsize=10,
    title="Legend",
    title_fontsize=11,
    frameon=True,
    edgecolor="#CBD5E1",
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_component_losses.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_component_losses.png")


# ─────────────────────────────────────────────
#  CHART 3 — Detection mAP (Box)
# ─────────────────────────────────────────────
print("Plotting Chart 3: Detection mAP...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Object Detection Performance — Bounding Box mAP", fontsize=15, fontweight="bold"
)

# mAP@50
ax = axes[0]
ax.plot(
    epochs_A,
    det_map50_A,
    color=C_A,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="A Regular",
)
ax.plot(
    epochs_A,
    det_map50_A_ema,
    color=C_EMA_A,
    lw=2.5,
    marker="D",
    ms=6,
    label="A EMA (best model)",
)
ax.plot(
    epochs_B,
    det_map50_B,
    color=C_B,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="B Regular",
)
ax.plot(
    epochs_B,
    det_map50_B_ema,
    color=C_EMA_B,
    lw=2.5,
    marker="D",
    ms=6,
    label="B EMA (best model)",
)
ax.fill_between(epochs_A, det_map50_A, det_map50_A_ema, alpha=0.08, color=C_A)
ax.fill_between(epochs_B, det_map50_B, det_map50_B_ema, alpha=0.08, color=C_B)
add_best_marker(ax, epochs_A, det_map50_A_ema, C_EMA_A)
add_best_marker(ax, epochs_B, det_map50_B_ema, C_EMA_B)
ax.set_ylim(90, 100)
style_ax(ax, "Detection mAP@50 (%)", ylabel="mAP@50 (%)")

# mAP@50:95
ax = axes[1]
ax.plot(
    epochs_A,
    det_map5095_A,
    color=C_A,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="A Regular",
)
ax.plot(
    epochs_A,
    det_map5095_A_ema,
    color=C_EMA_A,
    lw=2.5,
    marker="D",
    ms=6,
    label="A EMA (best model)",
)
ax.plot(
    epochs_B,
    det_map5095_B,
    color=C_B,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="B Regular",
)
ax.plot(
    epochs_B,
    det_map5095_B_ema,
    color=C_EMA_B,
    lw=2.5,
    marker="D",
    ms=6,
    label="B EMA (best model)",
)
ax.fill_between(epochs_A, det_map5095_A, det_map5095_A_ema, alpha=0.08, color=C_A)
ax.fill_between(epochs_B, det_map5095_B, det_map5095_B_ema, alpha=0.08, color=C_B)
add_best_marker(ax, epochs_A, det_map5095_A_ema, C_EMA_A)
add_best_marker(ax, epochs_B, det_map5095_B_ema, C_EMA_B)
style_ax(ax, "Detection mAP@50:95 (%)", ylabel="mAP@50:95 (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_detection_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_detection_map.png")


# ─────────────────────────────────────────────
#  CHART 4 — Segmentation mAP (Mask)
# ─────────────────────────────────────────────
print("Plotting Chart 4: Segmentation mAP...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Instance Segmentation Performance — Mask mAP", fontsize=15, fontweight="bold"
)

ax = axes[0]
ax.plot(
    epochs_A,
    seg_map50_A,
    color=C_A,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="A Regular",
)
ax.plot(
    epochs_A,
    seg_map50_A_ema,
    color=C_EMA_A,
    lw=2.5,
    marker="D",
    ms=6,
    label="A EMA (best model)",
)
ax.plot(
    epochs_B,
    seg_map50_B,
    color=C_B,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="B Regular",
)
ax.plot(
    epochs_B,
    seg_map50_B_ema,
    color=C_EMA_B,
    lw=2.5,
    marker="D",
    ms=6,
    label="B EMA (best model)",
)
ax.fill_between(epochs_A, seg_map50_A, seg_map50_A_ema, alpha=0.08, color=C_A)
ax.fill_between(epochs_B, seg_map50_B, seg_map50_B_ema, alpha=0.08, color=C_B)
add_best_marker(ax, epochs_A, seg_map50_A_ema, C_EMA_A)
add_best_marker(ax, epochs_B, seg_map50_B_ema, C_EMA_B)
ax.set_ylim(90, 100)
style_ax(ax, "Segmentation mAP@50 (%)", ylabel="Mask mAP@50 (%)")

ax = axes[1]
ax.plot(
    epochs_A,
    seg_map5095_A,
    color=C_A,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="A Regular",
)
ax.plot(
    epochs_A,
    seg_map5095_A_ema,
    color=C_EMA_A,
    lw=2.5,
    marker="D",
    ms=6,
    label="A EMA (best model)",
)
ax.plot(
    epochs_B,
    seg_map5095_B,
    color=C_B,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    alpha=0.6,
    label="B Regular",
)
ax.plot(
    epochs_B,
    seg_map5095_B_ema,
    color=C_EMA_B,
    lw=2.5,
    marker="D",
    ms=6,
    label="B EMA (best model)",
)
ax.fill_between(epochs_A, seg_map5095_A, seg_map5095_A_ema, alpha=0.08, color=C_A)
ax.fill_between(epochs_B, seg_map5095_B, seg_map5095_B_ema, alpha=0.08, color=C_B)
add_best_marker(ax, epochs_A, seg_map5095_A_ema, C_EMA_A)
add_best_marker(ax, epochs_B, seg_map5095_B_ema, C_EMA_B)
style_ax(ax, "Segmentation mAP@50:95 (%)", ylabel="Mask mAP@50:95 (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_segmentation_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_segmentation_map.png")


# ─────────────────────────────────────────────
#  CHART 5 — Precision / Recall / F1
# ─────────────────────────────────────────────
print("Plotting Chart 5: Precision / Recall / F1...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Precision, Recall & F1 Score — Detection vs Segmentation (EMA Model)",
    fontsize=14,
    fontweight="bold",
)

metric_sets = [
    ("Detection Precision (%)", prec_A, prec_B),
    ("Detection Recall (%)", rec_A, rec_B),
    ("Detection F1 Score (%)", f1_A, f1_B),
    ("Segmentation Precision (%)", seg_prec_A, seg_prec_B),
    ("Segmentation Recall (%)", seg_rec_A, seg_rec_B),
    ("Segmentation F1 Score (%)", seg_f1_A, seg_f1_B),
]

for idx, (title, vals_A, vals_B) in enumerate(metric_sets):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    ax.plot(
        epochs_A,
        vals_A,
        color=C_EMA_A,
        lw=2.5,
        marker="D",
        ms=5,
        label="Config A (lr=1e-4)",
    )
    ax.plot(
        epochs_B,
        vals_B,
        color=C_EMA_B,
        lw=2.5,
        marker="D",
        ms=5,
        label="Config B (lr=5e-5)",
    )
    ax.fill_between(epochs_A, vals_A, alpha=0.1, color=C_A)
    ax.fill_between(epochs_B, vals_B, alpha=0.1, color=C_B)
    add_best_marker(ax, epochs_A, vals_A, C_EMA_A)
    add_best_marker(ax, epochs_B, vals_B, C_EMA_B)
    ax.set_ylim(85, 100)
    style_ax(ax, title, ylabel="%")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_precision_recall_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_precision_recall_f1.png")


# ─────────────────────────────────────────────
#  CHART 6 — Config A vs B Head-to-Head Bar
# ─────────────────────────────────────────────
print("Plotting Chart 6: Config Comparison Bar Chart...")

# Use last epoch EMA values
metrics_labels = [
    "Det mAP@50",
    "Det mAP@50:95",
    "Seg mAP@50",
    "Seg mAP@50:95",
    "Precision",
    "Recall",
    "F1 Score",
]

last_A = [
    det_map50_A_ema[-1],
    det_map5095_A_ema[-1],
    seg_map50_A_ema[-1],
    seg_map5095_A_ema[-1],
    prec_A[-1],
    rec_A[-1],
    f1_A[-1],
]
last_B = [
    det_map50_B_ema[-1],
    det_map5095_B_ema[-1],
    seg_map50_B_ema[-1],
    seg_map5095_B_ema[-1],
    prec_B[-1],
    rec_B[-1],
    f1_B[-1],
]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars_A = ax.bar(
    x - width / 2,
    last_A,
    width,
    color=C_A,
    label=f"Config A  lr=1e-4  (Epoch {epochs_A[-1]})",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)
bars_B = ax.bar(
    x + width / 2,
    last_B,
    width,
    color=C_B,
    label=f"Config B  lr=5e-5  (Epoch {epochs_B[-1]})",
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
)

# Value labels on bars
for bar in bars_A:
    h = bar.get_height()
    if h:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_EMA_A,
            fontweight="bold",
        )
for bar in bars_B:
    h = bar.get_height()
    if h:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_EMA_B,
            fontweight="bold",
        )

# Highlight winner for each metric
for i, (a, b) in enumerate(zip(last_A, last_B)):
    if a and b:
        winner_x = (x[i] - width / 2) if a >= b else (x[i] + width / 2)
        ax.annotate(
            "▲", xy=(winner_x, max(a, b) + 1.2), ha="center", fontsize=9, color=ACCENT
        )

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, rotation=15, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(70, 103)
ax.set_title(
    "Config A vs Config B — Latest Epoch EMA Metrics (▲ = winner per metric)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_config_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_config_comparison_bar.png")


# ─────────────────────────────────────────────
#  CHART 7 — EMA vs Regular Model Gain
# ─────────────────────────────────────────────
print("Plotting Chart 7: EMA vs Regular Model Gain...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "EMA Model vs Regular Model — mAP Improvement per Epoch",
    fontsize=14,
    fontweight="bold",
)

for ax, (map50_reg, map50_ema, map5095_reg, map5095_ema, epochs, color, label) in zip(
    axes,
    [
        (
            det_map50_A,
            det_map50_A_ema,
            det_map5095_A,
            det_map5095_A_ema,
            epochs_A,
            C_A,
            "Config A",
        ),
        (
            det_map50_B,
            det_map50_B_ema,
            det_map5095_B,
            det_map5095_B_ema,
            epochs_B,
            C_B,
            "Config B",
        ),
    ],
):
    gain50 = [e - r for e, r in zip(map50_ema, map50_reg) if e and r]
    gain5095 = [e - r for e, r in zip(map5095_ema, map5095_reg) if e and r]
    ep_valid = [e for e, r in zip(epochs, map50_reg) if r]

    bars1 = ax.bar(
        [e - 0.18 for e in ep_valid],
        gain50,
        width=0.35,
        color=color,
        alpha=0.8,
        label="mAP@50 gain",
        edgecolor="white",
    )
    bars2 = ax.bar(
        [e + 0.18 for e in ep_valid],
        gain5095,
        width=0.35,
        color=color,
        alpha=0.4,
        label="mAP@50:95 gain",
        edgecolor="white",
        hatch="//",
    )

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"+{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT_COLOR,
        )

    ax.axhline(0, color="#94A3B8", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EMA Gain (percentage points)")
    ax.set_title(f"{label} — EMA Benefit over Regular Model", fontweight="bold", pad=8)
    ax.set_xticks(ep_valid)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_ema_vs_regular_gain.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 07_ema_vs_regular_gain.png")


# ─────────────────────────────────────────────
#  CHART 8 — Epoch Training Time
# ─────────────────────────────────────────────
print("Plotting Chart 8: Epoch Training Time...")
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    epochs_A,
    train_sec_A,
    color=C_A,
    lw=2.5,
    marker="o",
    ms=7,
    label="Config A  lr=1e-4",
)
ax.plot(
    epochs_B,
    train_sec_B,
    color=C_B,
    lw=2.5,
    marker="o",
    ms=7,
    label="Config B  lr=5e-5",
)

for e, t in zip(epochs_A, train_sec_A):
    if t:
        ax.annotate(
            f"{t:.1f}m",
            xy=(e, t),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=C_A,
        )
for e, t in zip(epochs_B, train_sec_B):
    if t:
        ax.annotate(
            f"{t:.1f}m",
            xy=(e, t),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=C_B,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Time (minutes)")
ax.set_title("Training Time per Epoch", fontweight="bold", pad=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Projected total
avg_A = np.mean([t for t in train_sec_A if t])
avg_B = np.mean([t for t in train_sec_B if t])
proj_A = avg_A * 100 / 60
proj_B = avg_B * 100 / 60
ax.text(
    0.98,
    0.97,
    f"Projected 100-epoch total:\n"
    f"  Config A: ~{proj_A:.0f}h\n"
    f"  Config B: ~{proj_B:.0f}h",
    transform=ax.transAxes,
    fontsize=8.5,
    va="top",
    ha="right",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.9
    ),
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_epoch_training_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 08_epoch_training_time.png")


# ─────────────────────────────────────────────
#  CHART 9 — Comprehensive Dashboard
# ─────────────────────────────────────────────
print("Plotting Chart 9: Comprehensive Dashboard...")
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("white")
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

fig.text(
    0.5,
    0.98,
    "RF-DETR-Seg Hyperparameter Comparison — Training Dashboard",
    ha="center",
    va="top",
    fontsize=17,
    fontweight="bold",
    color=TEXT_COLOR,
)
fig.text(
    0.5,
    0.955,
    "Texas State University · Cattle Vision Framework · Master's Thesis",
    ha="center",
    va="top",
    fontsize=10,
    color="#64748B",
)

# ── Panel 1: Total Loss ───────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(epochs_A, train_loss_A, color=C_A, lw=2, marker="o", ms=4, label="A Train")
ax1.plot(
    epochs_A,
    test_loss_A,
    color=C_A,
    lw=1.5,
    linestyle="--",
    marker="s",
    ms=4,
    alpha=0.7,
    label="A Val",
)
ax1.plot(epochs_B, train_loss_B, color=C_B, lw=2, marker="o", ms=4, label="B Train")
ax1.plot(
    epochs_B,
    test_loss_B,
    color=C_B,
    lw=1.5,
    linestyle="--",
    marker="s",
    ms=4,
    alpha=0.7,
    label="B Val",
)
ax1.fill_between(epochs_A, train_loss_A, test_loss_A, alpha=0.05, color=C_A)
ax1.fill_between(epochs_B, train_loss_B, test_loss_B, alpha=0.05, color=C_B)
style_ax(ax1, "Total Loss", ylabel="Loss")

# ── Panel 2: Detection mAP@50 ────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(
    epochs_A, det_map50_A_ema, color=C_EMA_A, lw=2, marker="D", ms=5, label="A EMA"
)
ax2.plot(
    epochs_B, det_map50_B_ema, color=C_EMA_B, lw=2, marker="D", ms=5, label="B EMA"
)
ax2.set_ylim(90, 100)
style_ax(ax2, "Det mAP@50 (%)", ylabel="%")

# ── Panel 3: Seg mAP@50:95 ───────────────────
ax3 = fig.add_subplot(gs[0, 3])
ax3.plot(
    epochs_A, seg_map5095_A_ema, color=C_EMA_A, lw=2, marker="D", ms=5, label="A EMA"
)
ax3.plot(
    epochs_B, seg_map5095_B_ema, color=C_EMA_B, lw=2, marker="D", ms=5, label="B EMA"
)
style_ax(ax3, "Seg mAP@50:95 (%)", ylabel="%")

# ── Panel 4: CE + GIoU Loss ──────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(
    epochs_A,
    extract(recs_A, "train_loss_ce"),
    color=C_A,
    lw=2,
    marker="o",
    ms=4,
    label="A",
)
ax4.plot(
    epochs_B,
    extract(recs_B, "train_loss_ce"),
    color=C_B,
    lw=2,
    marker="o",
    ms=4,
    label="B",
)
style_ax(ax4, "Train CE Loss", ylabel="Loss")

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(
    epochs_A,
    extract(recs_A, "train_loss_giou"),
    color=C_A,
    lw=2,
    marker="o",
    ms=4,
    label="A",
)
ax5.plot(
    epochs_B,
    extract(recs_B, "train_loss_giou"),
    color=C_B,
    lw=2,
    marker="o",
    ms=4,
    label="B",
)
style_ax(ax5, "Train GIoU Loss", ylabel="Loss")

# ── Panel 5: Precision & Recall ──────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(epochs_A, prec_A, color=C_EMA_A, lw=2, marker="D", ms=5, label="A Precision")
ax6.plot(
    epochs_A,
    rec_A,
    color=C_EMA_A,
    lw=2,
    marker="s",
    ms=5,
    linestyle="--",
    label="A Recall",
)
ax6.plot(epochs_B, prec_B, color=C_EMA_B, lw=2, marker="D", ms=5, label="B Precision")
ax6.plot(
    epochs_B,
    rec_B,
    color=C_EMA_B,
    lw=2,
    marker="s",
    ms=5,
    linestyle="--",
    label="B Recall",
)
ax6.set_ylim(85, 100)
style_ax(ax6, "Detection Precision & Recall (%)", ylabel="%")

# ── Panel 6: F1 ──────────────────────────────
ax7 = fig.add_subplot(gs[1, 3])
ax7.plot(epochs_A, f1_A, color=C_EMA_A, lw=2.5, marker="D", ms=5, label="A Det F1")
ax7.plot(
    epochs_A,
    seg_f1_A,
    color=C_A,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    label="A Seg F1",
)
ax7.plot(epochs_B, f1_B, color=C_EMA_B, lw=2.5, marker="D", ms=5, label="B Det F1")
ax7.plot(
    epochs_B,
    seg_f1_B,
    color=C_B,
    lw=2,
    marker="o",
    ms=5,
    linestyle="--",
    label="B Seg F1",
)
ax7.set_ylim(85, 100)
style_ax(ax7, "F1 Score — Det vs Seg (%)", ylabel="%")

# ── Panel 7: Metrics Table ───────────────────
ax8 = fig.add_subplot(gs[2, :])
ax8.axis("off")

table_data = []
for rec_list, col, config_name in [
    (recs_A, C_A, "Config A  lr=1e-4"),
    (recs_B, C_B, "Config B  lr=5e-5"),
]:
    for r in rec_list:
        ep = r.get("epoch", "?") + 1
        ema = r.get("ema_test_results_json", {})
        ema_mask = r.get("ema_test_results_json_masks", {})
        if not ema:
            continue
        cm = {c["class"]: c for c in ema.get("class_map", [])}
        cm_mask = {c["class"]: c for c in ema_mask.get("class_map", [])}
        cattle = cm.get("cattle", {})
        cattle_mask = cm_mask.get("cattle", {})
        table_data.append(
            [
                config_name,
                str(ep),
                f"{cattle.get('map@50', 0)*100:.2f}%",
                f"{cattle.get('map@50:95', 0)*100:.2f}%",
                f"{cattle_mask.get('map@50', 0)*100:.2f}%",
                f"{cattle_mask.get('map@50:95', 0)*100:.2f}%",
                f"{cattle.get('precision', 0)*100:.2f}%",
                f"{cattle.get('recall', 0)*100:.2f}%",
                f"{cattle.get('f1_score', 0)*100:.2f}%",
            ]
        )

col_labels = [
    "Config",
    "Epoch",
    "Det mAP@50",
    "Det mAP@50:95",
    "Seg mAP@50",
    "Seg mAP@50:95",
    "Precision",
    "Recall",
    "F1",
]

tbl = ax8.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# Style header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#1E293B")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# Color rows by config
for i, row in enumerate(table_data):
    cfg_color = C_A_LIGHT if "A" in row[0] else C_B_LIGHT
    for j in range(len(col_labels)):
        tbl[(i + 1, j)].set_facecolor(cfg_color)

ax8.set_title(
    "Full Metrics Table — EMA Model (all epochs)", fontweight="bold", pad=8, fontsize=11
)

plt.savefig(f"{OUTPUT_DIR}/09_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 09_dashboard.png")


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
print("\n✅ All charts saved to ./plots/")
print("─" * 45)
files = sorted(os.listdir(OUTPUT_DIR))
for f in files:
    path = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f:45s} {size:6.1f} KB")
print("\nCharts summary:")
print("  01_total_loss             — Train vs Val loss, both configs")
print("  02_component_losses       — CE, BBox, GIoU, Mask CE, Mask Dice")
print("  03_detection_map          — Bounding box mAP@50 and mAP@50:95")
print("  04_segmentation_map       — Mask mAP@50 and mAP@50:95")
print("  05_precision_recall_f1    — P/R/F1 for detection & segmentation")
print("  06_config_comparison_bar  — Side-by-side bar chart, latest epoch")
print("  07_ema_vs_regular_gain    — How much EMA improves over regular model")
print("  08_epoch_training_time    — Minutes per epoch + projected total")
print("  09_dashboard              — Full dashboard + metrics table")
