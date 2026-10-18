import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior.dataset import TubeletDataset, LABEL_NAMES

try:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, confusion_matrix,
    )
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError("matplotlib + numpy required")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_name"])
    model.classifier = nn.Linear(768, cfg["num_classes"])
    return model


def save_confusion_matrix(cm, class_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(len(class_names) + 2, len(class_names) + 2))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Confusion matrix saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="YAML config path")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--split",      required=False, default=None,
                        help="Override split (train/val/test). Defaults to val split in config.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    split          = args.split if args.split is not None else cfg["val"].get("split_filter", "val")
    dataset_filter = cfg["val"].get("dataset_filter")
    label_subset   = cfg["val"].get("label_subset")
    num_classes    = cfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    ds = TubeletDataset(
        cfg["labels_csv"],
        dataset_filter=dataset_filter,
        split_filter=split,
        label_subset=label_subset,
    )
    log(f"Eval split='{split}': {len(ds)} samples")
    if len(ds) == 0:
        log("WARNING: dataset is empty — check split / dataset_filter in config.")

    num_workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = build_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_f1    = ckpt.get("val_macro_f1", "?")
    log(f"Loaded checkpoint: epoch={ckpt_epoch}  saved_val_macro_f1={ckpt_f1}")

    model.eval()
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.permute(0, 2, 1, 3, 4).to(device)  # [B,T,C,H,W]
            out    = model(pixel_values=videos)
            logits = out.logits.cpu()                           # [B, num_classes]
            preds  = logits.argmax(dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_logits.append(logits.numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # [N, num_classes]

    # --- aggregate metrics ---
    label_list  = list(range(num_classes))
    class_names = [LABEL_NAMES.get(i, str(i)) for i in range(num_classes)]

    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        labels=label_list, zero_division=0)
    per_f1   = f1_score(all_labels, all_preds, average=None,
                        labels=label_list, zero_division=0)
    per_prec = precision_score(all_labels, all_preds, average=None,
                               labels=label_list, zero_division=0)
    per_rec  = recall_score(all_labels, all_preds, average=None,
                            labels=label_list, zero_division=0)

    log(f"\n{'Class':<15} {'F1':>6} {'Prec':>6} {'Recall':>6}")
    log("-" * 38)
    for i in range(num_classes):
        log(f"{class_names[i]:<15} {per_f1[i]:>6.4f} {per_prec[i]:>6.4f} {per_rec[i]:>6.4f}")
    log("-" * 38)
    log(f"{'Macro-F1':<15} {macro_f1:>6.4f}")

    # --- output paths ---
    exp_name     = cfg.get("experiment_name", Path(args.config).stem)
    results_root = Path("results/behavior")

    cm_path    = results_root / "confusion_matrices" / f"{exp_name}_{split}.png"
    csv_path   = results_root / "f1_per_class.csv"
    preds_path = results_root / "predictions" / f"{exp_name}_{split}.csv"

    # confusion matrix PNG
    cm = confusion_matrix(all_labels, all_preds, labels=label_list)
    save_confusion_matrix(cm, class_names, cm_path)

    # f1_per_class.csv — append / create
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = (
                ["experiment", "split", "macro_f1"]
                + [f"f1_{LABEL_NAMES.get(i, str(i)).lower()}"   for i in range(num_classes)]
                + [f"prec_{LABEL_NAMES.get(i, str(i)).lower()}" for i in range(num_classes)]
                + [f"rec_{LABEL_NAMES.get(i, str(i)).lower()}"  for i in range(num_classes)]
            )
            writer.writerow(header)
        row = (
            [exp_name, split, f"{macro_f1:.6f}"]
            + [f"{v:.6f}" for v in per_f1]
            + [f"{v:.6f}" for v in per_prec]
            + [f"{v:.6f}" for v in per_rec]
        )
        writer.writerow(row)
    log(f"Per-class metrics saved → {csv_path}")

    # predictions.csv — one row per tubelet, includes logits for Phase 7 overlap resolution
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    logit_cols = [f"logit_{i}" for i in range(num_classes)]
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "video_id", "tubelet_dir", "start_frame", "end_frame",
             "label_id", "pred_label_id"] + logit_cols
        )
        for seq_idx in range(len(ds.df)):
            meta    = ds.df.iloc[seq_idx]
            logit_v = all_logits[seq_idx]
            writer.writerow(
                [
                    meta["dataset"],
                    meta["video_id"],
                    meta["tubelet_dir"],
                    meta["start_frame"],
                    meta["end_frame"],
                    int(meta["label_id"]),
                    all_preds[seq_idx],
                ]
                + [f"{v:.6f}" for v in logit_v]
            )
    log(f"Per-tubelet predictions saved → {preds_path}")


if __name__ == "__main__":
    main()
