"""
Parse HiPE1 v2 training log files into structured CSVs.

Log format (per epoch):
  Epoch   N | lr=X.XXe+00 | train_loss=A train_f1=B | val_loss=C val_macro_f1=D
           per-class val F1: Standing=E  Lying=F  Foraging=G  Drinking=H  Ruminating=I  Grooming=J  Other=K

Output schema matches v1 CSVs exactly:
  epoch, train_loss, val_loss, val_macro_f1,
  val_f1_standing, val_f1_lying, val_f1_foraging, val_f1_drinking,
  val_f1_ruminating, val_f1_grooming, val_f1_other, lr

Log-to-CSV mapping:
  logs/cbvd5_v2.log           -> results/behavior/training_logs/videomae_cbvd5_v2.csv
  logs/cvb_v2.log             -> results/behavior/training_logs/videomae_cvb_v2.csv
  logs/combined_v2.log        -> results/behavior/training_logs/videomae_combined_v2.csv
  logs/cbvd5_to_cvb_v2.log    -> results/behavior/training_logs/videomae_cbvd5_to_cvb_v2.csv
  logs/cvb_to_cbvd5_v2.log    -> results/behavior/training_logs/videomae_cvb_to_cbvd5_v2.csv

Run:
  python scripts/parse_v2_logs.py
"""

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LOG_MAP = [
    ("logs/cbvd5_v2.log",        "results/behavior/training_logs/videomae_cbvd5_v2.csv"),
    ("logs/cvb_v2.log",          "results/behavior/training_logs/videomae_cvb_v2.csv"),
    ("logs/combined_v2.log",     "results/behavior/training_logs/videomae_combined_v2.csv"),
    ("logs/cbvd5_to_cvb_v2.log", "results/behavior/training_logs/videomae_cbvd5_to_cvb_v2.csv"),
    ("logs/cvb_to_cbvd5_v2.log", "results/behavior/training_logs/videomae_cvb_to_cbvd5_v2.csv"),
]

COLUMNS = [
    "epoch", "train_loss", "val_loss", "val_macro_f1",
    "val_f1_standing", "val_f1_lying", "val_f1_foraging", "val_f1_drinking",
    "val_f1_ruminating", "val_f1_grooming", "val_f1_other", "lr",
]

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+lr=([\deE+\-.]+)\s+\|\s+"
    r"train_loss=([\d.]+)\s+train_f1=[\d.]+\s+\|\s+"
    r"val_loss=([\d.]+)\s+val_macro_f1=([\d.]+)"
)
CLASS_RE = re.compile(
    r"per-class val F1:\s+"
    r"Standing=([\d.]+)\s+Lying=([\d.]+)\s+Foraging=([\d.]+)\s+"
    r"Drinking=([\d.]+)\s+Ruminating=([\d.]+)\s+Grooming=([\d.]+)\s+Other=([\d.]+)"
)


def parse_log(log_path: Path):
    rows = []
    pending = None
    for line in log_path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            pending = {
                "epoch":        int(m.group(1)),
                "lr":           float(m.group(2)),
                "train_loss":   float(m.group(3)),
                "val_loss":     float(m.group(4)),
                "val_macro_f1": float(m.group(5)),
            }
            continue
        if pending is not None:
            c = CLASS_RE.search(line)
            if c:
                pending.update({
                    "val_f1_standing":   float(c.group(1)),
                    "val_f1_lying":      float(c.group(2)),
                    "val_f1_foraging":   float(c.group(3)),
                    "val_f1_drinking":   float(c.group(4)),
                    "val_f1_ruminating": float(c.group(5)),
                    "val_f1_grooming":   float(c.group(6)),
                    "val_f1_other":      float(c.group(7)),
                })
                rows.append(pending)
                pending = None
    return rows


def main():
    for log_rel, csv_rel in LOG_MAP:
        log_path = REPO_ROOT / log_rel
        csv_path = REPO_ROOT / csv_rel

        if not log_path.exists():
            print(f"SKIP {log_rel}  (not found)")
            continue

        rows = parse_log(log_path)
        if not rows:
            print(f"WARN {log_rel}  (no epochs parsed)")
            continue

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        best = max(rows, key=lambda r: r["val_macro_f1"])
        print(
            f"  {csv_path.relative_to(REPO_ROOT)}  "
            f"({len(rows)} epochs, best val_macro_f1={best['val_macro_f1']:.4f} @ epoch {best['epoch']})"
        )


if __name__ == "__main__":
    main()
