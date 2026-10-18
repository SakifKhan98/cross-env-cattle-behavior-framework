import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from behavior.dataset import TubeletDataset, LABEL_NAMES

try:
    from sklearn.metrics import f1_score
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> nn.Module:
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_name"])
    model.classifier = nn.Linear(768, cfg["num_classes"])
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    return model


def build_optimizer(model: nn.Module, cfg: dict):
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr"]},
            {"params": head_params,     "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["num_epochs"]    * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
              grad_accum, num_classes, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()
    for step, (videos, labels) in enumerate(loader):
        # dataset returns [B,3,16,224,224]; VideoMAE expects pixel_values [B,T,C,H,W]
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)

        with autocast("cuda", enabled=(scaler is not None)):
            out = model(pixel_values=videos)
            loss = criterion(out.logits, labels) / grad_accum

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds = out.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    # flush leftover gradients when steps not divisible by grad_accum
    if train and (len(loader) % grad_accum != 0):
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(len(loader), 1)
    label_list = list(range(num_classes))
    macro_f1    = f1_score(all_labels, all_preds, average="macro",  labels=label_list, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,   labels=label_list, zero_division=0)
    return avg_loss, macro_f1, per_class_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    num_classes = cfg["num_classes"]
    labels_csv  = cfg["labels_csv"]

    train_ds = TubeletDataset(
        labels_csv,
        dataset_filter=cfg["train"].get("dataset_filter"),
        split_filter=cfg["train"].get("split_filter"),
        label_subset=cfg["train"].get("label_subset"),
    )
    val_ds = TubeletDataset(
        labels_csv,
        dataset_filter=cfg["val"].get("dataset_filter"),
        split_filter=cfg["val"].get("split_filter"),
        label_subset=cfg["val"].get("label_subset"),
    )
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    num_workers  = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

    model = build_model(cfg).to(device)

    if cfg.get("use_class_weights", True):
        weights   = train_ds.class_weights(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer  = build_optimizer(model, cfg)
    grad_accum = cfg.get("grad_accum_steps", 1)
    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    scheduler  = build_scheduler(optimizer, cfg, steps_per_epoch)

    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    # --- log.csv setup ---
    class_names   = [LABEL_NAMES.get(i, str(i)).lower() for i in range(num_classes)]
    per_class_cols = [f"val_f1_{n}" for n in class_names]

    log_path   = output_dir / "log.csv"
    log_exists = log_path.exists()
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_macro_f1"] + per_class_cols + ["lr"]
        )

    best_f1 = -1.0
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 999)

    for epoch in range(1, cfg["num_epochs"] + 1):
        # read backbone LR at start of epoch (before any scheduler steps this epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_f1, _ = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, grad_accum, num_classes, train=True,
        )
        with torch.no_grad():
            val_loss, val_f1, val_per_class_f1 = run_epoch(
                model, val_loader, criterion, optimizer, scheduler, scaler,
                device, grad_accum, num_classes, train=False,
            )

        # --- console ---
        per_class_str = "  ".join(
            f"{LABEL_NAMES.get(i,str(i))}={val_per_class_f1[i]:.3f}"
            for i in range(num_classes)
        )
        log(
            f"Epoch {epoch:3d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}"
        )
        log(f"         per-class val F1: {per_class_str}")

        # --- log.csv row ---
        log_writer.writerow(
            [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_f1:.6f}"]
            + [f"{v:.6f}" for v in val_per_class_f1]
            + [f"{current_lr:.8e}"]
        )
        log_file.flush()

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "val_macro_f1": val_f1},
            output_dir / "checkpoint_last.pt",
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_macro_f1": val_f1},
                output_dir / "checkpoint_best.pt",
            )
            log(f"  -> new best val_macro_f1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    log_file.close()
    log(f"Done. Best val_macro_f1={best_f1:.4f}. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
