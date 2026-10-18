"""
train_hipe2.py — RF-DETR-Seg-Large, lr=5e-5 (Larger model, conservative LR)
Run on HiPE2 using both V100s via PyTorch DDP.

Rationale for this config vs HiPE1:
  - Seg-Large has 36.2M params vs 35.7M for Medium, but runs at 504×504
    resolution vs 432×432 — captures finer mask boundaries on large cattle
  - lr=5e-5 is a conservative fine-tuning rate, often outperforms 1e-4
    when starting from strong pretrained weights (less risk of catastrophic
    forgetting of COCO features)
  - Together these form the strongest expected config for thesis results

Usage (inside Docker container):
    python3 train_hipe2.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RUN_NAME = "seg_large_lr5e5_conservative"
DATASET_DIR = "/workspace/data/rfdetr_seg/cattle"
OUTPUT_DIR = f"/workspace/runs/{RUN_NAME}"
EPOCHS = int(os.environ.get("EPOCHS", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 6))    # V100 default=6; use 2-4 for local RTX 3060 smoke test
GRAD_ACCUM = 2  # effective batch = 24 (close enough to 16-32 sweet spot)
LR = 5e-5
CHECKPOINT = None  # None = start from COCO pretrained weights


# ── Validate dataset ──────────────────────────────────────────────────────────
def validate_dataset(dataset_dir):
    for split in ["train", "valid"]:
        ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
        assert ann_path.exists(), f"Missing: {ann_path}"
        with open(ann_path) as f:
            d = json.load(f)
        n_img = len(d["images"])
        n_ann = len(d["annotations"])
        print(f"  [{split}] {n_img:,} images, {n_ann:,} annotations")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"RF-DETR-Seg Training — {RUN_NAME}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # GPU info
    import torch

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory // 1024**3}GB")
    print()

    # Dataset check
    print("Dataset validation:")
    validate_dataset(DATASET_DIR)

    # Output dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility
    config = {
        "run_name": RUN_NAME,
        "model": "RFDETRSegLarge",
        "dataset_dir": DATASET_DIR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "n_gpus": n_gpus,
        "started": datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_DIR}/run_config.json")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    from rfdetr import RFDETRSegLarge

    model = RFDETRSegLarge()

    # run_test=False: RF-DETR auto-sets dataset_file="roboflow" when dataset_dir
    # is provided, which makes it look for a test/ split. We only have train/valid,
    # so disable the redundant test eval pass (valid/ is already evaluated each epoch).
    train_kwargs = dict(
        dataset_dir=DATASET_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LR,
        output_dir=OUTPUT_DIR,
        use_ema=True,
        run_test=False,
    )
    start = time.time()
    model.train(**train_kwargs)
    elapsed = (time.time() - start) / 3600

    print()
    print("=" * 60)
    print(f"Training complete — {elapsed:.1f} hours")
    print(f"Best checkpoint: {OUTPUT_DIR}/checkpoint_best_total.pth")
    print("=" * 60)

    # Save completion marker
    config["completed"] = datetime.now().isoformat()
    config["runtime_hours"] = round(elapsed, 2)
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
