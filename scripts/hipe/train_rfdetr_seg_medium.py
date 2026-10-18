"""
train_hipe1.py — RF-DETR-Seg-Medium, lr=1e-4 (Baseline)
Run on HiPE1 using both V100s via PyTorch DDP.

Usage (inside Docker container):
    python3 train_hipe1.py

Or via torchrun for explicit DDP:
    torchrun --nproc_per_node=2 train_hipe1.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RUN_NAME     = "seg_medium_lr1e4_baseline"
DATASET_DIR  = "/workspace/data/rfdetr_seg/cattle"
OUTPUT_DIR   = f"/workspace/runs/{RUN_NAME}"
EPOCHS       = int(os.environ.get("EPOCHS", 100))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 4))   # 4 per GPU on V100 16GB (multi-scale 552 spikes memory); use 2 for local smoke test
GRAD_ACCUM   = 2    # batch_size=4 × grad_accum=2 → effective batch=8
LR           = 1e-4
CHECKPOINT   = None  # None = start from COCO pretrained weights

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
        "run_name":    RUN_NAME,
        "model":       "RFDETRSegMedium",
        "dataset_dir": DATASET_DIR,
        "epochs":      EPOCHS,
        "batch_size":  BATCH_SIZE,
        "grad_accum":  GRAD_ACCUM,
        "lr":          LR,
        "n_gpus":      n_gpus,
        "started":     datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_DIR}/run_config.json")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    # run_test=False: RF-DETR auto-sets dataset_file="roboflow" when dataset_dir
    # is provided, which makes it look for a test/ split. We only have train/valid,
    # so disable the redundant test eval pass (valid/ is already evaluated each epoch).
    train_kwargs = dict(
        dataset_dir     = DATASET_DIR,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        grad_accum_steps= GRAD_ACCUM,
        lr              = LR,
        output_dir      = OUTPUT_DIR,
        use_ema         = True,
        run_test        = False,
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