"""
train_server.py — Parameterized RF-DETR-Seg training script for HiPE servers.

All configuration is read from environment variables so the same script runs
any HP combination without modification. Pass env vars via `docker run -e`.

Environment variables:
    MODEL       medium | large          (default: medium)
    LR          learning rate float     (default: 1e-4)
    BATCH_SIZE  per-GPU batch size      (default: 8 for medium, 6 for large)
    GRAD_ACCUM  gradient accum steps    (default: 1 for medium, 2 for large)
    EPOCHS      number of epochs        (default: 100)
    RUN_NAME    output directory name   (default: seg_{MODEL}_lr{LR})

Example — run Config C (Seg-Large, lr=1e-4) on GPU 0:

    docker run --rm \\
        --gpus '"device=0"' \\
        --shm-size=8g \\
        -v ~/cattle_seg/data:/workspace/data:ro \\
        -v ~/cattle_seg/runs:/workspace/runs \\
        -v ~/cattle_seg/scripts/train_server.py:/workspace/train_server.py:ro \\
        -e MODEL=large -e LR=1e-4 -e RUN_NAME=seg_large_lr1e4 \\
        cattle-rfdetr-seg:v1 \\
        train_server.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── Config from environment ────────────────────────────────────────────────────
MODEL      = os.environ.get("MODEL", "medium").lower()
LR         = float(os.environ.get("LR", "1e-4"))
EPOCHS     = int(os.environ.get("EPOCHS", "100"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1" if MODEL == "medium" else "2"))

# Default batch size depends on model: medium fits 8/GPU, large fits 6/GPU on V100 16GB
_default_bs = "4" if MODEL == "medium" else "4"  # V100 16GB: multi-scale 552 spikes memory; 4 is safe for both
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", _default_bs))

# Auto-generate run name from config if not provided
_lr_tag    = f"lr{str(LR).replace('0.', '').replace('-', 'e-')}"
RUN_NAME   = os.environ.get("RUN_NAME", f"seg_{MODEL}_{_lr_tag}")

DATASET_DIR = "/workspace/data/rfdetr_seg/cattle"
OUTPUT_DIR  = f"/workspace/runs/{RUN_NAME}"


# ── Validate dataset ───────────────────────────────────────────────────────────
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"RF-DETR-Seg Training — {RUN_NAME}")
    print(f"  model={MODEL}  lr={LR}  batch={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  epochs={EPOCHS}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory // 1024**3}GB")
    print()

    print("Dataset validation:")
    validate_dataset(DATASET_DIR)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    config = {
        "run_name":   RUN_NAME,
        "model":      f"RFDETRSeg{MODEL.capitalize()}",
        "lr":         LR,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs":     EPOCHS,
        "n_gpus":     n_gpus,
        "started":    datetime.now().isoformat(),
    }
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {OUTPUT_DIR}/run_config.json\n")

    # ── Load model ────────────────────────────────────────────────────────────
    if MODEL == "large":
        from rfdetr import RFDETRSegLarge
        model = RFDETRSegLarge()
    else:
        from rfdetr import RFDETRSegMedium
        model = RFDETRSegMedium()

    # run_test=False: RF-DETR auto-sets dataset_file="roboflow" when dataset_dir
    # is provided, which makes it look for a test/ split. We only have train/valid.
    train_kwargs = dict(
        dataset_dir      = DATASET_DIR,
        epochs           = EPOCHS,
        batch_size       = BATCH_SIZE,
        grad_accum_steps = GRAD_ACCUM,
        lr               = LR,
        output_dir       = OUTPUT_DIR,
        use_ema          = True,
        run_test         = False,
    )

    start = time.time()
    model.train(**train_kwargs)
    elapsed = (time.time() - start) / 3600

    print()
    print("=" * 60)
    print(f"Training complete — {elapsed:.1f} hours")
    print(f"Best checkpoint: {OUTPUT_DIR}/checkpoint_best_total.pth")
    print("=" * 60)

    config["completed"]     = datetime.now().isoformat()
    config["runtime_hours"] = round(elapsed, 2)
    with open(f"{OUTPUT_DIR}/run_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
