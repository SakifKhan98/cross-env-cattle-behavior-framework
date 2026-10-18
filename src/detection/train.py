"""
src/detection/train.py

RF-DETR cattle detector training script.

All hyperparameters are driven by a YAML config file.
Supports:
  - Resume from checkpoint (--resume)
  - Early stopping
  - TensorBoard logging
  - Sanity-check mode (--sanity: 2 epochs on mini dataset)

Usage:
    # Full training run
    python src/detection/train.py --config configs/detection/rfdetr_combined.yaml

    # Train on a specific dataset (overrides config)
    python src/detection/train.py --config configs/detection/rfdetr_cbvd5.yaml

    # Resume from a crashed run
    python src/detection/train.py --config configs/detection/rfdetr_combined.yaml --resume

    # Quick sanity check (2 epochs, mini dataset, confirms setup works)
    python src/detection/train.py --config configs/detection/rfdetr_combined.yaml --sanity
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

# ── Config loader ─────────────────────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """Load and validate YAML config file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    required = ["experiment_name", "model", "dataset", "training", "output"]
    for key in required:
        if key not in cfg:
            print(f"[ERROR] Missing required config key: '{key}'")
            sys.exit(1)

    return cfg


# ── Pre-flight checks ─────────────────────────────────────────────────────────


def preflight_checks(cfg: dict, resume: bool):
    """
    Verify everything is in place before training starts.
    Catches common mistakes early so you don't wait 10 minutes to hit an error.
    """
    errors = []
    warnings = []

    dataset_dir = Path(cfg["dataset"]["dataset_dir"])

    if not dataset_dir.exists():
        errors.append(f"dataset_dir not found: {dataset_dir}")
    else:
        for split in ["train", "valid"]:
            split_dir = dataset_dir / split
            ann_file = split_dir / "_annotations.coco.json"

            if not split_dir.exists():
                errors.append(f"Missing split folder: {split_dir}")
            elif not ann_file.exists():
                errors.append(f"Missing annotation file: {ann_file}")
            else:
                images = [
                    f
                    for f in split_dir.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
                if len(images) == 0:
                    errors.append(f"No images found in {split_dir}")
                else:
                    print(
                        f"  [OK] {split}: {len(images)} images, {ann_file.name} found"
                    )

    output_dir = Path(cfg["output"]["output_dir"])
    checkpoint = output_dir / "checkpoint.pth"
    if checkpoint.exists() and not resume:
        warnings.append(
            f"checkpoint.pth found in {output_dir} but --resume not set. "
            "Training will start from scratch. Use --resume to continue, "
            "or change experiment_name in the config."
        )

    for w in warnings:
        print(f"  [WARNING] {w}")

    if errors:
        print("\n[PREFLIGHT FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)


# ── Model builder ─────────────────────────────────────────────────────────────


def build_model(cfg: dict):
    """Instantiate the RF-DETR model specified in config."""
    try:
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium
    except ImportError:
        print("[ERROR] rfdetr is not installed. Run: pip install rfdetr")
        sys.exit(1)

    model_type = cfg["model"].get("type", "RFDETRMedium")
    num_classes = cfg["dataset"].get("num_classes", 1)

    model_map = {
        "RFDETRBase": RFDETRBase,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }

    if model_type not in model_map:
        print(f"[ERROR] Unknown model type: {model_type}")
        print(f"  Valid options: {list(model_map.keys())}")
        sys.exit(1)

    model = model_map[model_type](num_classes=num_classes)
    print(f"  [OK] Model: {model_type} | num_classes: {num_classes}")
    return model


# ── Augmentation loader ───────────────────────────────────────────────────────


def get_aug_config(cfg: dict):
    """
    Load augmentation preset or custom dict from config.

    Config options:
      augmentation:
        preset: AUG_CONSERVATIVE     <- named preset
      # OR
      augmentation:
        custom:                      <- custom albumentations dict
          HorizontalFlip: {p: 0.5}
    """
    aug_cfg = cfg.get("augmentation", {})
    if not aug_cfg:
        print("  [OK] Augmentation: disabled")
        return {}

    preset_name = aug_cfg.get("preset")
    custom = aug_cfg.get("custom")

    if preset_name:
        try:
            from rfdetr.datasets.aug_config import (
                AUG_AERIAL,
                AUG_AGGRESSIVE,
                AUG_CONSERVATIVE,
                AUG_INDUSTRIAL,
            )
        except ImportError:
            print("  [WARNING] Could not import RF-DETR aug presets. Disabled.")
            return {}

        preset_map = {
            "AUG_CONSERVATIVE": AUG_CONSERVATIVE,
            "AUG_AGGRESSIVE": AUG_AGGRESSIVE,
            "AUG_AERIAL": AUG_AERIAL,
            "AUG_INDUSTRIAL": AUG_INDUSTRIAL,
        }

        if preset_name not in preset_map:
            print(f"  [WARNING] Unknown aug preset '{preset_name}'. Disabled.")
            return {}

        print(f"  [OK] Augmentation: preset {preset_name}")
        return preset_map[preset_name]

    if custom:
        print(f"  [OK] Augmentation: custom ({len(custom)} transforms)")
        return custom

    print("  [OK] Augmentation: disabled")
    return {}


# ── Resume helper ─────────────────────────────────────────────────────────────


def find_resume_checkpoint(output_dir: Path):
    """
    Find the best checkpoint to resume from.
    Priority: checkpoint.pth (latest) > checkpoint_best_total.pth
    """
    for name in ("checkpoint.pth", "checkpoint_best_total.pth"):
        p = output_dir / name
        if p.exists():
            return str(p)
    return None


# ── Summary printer ───────────────────────────────────────────────────────────


def print_training_summary(cfg: dict, resume_path, sanity: bool):
    """Print a clear summary of what is about to run."""
    t = cfg["training"]
    out = cfg["output"]
    eff_batch = t["batch_size"] * t["grad_accum_steps"]

    print("\n" + "=" * 60)
    print(f"  Experiment : {cfg['experiment_name']}")
    print(f"  Model      : {cfg['model']['type']}")
    print(f"  Dataset    : {cfg['dataset']['dataset_dir']}")
    print(f"  Output     : {out['output_dir']}")
    print("-" * 60)
    epochs_display = 2 if sanity else t["epochs"]
    print(f"  Epochs     : {epochs_display}" + (" [SANITY MODE]" if sanity else ""))
    print(
        f"  Batch      : {t['batch_size']} x {t['grad_accum_steps']} "
        f"grad_accum = {eff_batch} effective"
    )
    print(f"  Resolution : {t['resolution']}")
    print(f"  LR         : {t['lr']}  |  LR encoder: {t['lr_encoder']}")
    if t.get("early_stopping"):
        print(
            f"  Early stop : patience={t['early_stopping_patience']}, "
            f"min_delta={t['early_stopping_min_delta']}"
        )
    if resume_path:
        print(f"  Resume     : {resume_path}")
    print("=" * 60 + "\n")


# ── Main training function ────────────────────────────────────────────────────


def train(cfg: dict, resume: bool, sanity: bool):
    """
    Build model and call RF-DETR train() with all parameters from config.

    RF-DETR saves these checkpoints automatically:
      checkpoint.pth              - latest epoch  (use for --resume)
      checkpoint_best_ema.pth     - best val mAP, EMA weights
      checkpoint_best_total.pth   - best val mAP, final inference weights
      checkpoint_<N>.pth          - periodic saves (every checkpoint_interval epochs)
    """
    t = cfg["training"]
    out = cfg["output"]

    output_dir = Path(out["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve resume checkpoint
    resume_path = None
    if resume:
        resume_path = find_resume_checkpoint(output_dir)
        if resume_path:
            print(f"  [OK] Resuming from: {resume_path}")
        else:
            print(
                "  [WARNING] --resume set but no checkpoint found. "
                "Starting from scratch."
            )

    model = build_model(cfg)
    aug_config = get_aug_config(cfg)

    # Sanity mode: use mini dataset and cap at 2 epochs
    dataset_dir = cfg["dataset"]["dataset_dir"]
    epochs = t["epochs"]

    if sanity:
        mini_dir = str(Path(dataset_dir).parent / (Path(dataset_dir).name + "_mini"))
        if Path(mini_dir).exists():
            dataset_dir = mini_dir
            print(f"  [SANITY] Using mini dataset: {dataset_dir}")
        else:
            print(f"  [SANITY] Mini dataset not found at {mini_dir}")
            print(
                f"           Run src/data/make_mini.py first, or using: {dataset_dir}"
            )
        epochs = 2

    print_training_summary(cfg, resume_path, sanity)

    # Build kwargs for RF-DETR train()
    train_kwargs = {
        "dataset_dir": dataset_dir,
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": t["batch_size"],
        "grad_accum_steps": t["grad_accum_steps"],
        "lr": t["lr"],
        "lr_encoder": t["lr_encoder"],
        "weight_decay": t.get("weight_decay", 1e-4),
        "resolution": t["resolution"],
        "use_ema": t.get("use_ema", True),
        "gradient_checkpointing": t.get("gradient_checkpointing", False),
        "tensorboard": out.get("tensorboard", True),
        "checkpoint_interval": out.get("checkpoint_interval", 10),
    }

    # Early stopping — skip in sanity mode (2 epochs is not enough to trigger it)
    if t.get("early_stopping", False) and not sanity:
        train_kwargs.update(
            {
                "early_stopping": True,
                "early_stopping_patience": t.get("early_stopping_patience", 15),
                "early_stopping_min_delta": t.get("early_stopping_min_delta", 0.005),
                "early_stopping_use_ema": t.get("early_stopping_use_ema", True),
            }
        )

    if resume_path:
        train_kwargs["resume"] = resume_path

    if aug_config:
        train_kwargs["aug_config"] = aug_config

    # Launch
    start_time = time.time()
    print("Starting training...\n")

    try:
        model.train(**train_kwargs)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")
        print(
            f"  Resume: python src/detection/train.py "
            f"--config {cfg.get('_config_path', '<config>')} --resume"
        )
        sys.exit(0)

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    print("\n" + "=" * 60)
    print(f"  Training complete in {h}h {m}m {s}s")
    print(f"  Checkpoints: {output_dir}")
    print(f"  Best model : {output_dir / 'checkpoint_best_total.pth'}")
    print(f"  TensorBoard: tensorboard --logdir {output_dir}")
    print("=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train RF-DETR cattle detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training on combined dataset
  python src/detection/train.py --config configs/detection/rfdetr_combined.yaml

  # Train on CBVD-5 only
  python src/detection/train.py --config configs/detection/rfdetr_cbvd5.yaml

  # Resume a crashed run
  python src/detection/train.py --config configs/detection/rfdetr_combined.yaml --resume

  # Sanity check (2 epochs — confirms everything works before full run)
  python src/detection/train.py --config configs/detection/rfdetr_combined.yaml --sanity
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/detection/rfdetr_combined.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run 2 epochs on mini dataset to verify setup before full training",
    )
    args = parser.parse_args()

    print(f"\nLoading config: {args.config}")
    cfg = load_config(args.config)
    cfg["_config_path"] = args.config  # stored for interrupt message

    print("\nRunning preflight checks...")
    preflight_checks(cfg, args.resume)

    train(cfg, resume=args.resume, sanity=args.sanity)


if __name__ == "__main__":
    main()
