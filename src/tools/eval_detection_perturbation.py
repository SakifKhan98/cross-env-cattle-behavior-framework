"""
src/tools/eval_detection_perturbation.py

Perturbation robustness evaluation for the RF-DETR cattle detector.

Applies five classes of synthetic image transforms at two severity levels to each
OOD dataset test split, runs RF-DETR inference in-memory on the perturbed images,
and writes one intermediate result JSON per (dataset, perturbation_type, severity)
condition to results/generalization/perturbation_runs/.

Clean baselines are read from the already-committed OOD eval JSONs in
results/detection/ — clean inference is never re-run.

Perturbation types and severity parameters
------------------------------------------
Type            | Low                              | High
----------------|----------------------------------|-----------------------------------
brightness      | brightness_limit = -0.50 (→50%)  | brightness_limit = -0.75 (→25%)
gaussian_noise  | σ = 25                           | σ = 50
motion_blur     | kernel = 7                       | kernel = 15
fog             | fog_coef = 0.3                   | fog_coef = 0.6
rain            | slant±10, length=20, width=1     | slant±20, length=40, width=2

Usage (single condition):
    python src/tools/eval_detection_perturbation.py \\
        --dataset_dir data/processed/detection/opencows2020 \\
        --dataset_name opencows2020 \\
        --clean_baseline results/detection/opencows2020_eval.json \\
        --perturbation brightness --severity low

Usage (all 10 conditions for one dataset):
    python src/tools/eval_detection_perturbation.py \\
        --dataset_dir data/processed/detection/opencows2020 \\
        --dataset_name opencows2020 \\
        --clean_baseline results/detection/opencows2020_eval.json \\
        --all
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Perturbation transform layer ──────────────────────────────────────────────

PERTURBATION_TYPES = ["brightness", "gaussian_noise", "motion_blur", "fog", "rain"]
SEVERITIES = ["low", "high"]


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def make_transform(perturbation: str, severity: str):
    """Return a callable: PIL Image → perturbed PIL Image."""
    import albumentations as A

    if perturbation == "brightness":
        limit = -0.50 if severity == "low" else -0.75
        aug = A.RandomBrightnessContrast(
            brightness_limit=(limit, limit), contrast_limit=0, p=1.0
        )
    elif perturbation == "gaussian_noise":
        sigma = 25 if severity == "low" else 50
        std = sigma / 255.0
        aug = A.GaussNoise(std_range=(std, std), p=1.0)
    elif perturbation == "motion_blur":
        k = 7 if severity == "low" else 15
        aug = A.MotionBlur(blur_limit=(k, k), p=1.0)
    elif perturbation == "fog":
        coef = 0.3 if severity == "low" else 0.6
        aug = A.RandomFog(fog_coef_range=(coef, coef), p=1.0)
    elif perturbation == "rain":
        if severity == "low":
            aug = A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1, p=1.0)
        else:
            aug = A.RandomRain(slant_range=(-20, 20), drop_length=40, drop_width=2, p=1.0)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

    def apply(pil_img: Image.Image) -> Image.Image:
        arr = _pil_to_np(pil_img)
        result = aug(image=arr)["image"]
        return _np_to_pil(result)

    return apply


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint: str):
    try:
        from rfdetr import RFDETRMedium
    except ImportError:
        print("[ERROR] rfdetr is not installed.")
        sys.exit(1)

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"[perturb] Loading RF-DETR from {ckpt.name} ...")
    return RFDETRMedium(pretrained_weights=str(ckpt))


# ── Inference + COCO eval (with transform hook) ───────────────────────────────

def collect_predictions(model, dataset_dir: Path, threshold: float, transform=None):
    ann_path = dataset_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"_annotations.coco.json not found in {dataset_dir}")

    with open(ann_path) as f:
        gt_data = json.load(f)

    predictions = []
    n_processed = 0

    for img_info in tqdm(gt_data["images"], desc="inference", leave=False):
        img_path = dataset_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        if transform is not None:
            image = transform(image)
        img_w, img_h = image.size

        try:
            dets = model.predict(image, threshold=threshold)
        except Exception as e:
            print(f"  [WARN] {img_info['file_name']}: {e}")
            continue

        n_processed += 1

        if dets is None or len(dets) == 0:
            continue

        for i in range(len(dets)):
            score = float(dets.confidence[i]) if dets.confidence is not None else 1.0
            if dets.xyxy is not None:
                x1, y1, x2, y2 = [float(v) for v in dets.xyxy[i]]
                x1 = max(0.0, x1); y1 = max(0.0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
            else:
                bbox = [0, 0, img_w, img_h]
            predictions.append({
                "image_id": img_info["id"],
                "category_id": 1,
                "bbox": bbox,
                "score": round(score, 4),
            })

    return predictions, n_processed


def run_coco_eval(dataset_dir: Path, predictions: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = dataset_dir / "_annotations.coco.json"
    coco_gt = COCO(str(ann_path))

    if not predictions:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0, "AR100": 0.0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(predictions, tf)
        tmp_path = tf.name

    coco_dt = coco_gt.loadRes(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats

    return {
        "mAP":   round(float(s[0]), 4),
        "mAP50": round(float(s[1]), 4),
        "mAP75": round(float(s[2]), 4),
        "AR100": round(float(s[8]), 4),
    }


# ── Single condition runner ───────────────────────────────────────────────────

def run_condition(
    model,
    dataset_dir: Path,
    dataset_name: str,
    clean_baseline: dict,
    perturbation: str,
    severity: str,
    threshold: float,
    out_dir: Path,
) -> dict:
    label = f"{dataset_name}_{perturbation}_{severity}"
    out_path = out_dir / f"{label}.json"

    if out_path.exists():
        print(f"[perturb] Skipping {label} (already exists)")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n[perturb] Running {label} ...")
    transform = make_transform(perturbation, severity)
    predictions, n_images = collect_predictions(model, dataset_dir, threshold, transform)

    print(f"[perturb] Running COCO eval for {label} ...")
    metrics = run_coco_eval(dataset_dir, predictions)

    result = {
        "dataset":          dataset_name,
        "perturbation_type": perturbation,
        "severity":         severity,
        "n_images":         n_images,
        "mAP50_clean":      clean_baseline["mAP50"],
        "mAP50_perturbed":  metrics["mAP50"],
        "delta_mAP50":      round(metrics["mAP50"] - clean_baseline["mAP50"], 4),
        "mAP_clean":        clean_baseline["mAP"],
        "mAP_perturbed":    metrics["mAP"],
        "AR100_clean":      clean_baseline["AR100"],
        "AR100_perturbed":  metrics["AR100"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  mAP50: {clean_baseline['mAP50']:.4f} → {metrics['mAP50']:.4f}  "
          f"(delta {result['delta_mAP50']:+.4f})")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RF-DETR perturbation robustness evaluation")
    p.add_argument("--checkpoint",
                   default="runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--clean_baseline", required=True,
                   help="Path to existing OOD eval JSON (e.g. results/detection/opencows2020_eval.json)")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--out_dir",
                   default="results/generalization/perturbation_runs")
    p.add_argument("--perturbation", choices=PERTURBATION_TYPES,
                   help="Single perturbation type (ignored when --all is set)")
    p.add_argument("--severity", choices=SEVERITIES,
                   help="Single severity level (ignored when --all is set)")
    p.add_argument("--all", dest="run_all", action="store_true",
                   help="Run all 10 (perturbation × severity) conditions for this dataset")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)

    with open(args.clean_baseline) as f:
        clean_baseline = json.load(f)

    model = load_model(args.checkpoint)

    if args.run_all:
        conditions = [(p, s) for p in PERTURBATION_TYPES for s in SEVERITIES]
    else:
        if not args.perturbation or not args.severity:
            print("[ERROR] Provide --perturbation and --severity, or use --all")
            sys.exit(1)
        conditions = [(args.perturbation, args.severity)]

    results = []
    for perturbation, severity in conditions:
        r = run_condition(
            model, dataset_dir, args.dataset_name, clean_baseline,
            perturbation, severity, args.threshold, out_dir,
        )
        results.append(r)

    # Summary table
    print(f"\n{'─'*70}")
    print(f"  {'Dataset':<16} {'Perturbation':<16} {'Sev':<5} {'Clean':>7} {'Perturb':>8} {'Delta':>7}")
    print(f"{'─'*70}")
    for r in results:
        print(f"  {r['dataset']:<16} {r['perturbation_type']:<16} {r['severity']:<5} "
              f"{r['mAP50_clean']:>7.4f} {r['mAP50_perturbed']:>8.4f} {r['delta_mAP50']:>+7.4f}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
