"""
src/tools/eval_detection_ood.py

Evaluates the fine-tuned RF-DETR cattle detector on OOD datasets.
Runs model.predict() on every image in a COCO-format directory and computes
standard COCO detection metrics (mAP@50, mAP@50:95, AR@100) via pycocotools.

Intended for Phase 8 additional-dataset evaluation:
    - OpenCows2020  → results/detection/opencows2020_eval.json
    - Cows2021      → results/detection/cows2021_eval.json

Note: tracking evaluation (IDF1) is NOT supported here.
      Cows2021 detection_and_localisation annotations carry no cow IDs.

Usage:
    # OpenCows2020
    python src/tools/eval_detection_ood.py \
        --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \
        --dataset_dir data/processed/detection/opencows2020 \
        --output results/detection/opencows2020_eval.json \
        --dataset_name opencows2020

    # Cows2021 (test split)
    python src/tools/eval_detection_ood.py \
        --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \
        --dataset_dir data/processed/detection/cows2021/test \
        --output results/detection/cows2021_eval.json \
        --dataset_name cows2021

    # Lower threshold for OOD (default 0.3)
    python src/tools/eval_detection_ood.py ... --threshold 0.2
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint: str, seg: bool = False):
    try:
        from rfdetr import RFDETRMedium, RFDETRSegMedium
    except ImportError:
        print("[ERROR] rfdetr is not installed. Run: pip install rfdetr")
        sys.exit(1)

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    model_cls = RFDETRSegMedium if seg else RFDETRMedium
    print(f"[eval] Loading {model_cls.__name__} from {ckpt.name} ...")
    model = model_cls(pretrain_weights=str(ckpt))
    return model


# ── Inference + prediction collection ────────────────────────────────────────

def collect_predictions(model, dataset_dir: Path, threshold: float) -> tuple[list, int, int]:
    """
    Run model.predict() on all images listed in _annotations.coco.json.

    Returns:
        predictions: list of COCO-format prediction dicts
        n_processed: number of images successfully inferred
        n_detections: total detections produced
    """
    from PIL import Image

    ann_path = dataset_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"_annotations.coco.json not found in {dataset_dir}")

    with open(ann_path) as f:
        gt_data = json.load(f)

    n_images = len(gt_data["images"])
    print(f"[eval] {n_images} images in {dataset_dir}")

    predictions = []
    n_processed = 0
    n_detections = 0

    for img_info in tqdm(gt_data["images"], desc="inference"):
        img_path = dataset_dir / img_info["file_name"]

        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Cap input size so the seg model's mask path doesn't OOM on high-res images.
        # The detection backbone resizes to ~432–576 px internally, so this is lossless.
        # Predictions are scaled back to original coordinates for correct COCO eval.
        _MAX_SIDE = 1120
        coord_scale = 1.0
        if max(img_w, img_h) > _MAX_SIDE:
            coord_scale = max(img_w, img_h) / _MAX_SIDE
            image = image.resize(
                (int(img_w / coord_scale), int(img_h / coord_scale)), Image.BILINEAR
            )

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
                x1, y1, x2, y2 = [float(v) * coord_scale for v in dets.xyxy[i]]
                # Clamp to original image bounds
                x1 = max(0.0, x1); y1 = max(0.0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
                bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
            else:
                bbox = [0, 0, img_w, img_h]

            predictions.append({
                "image_id":    img_info["id"],
                "category_id": 1,
                "bbox":        bbox,
                "score":       round(score, 4),
            })
            n_detections += 1

    print(f"[eval] Processed {n_processed}/{n_images} images, {n_detections} detections")
    return predictions, n_processed, n_detections


# ── COCO mAP evaluation ───────────────────────────────────────────────────────

def run_coco_eval(dataset_dir: Path, predictions: list) -> dict:
    """Run pycocotools bbox evaluation against the COCO annotation file."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = dataset_dir / "_annotations.coco.json"
    coco_gt = COCO(str(ann_path))

    if not predictions:
        print("[eval] WARNING: no predictions — returning zero metrics")
        return {}

    # pycocotools needs predictions written to a temp file to use loadRes
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(predictions, tf)
        tmp_path = tf.name

    coco_dt = coco_gt.loadRes(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]

    return {
        "mAP":       round(float(s[0]), 4),
        "mAP50":     round(float(s[1]), 4),
        "mAP75":     round(float(s[2]), 4),
        "mAP_s":     round(float(s[3]), 4),
        "mAP_m":     round(float(s[4]), 4),
        "mAP_l":     round(float(s[5]), 4),
        "AR1":       round(float(s[6]), 4),
        "AR10":      round(float(s[7]), 4),
        "AR100":     round(float(s[8]), 4),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RF-DETR OOD detection evaluation")
    p.add_argument("--checkpoint",
                   default="runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth",
                   help="Path to RF-DETR checkpoint (.pth)")
    p.add_argument("--dataset_dir",
                   required=True,
                   help="Dir containing _annotations.coco.json and image files/symlinks")
    p.add_argument("--output",
                   required=True,
                   help="Output JSON path (e.g. results/detection/opencows2020_eval.json)")
    p.add_argument("--dataset_name",
                   default=None,
                   help="Human-readable dataset name stored in the result JSON")
    p.add_argument("--threshold",
                   type=float,
                   default=0.3,
                   help="Detection confidence threshold (default: 0.3)")
    p.add_argument("--seg",
                   action="store_true",
                   help="Use RFDETRSegMedium instead of RFDETRMedium (for seg checkpoints)")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or dataset_dir.name

    model = load_model(args.checkpoint, seg=args.seg)

    predictions, n_processed, n_detections = collect_predictions(
        model, dataset_dir, args.threshold
    )

    print("[eval] Running pycocotools mAP evaluation ...")
    metrics = run_coco_eval(dataset_dir, predictions)

    result = {
        "dataset":     dataset_name,
        "checkpoint":  str(args.checkpoint),
        "threshold":   args.threshold,
        "n_images":    n_processed,
        "n_detections": n_detections,
        **metrics,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n── Results: {dataset_name} ──────────────────")
    print(f"  mAP@50     : {metrics.get('mAP50', 0):.4f}  ({metrics.get('mAP50', 0)*100:.1f}%)")
    print(f"  mAP@50:95  : {metrics.get('mAP', 0):.4f}   ({metrics.get('mAP', 0)*100:.1f}%)")
    print(f"  AR@100     : {metrics.get('AR100', 0):.4f}")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
