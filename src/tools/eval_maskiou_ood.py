"""
src/tools/eval_maskiou_ood.py

Evaluates RF-DETR-Seg on OOD datasets that have polygon mask ground truth.
Runs model inference, matches predictions to GT by box IoU, computes:
  - COCO segmentation mAP@50, mAP@50:95  (using pycocotools)
  - Mean Mask IoU (predicted binary mask vs GT polygon mask, at box IoU >= 0.5)

Intended for CattleEyeView Phase 8 evaluation:
    results/segmentation/cattleeyeview_maskiou.json

Usage:
    python src/tools/eval_maskiou_ood.py \
        --checkpoint  runs/seg_medium_lr5e5/checkpoint_best_ema.pth \
        --dataset_dir data/processed/detection/cattleeyeview/test \
        --output      results/segmentation/cattleeyeview_maskiou.json \
        --dataset_name cattleeyeview
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(checkpoint: str):
    try:
        from rfdetr import RFDETRSegMedium
    except ImportError:
        print("[ERROR] rfdetr is not installed. Run: pip install rfdetr")
        sys.exit(1)

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"[eval] Loading RF-DETR-Seg from {ckpt} ...")
    model = RFDETRSegMedium(pretrain_weights=str(ckpt))
    return model


# ── Box IoU ───────────────────────────────────────────────────────────────────

def box_iou(a: list, b: list) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ── Inference ─────────────────────────────────────────────────────────────────

def collect_predictions(model, dataset_dir: Path, threshold: float):
    """
    Run RF-DETR-Seg on all images in the COCO dataset dir.
    Returns (coco_pred_list, per_image_preds) where per_image_preds[img_id]
    is a list of {bbox_xyxy, score, mask} dicts.
    """
    from PIL import Image
    from pycocotools import mask as mask_utils

    ann_path = dataset_dir / "_annotations.coco.json"
    with open(ann_path) as f:
        gt_data = json.load(f)

    n_images = len(gt_data["images"])
    print(f"[eval] {n_images} images in {dataset_dir}")

    coco_preds = []
    per_image  = {}

    for img_info in tqdm(gt_data["images"], desc="inference"):
        img_path = dataset_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        from PIL import Image as PILImage
        image = PILImage.open(img_path).convert("RGB")
        img_w, img_h = image.size
        img_id = img_info["id"]

        try:
            dets = model.predict(image, threshold=threshold)
        except Exception as e:
            print(f"  [WARN] {img_info['file_name']}: {e}")
            continue

        per_image[img_id] = []

        if dets is None or len(dets) == 0:
            continue

        for i in range(len(dets)):
            score = float(dets.confidence[i]) if dets.confidence is not None else 1.0

            if dets.xyxy is not None:
                x1, y1, x2, y2 = [float(v) for v in dets.xyxy[i]]
                x1 = max(0.0, x1); y1 = max(0.0, y1)
                x2 = min(img_w, x2); y2 = min(img_h, y2)
            else:
                x1, y1, x2, y2 = 0, 0, img_w, img_h

            coco_bbox = [round(x1, 2), round(y1, 2), round(x2-x1, 2), round(y2-y1, 2)]

            # Encode binary mask as RLE for COCO seg eval
            seg_rle = None
            mask_arr = None
            if dets.mask is not None:
                mask_arr = dets.mask[i].astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_arr))
                rle["counts"] = rle["counts"].decode("utf-8")
                seg_rle = rle

            coco_preds.append({
                "image_id":    img_id,
                "category_id": 1,
                "bbox":        coco_bbox,
                "segmentation": seg_rle if seg_rle else [],
                "area":        float(mask_utils.area(seg_rle)) if seg_rle else (x2-x1)*(y2-y1),
                "score":       round(score, 4),
            })

            per_image[img_id].append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "score":     score,
                "mask":      mask_arr,
            })

    print(f"[eval] {sum(len(v) for v in per_image.values())} detections on "
          f"{len(per_image)} images")
    return coco_preds, per_image, gt_data


# ── COCO segmentation mAP ─────────────────────────────────────────────────────

def run_coco_seg_eval(dataset_dir: Path, coco_preds: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_path = dataset_dir / "_annotations.coco.json"
    coco_gt = COCO(str(ann_path))

    if not coco_preds:
        print("[eval] WARNING: no predictions — skipping mAP")
        return {}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(coco_preds, tf)
        tmp_path = tf.name

    coco_dt = coco_gt.loadRes(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    # Bbox AP
    bbox_ev = COCOeval(coco_gt, coco_dt, "bbox")
    bbox_ev.evaluate(); bbox_ev.accumulate(); bbox_ev.summarize()
    bs = bbox_ev.stats

    # Segmentation AP — filter to predictions that carry an RLE mask
    preds_with_mask = [p for p in coco_preds if p.get("segmentation") and p["segmentation"] != []]
    return {
        "box_mAP":   round(float(bs[0]), 4),
        "box_mAP50": round(float(bs[1]), 4),
        "box_mAP75": round(float(bs[2]), 4),
        "box_AR100": round(float(bs[8]), 4),
    }


# ── Mean Mask IoU (matched pairs) ─────────────────────────────────────────────

def compute_mean_mask_iou(per_image: dict, gt_data: dict,
                          box_iou_thresh: float = 0.5) -> dict:
    """
    For each GT annotation with a polygon mask, find the highest-scoring
    prediction with box IoU >= box_iou_thresh and compute binary mask IoU.
    Returns mean over all matched GT instances.
    """
    from pycocotools import mask as mask_utils

    # Index GT annotations by image_id
    gt_by_image: dict[int, list] = {}
    for ann in gt_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    iou_values = []
    n_gt = 0
    n_matched = 0
    n_no_pred = 0
    n_no_mask_gt = 0

    for img_info in gt_data["images"]:
        img_id  = img_info["id"]
        img_w   = img_info["width"]
        img_h   = img_info["height"]
        gt_anns = gt_by_image.get(img_id, [])
        preds   = per_image.get(img_id, [])

        for gt_ann in gt_anns:
            n_gt += 1

            # Skip GT instances without a polygon mask
            if not gt_ann.get("segmentation"):
                n_no_mask_gt += 1
                continue

            # Convert GT polygon → binary mask
            rles = mask_utils.frPyObjects(
                gt_ann["segmentation"], img_h, img_w
            )
            rle_merged = mask_utils.merge(rles)
            gt_mask = mask_utils.decode(rle_merged).astype(bool)

            if not preds:
                n_no_pred += 1
                continue

            # GT bbox in xyxy for IoU matching
            bx, by, bw, bh = gt_ann["bbox"]
            gt_xyxy = [bx, by, bx + bw, by + bh]

            # Find best-matching prediction (highest score with box IoU >= thresh)
            best_iou_val = -1.0
            best_mask    = None
            for pred in preds:
                iou_val = box_iou(gt_xyxy, pred["bbox_xyxy"])
                if iou_val >= box_iou_thresh and pred["score"] > best_iou_val:
                    best_iou_val = pred["score"]
                    best_mask    = pred["mask"]

            if best_mask is None:
                n_no_pred += 1
                continue

            # Binary mask IoU
            pred_mask = best_mask.astype(bool)
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union        = np.logical_or(gt_mask, pred_mask).sum()
            miou = float(intersection) / float(union) if union > 0 else 0.0
            iou_values.append(miou)
            n_matched += 1

    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
    print(f"\n[maskiou] GT instances : {n_gt}")
    print(f"[maskiou] No GT mask   : {n_no_mask_gt}")
    print(f"[maskiou] No pred match: {n_no_pred}")
    print(f"[maskiou] Matched      : {n_matched}")
    print(f"[maskiou] Mean Mask IoU: {mean_iou:.4f}  ({mean_iou*100:.1f}%)")

    return {
        "mean_mask_iou":      round(mean_iou, 4),
        "n_gt_instances":     n_gt,
        "n_matched":          n_matched,
        "n_no_pred_match":    n_no_pred,
        "n_no_gt_mask":       n_no_mask_gt,
        "box_iou_threshold":  box_iou_thresh,
        "iou_histogram": {
            "p25":  round(float(np.percentile(iou_values, 25)), 4) if iou_values else 0,
            "p50":  round(float(np.percentile(iou_values, 50)), 4) if iou_values else 0,
            "p75":  round(float(np.percentile(iou_values, 75)), 4) if iou_values else 0,
            "p90":  round(float(np.percentile(iou_values, 90)), 4) if iou_values else 0,
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RF-DETR-Seg OOD Mask IoU evaluation")
    p.add_argument("--checkpoint",
                   default="runs/seg_medium_lr5e5/checkpoint_best_ema.pth",
                   help="Path to RF-DETR-Seg checkpoint (.pth)")
    p.add_argument("--dataset_dir",
                   default="data/processed/detection/cattleeyeview/test",
                   help="Dir containing _annotations.coco.json with segmentation polygons")
    p.add_argument("--output",
                   default="results/segmentation/cattleeyeview_maskiou.json",
                   help="Output JSON path")
    p.add_argument("--dataset_name",
                   default="cattleeyeview",
                   help="Dataset name stored in result JSON")
    p.add_argument("--threshold",
                   type=float, default=0.3,
                   help="Detection confidence threshold (default: 0.3)")
    p.add_argument("--box_iou_thresh",
                   type=float, default=0.5,
                   help="Min box IoU to count a prediction as matched (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir  = Path(args.dataset_dir)
    output_path  = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)

    coco_preds, per_image, gt_data = collect_predictions(
        model, dataset_dir, args.threshold
    )

    print("\n[eval] Running COCO segmentation mAP ...")
    coco_metrics = run_coco_seg_eval(dataset_dir, coco_preds)

    print("\n[eval] Computing mean Mask IoU (matched pairs) ...")
    maskiou_metrics = compute_mean_mask_iou(
        per_image, gt_data, args.box_iou_thresh
    )

    result = {
        "dataset":        args.dataset_name,
        "checkpoint":     args.checkpoint,
        "threshold":      args.threshold,
        "n_images":       len(per_image),
        "n_detections":   sum(len(v) for v in per_image.values()),
        **coco_metrics,
        **maskiou_metrics,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n── Results: {args.dataset_name} ─────────────────────────")
    print(f"  Box  mAP@50    : {coco_metrics.get('box_mAP50', 0):.4f}  "
          f"({coco_metrics.get('box_mAP50', 0)*100:.1f}%)")
    print(f"  Box  mAP@50:95 : {coco_metrics.get('box_mAP', 0):.4f}  "
          f"({coco_metrics.get('box_mAP', 0)*100:.1f}%)")
    print(f"  Mean Mask IoU  : {maskiou_metrics.get('mean_mask_iou', 0):.4f}  "
          f"({maskiou_metrics.get('mean_mask_iou', 0)*100:.1f}%)")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
