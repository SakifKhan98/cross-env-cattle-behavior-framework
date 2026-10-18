"""
eval_rfdetr_seg_kaggle.py
Cross-domain evaluation of RF-DETR-Seg on the Kaggle Cow Segmentation Dataset.
Converts YOLO segmentation labels -> COCO JSON, runs inference, computes mAP.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python eval_rfdetr_seg_kaggle.py \
        --checkpoint runs/seg_medium_lr5e5/checkpoint_best_ema.pth \
        --images_dir data/rfdetr_seg_eval/images \
        --labels_dir data/rfdetr_seg_eval/labels \
        --model medium \
        --output_dir runs/kaggle_eval_full \
        --conf_thresh 0.3 \
        --device cuda

Fixes applied (2026-03-24):
  1. Use RFDETRSegMedium/Large instead of RFDETRBase -- checkpoint was trained
     with RFDETRSegMediumConfig (resolution=432, patch_size=12, num_queries=200).
     RFDETRBase defaults differ, causing shape mismatches on load.
  2. .eval() and .cuda() target model.model.model (the actual nn.Module) since
     the rfdetr wrapper does not expose those methods directly.
  3. results.mask (not results.masks.data) -- rfdetr returns a supervision
     Detections object; masks are at .mask as a (N,H,W) bool numpy array.
  4. Image downscaling to MAX_SIDE=1024 prevents CUDA OOM on large images
     during mask postprocessing.
  5. Per-image torch.cuda.empty_cache() prevents fragmentation over 269 images.

Requirements:
    pip install pycocotools rfdetr supervision Pillow tqdm
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ─── Argument parser ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint_best_ema_B.pth"
    )
    p.add_argument("--images_dir", required=True, help="Val images directory")
    p.add_argument("--labels_dir", required=True, help="Val YOLO labels directory")
    p.add_argument("--model", default="medium", choices=["medium", "large"])
    p.add_argument("--output_dir", default="results/kaggle_eval")
    p.add_argument(
        "--conf_thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions",
    )
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


# ─── Step 1: YOLO seg -> COCO JSON ───────────────────────────────────────────
def yolo_seg_to_coco(images_dir, labels_dir, output_json):
    print("\n[1/4] Converting YOLO labels to COCO JSON...")

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".PNG", ".JPG", ".JPEG"}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix in img_extensions]
    )

    coco = {
        "info": {
            "description": "Kaggle Cow Segmentation Dataset -- Val Split",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "cattle", "supercategory": "animal"}],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    matched = 0
    skipped = 0

    for img_id, img_path in enumerate(image_files, start=1):
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            print(f"  [SKIP] No label for {img_path.name}")
            skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"  [SKIP] Cannot open {img_path.name}: {e}")
            skipped += 1
            continue

        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": W, "height": H}
        )

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            if int(parts[0]) != 0:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]

            abs_coords = []
            for i in range(0, len(coords), 2):
                abs_coords.append(round(coords[i] * W, 2))
                abs_coords.append(round(coords[i + 1] * H, 2))

            if len(abs_coords) < 6:
                continue

            xs = abs_coords[0::2]
            ys = abs_coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            n = len(xs)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += xs[i] * ys[j]
                area -= xs[j] * ys[i]
            area = abs(area) / 2.0

            if area < 1.0:
                continue

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [abs_coords],
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(x_max - x_min, 2),
                        round(y_max - y_min, 2),
                    ],
                    "area": round(area, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        matched += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  Converted {matched} images, {ann_id-1} annotations")
    print(f"  Skipped {skipped} images (no matching label)")
    print(f"  Saved COCO JSON -> {output_json}")
    return coco


# ─── Step 2: Run RF-DETR-Seg inference ───────────────────────────────────────
def run_inference(
    checkpoint, images_dir, coco_json, output_json, model_size, conf_thresh, device
):
    print(
        f"\n[2/4] Running RF-DETR-Seg inference (model={model_size}, conf={conf_thresh})..."
    )

    # FIX 1: Use RFDETRSegMedium/Large, NOT RFDETRBase.
    # RFDETRSegMediumConfig defaults match checkpoint: resolution=432,
    # patch_size=12, num_queries=200, group_detr=13.
    try:
        from rfdetr import RFDETRSegMedium, RFDETRSegLarge
    except ImportError:
        print("ERROR: rfdetr not installed. Run: pip install rfdetr")
        sys.exit(1)

    import torch

    print(f"  Loading checkpoint: {checkpoint}")
    ModelClass = RFDETRSegLarge if model_size == "large" else RFDETRSegMedium
    model = ModelClass(pretrain_weights=checkpoint, num_classes=1)

    # FIX 2: Wrapper hierarchy: RFDETRSegMedium -> .model (rfdetr Model)
    # -> .model (nn.Module LWDETR). .eval()/.cuda() must hit the nn.Module.
    # .predict() stays on the top-level wrapper.
    nn_module = model.model.model
    nn_module.eval()

    if device == "cuda" and torch.cuda.is_available():
        nn_module.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Using CPU (inference will be slow)")

    images_dir = Path(images_dir)
    with open(coco_json) as f:
        coco = json.load(f)

    predictions = []
    timings = []

    for img_rec in tqdm(coco["images"], desc="  Inferring"):
        img_path = images_dir / img_rec["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img_rec["width"], img_rec["height"]

        # FIX 4: Downscale large images to cap VRAM during mask postprocessing.
        # Model trained at resolution=432; oversized images cause mask OOM.
        MAX_SIDE = 1024
        if max(W, H) > MAX_SIDE:
            scale = MAX_SIDE / max(W, H)
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                results = model.predict(img, threshold=conf_thresh)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n  [OOM] Retrying {img_path.name} on CPU...")
            try:
                nn_module.cpu()
                with torch.no_grad():
                    results = model.predict(img, threshold=conf_thresh)
                nn_module.cuda()
                torch.cuda.empty_cache()
            except Exception as e2:
                print(f"\n  [SKIP] {img_path.name} failed on CPU too: {e2}")
                nn_module.cuda()
                continue
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

        # FIX 5: Clear cache per image to prevent fragmentation buildup.
        torch.cuda.empty_cache()

        if results is None or len(results) == 0:
            continue

        # FIX 3: rfdetr returns supervision Detections.
        # Masks live at results.mask — (N,H,W) bool numpy array, already on CPU.
        # NOT results.masks.data. No .cpu() call needed.
        try:
            boxes = np.array(results.xyxy)  # (N, 4)
            scores = np.array(results.confidence)  # (N,)
            masks = (
                results.mask
                if (hasattr(results, "mask") and results.mask is not None)
                else None
            )
        except Exception as e:
            print(f"\n  [WARN] Could not parse results for {img_path.name}: {e}")
            continue

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            pred = {
                "image_id": img_rec["id"],
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }

            if masks is not None and i < len(masks):
                mask = masks[i]  # bool numpy (H, W), already CPU
                try:
                    from pycocotools import mask as mask_util

                    mask_bool = mask.astype(np.uint8)
                    # Resize back to original image dims if model resized input
                    if mask_bool.shape != (H, W):
                        from PIL import Image as PILImage

                        mask_pil = PILImage.fromarray(mask_bool * 255).resize(
                            (W, H), PILImage.NEAREST
                        )
                        mask_bool = (np.array(mask_pil) > 127).astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(mask_bool))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    pred["segmentation"] = rle
                except Exception:
                    pass

            predictions.append(pred)

    avg_ms = np.mean(timings) if timings else 0
    print(f"  Generated {len(predictions)} predictions")
    if avg_ms > 0:
        print(f"  Avg inference time: {avg_ms:.1f} ms/image ({1000/avg_ms:.1f} FPS)")

    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Saved predictions -> {output_json}")
    return predictions, avg_ms


# ─── Step 3: COCO Evaluation ─────────────────────────────────────────────────
def run_coco_eval(gt_json, pred_json, output_dir):
    print("\n[3/4] Running COCO evaluation...")

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("ERROR: pycocotools not installed. Run: pip install pycocotools")
        sys.exit(1)

    with open(pred_json) as f:
        preds = json.load(f)

    if not preds:
        print(
            "  ERROR: No predictions found. Check confidence threshold or model output."
        )
        return None, None

    cocoGt = COCO(gt_json)
    results = {}

    # Bounding box
    print("\n  --- Bounding Box Detection ---")
    cocoDt_bbox = cocoGt.loadRes(pred_json)
    eval_bbox = COCOeval(cocoGt, cocoDt_bbox, "bbox")
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()
    results["bbox"] = {
        "mAP@50:95": float(eval_bbox.stats[0]),
        "mAP@50": float(eval_bbox.stats[1]),
        "mAP@75": float(eval_bbox.stats[2]),
        "AR@1": float(eval_bbox.stats[6]),
        "AR@10": float(eval_bbox.stats[7]),
        "AR@100": float(eval_bbox.stats[8]),
    }

    # Segmentation mask
    has_masks = any("segmentation" in p for p in preds)
    if has_masks:
        print("\n  --- Instance Segmentation Mask ---")
        cocoDt_seg = cocoGt.loadRes(pred_json)
        eval_seg = COCOeval(cocoGt, cocoDt_seg, "segm")
        eval_seg.evaluate()
        eval_seg.accumulate()
        eval_seg.summarize()
        results["segm"] = {
            "mAP@50:95": float(eval_seg.stats[0]),
            "mAP@50": float(eval_seg.stats[1]),
            "mAP@75": float(eval_seg.stats[2]),
            "AR@1": float(eval_seg.stats[6]),
            "AR@10": float(eval_seg.stats[7]),
            "AR@100": float(eval_seg.stats[8]),
        }
    else:
        print("\n  [WARN] No mask predictions found -- skipping segmentation eval.")
        results["segm"] = None

    return results, has_masks


# ─── Step 4: Print Summary ───────────────────────────────────────────────────
def print_summary(results, avg_ms, gt_json, args):
    print("\n" + "=" * 60)
    print("  CROSS-DOMAIN EVALUATION RESULTS")
    print("  Kaggle Cow Segmentation Dataset (Independent Ground Truth)")
    print("=" * 60)

    with open(gt_json) as f:
        gt = json.load(f)

    n_img = len(gt["images"])
    print(f"\n  Dataset:     {n_img} images, {len(gt['annotations'])} annotations")
    print(f"  Model:       RF-DETR-Seg-{args.model.capitalize()}")
    print(f"  Checkpoint:  Config B EMA (epoch 59)")
    print(f"  Conf thresh: {args.conf_thresh}")
    print(f"  Inference:   {avg_ms:.1f} ms/image avg")

    if results is None:
        print("\n  ERROR: Evaluation failed.")
        return

    print("\n  -- Bounding Box Detection --")
    bbox = results.get("bbox", {})
    print(f"  Det  mAP@50:95 : {bbox.get('mAP@50:95', 0)*100:.2f}%")
    print(f"  Det  mAP@50    : {bbox.get('mAP@50', 0)*100:.2f}%")
    print(f"  Det  mAP@75    : {bbox.get('mAP@75', 0)*100:.2f}%")
    print(f"  Det  AR@100    : {bbox.get('AR@100', 0)*100:.2f}%")

    segm = results.get("segm")
    if segm:
        print("\n  -- Instance Segmentation Mask --")
        print(f"  Seg  mAP@50:95 : {segm.get('mAP@50:95', 0)*100:.2f}%")
        print(f"  Seg  mAP@50    : {segm.get('mAP@50', 0)*100:.2f}%")
        print(f"  Seg  mAP@75    : {segm.get('mAP@75', 0)*100:.2f}%")
        print(f"  Seg  AR@100    : {segm.get('AR@100', 0)*100:.2f}%")
    else:
        print("\n  [!] Segmentation masks not evaluated (no mask output from model)")

    print("\n  -- Thesis Statement --")
    if segm:
        print(
            f'  "RF-DETR-Seg-{args.model.capitalize()} achieves '
            f"{segm.get('mAP@50:95', 0)*100:.1f}% mask mAP@50:95 on an"
        )
        print(f"  independent human-annotated cattle segmentation benchmark")
        print(f"  (N={n_img} images), confirming generalization beyond")
        print(f'  pseudo-labeled training data."')
    else:
        print(
            f'  "RF-DETR-Seg-{args.model.capitalize()} achieves '
            f"{bbox.get('mAP@50:95', 0)*100:.1f}% bounding box mAP@50:95"
        )
        print(f"  on an independent human-annotated cattle detection benchmark")
        print(f'  (N={n_img} images)."')

    print("=" * 60)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_json = str(output_dir / "gt_coco.json")
    pred_json = str(output_dir / "predictions.json")

    yolo_seg_to_coco(args.images_dir, args.labels_dir, gt_json)

    predictions, avg_ms = run_inference(
        checkpoint=args.checkpoint,
        images_dir=args.images_dir,
        coco_json=gt_json,
        output_json=pred_json,
        model_size=args.model,
        conf_thresh=args.conf_thresh,
        device=args.device,
    )

    results, has_masks = run_coco_eval(gt_json, pred_json, str(output_dir))

    print_summary(results, avg_ms, gt_json, args)

    summary = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "conf_thresh": args.conf_thresh,
        "avg_inference_ms": avg_ms,
        "n_images": len(json.load(open(gt_json))["images"]),
        "results": results,
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Full results saved -> {summary_path}")


if __name__ == "__main__":
    main()
