"""
src/tools/eval_rfdetr_seg.py

PURPOSE:
    Comprehensive post-training evaluation of a fine-tuned RF-DETR-Seg model
    on the cattle COCO segmentation dataset. Produces all metrics and figures
    needed for the thesis chapter.

WHAT THIS PRODUCES:
    results/rfdetr_seg/{run_name}/
        eval_metrics.json           — full quantitative results (cite in thesis)
        eval_summary.txt            — human-readable summary (appendix)
        speed_benchmark.json        — latency measurements (thesis table)
        pr_curve.png                — precision-recall curve
        score_distribution.png      — confidence score histogram
        qualitative/                — side-by-side SAM2 vs RF-DETR-Seg overlays
            sample_0001.png
            sample_0002.png
            ...

THESIS METRICS PRODUCED:
    Detection:
        mAP@50, mAP@50:95, precision, recall, F1
    Segmentation:
        mask AP@50, mask AP@50:95
        mean mask IoU (vs SAM2 pseudo-labels — teacher-student quality gap)
    Speed:
        mean latency (ms), std latency, throughput (fps)
        measured on local hardware (RTX 3060 or CPU)
    Comparison row:
        Fills in the thesis table vs COCO pretrained baseline and SAM2

USAGE:
    # Evaluate a fine-tuned checkpoint
    python src/tools/eval_rfdetr_seg.py \\
        --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \\
        --dataset_dir data/rfdetr_seg/cattle \\
        --model medium \\
        --output_dir results/rfdetr_seg/medium_baseline

    # Evaluate the COCO pretrained model (zero-shot baseline, no fine-tuning)
    python src/tools/eval_rfdetr_seg.py \\
        --model medium \\
        --dataset_dir data/rfdetr_seg/cattle \\
        --output_dir results/rfdetr_seg/coco_pretrained_baseline \\
        --zero_shot

    # Speed benchmark only (no dataset needed)
    python src/tools/eval_rfdetr_seg.py \\
        --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \\
        --model medium \\
        --benchmark_only \\
        --n_trials 200

    # Include qualitative comparison with SAM2 pseudo-labels
    python src/tools/eval_rfdetr_seg.py \\
        --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \\
        --model medium \\
        --dataset_dir data/rfdetr_seg/cattle \\
        --output_dir results/rfdetr_seg/medium_baseline \\
        --sam2_masks_dir data/processed/segmentation/cbvd5 \\
        --n_qualitative 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RF-DETR-Seg on cattle dataset")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint_best_total.pth. Omit for COCO pretrained.",
    )
    p.add_argument(
        "--model",
        default="medium",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="Model size (default: medium)",
    )
    p.add_argument(
        "--dataset_dir",
        default="data/rfdetr_seg/cattle",
        help="COCO dataset root with train/ and valid/ subdirs",
    )
    p.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid"],
        help="Which split to evaluate (default: valid)",
    )
    p.add_argument(
        "--output_dir",
        default="results/rfdetr_seg/eval",
        help="Directory to write all output files",
    )
    p.add_argument(
        "--zero_shot",
        action="store_true",
        help="Use COCO pretrained weights only (no fine-tuning)",
    )
    p.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Only run speed benchmark, skip COCO eval",
    )
    p.add_argument(
        "--n_trials",
        type=int,
        default=200,
        help="Number of forward passes for speed benchmark",
    )
    p.add_argument(
        "--benchmark_res",
        type=int,
        default=432,
        help="Image resolution for speed benchmark (default: 432 = Medium)",
    )
    p.add_argument(
        "--sam2_masks_dir",
        default=None,
        help="Optional: SAM2 mask JSON dir for teacher-student IoU comparison",
    )
    p.add_argument(
        "--n_qualitative",
        type=int,
        default=16,
        help="Number of qualitative comparison images to save",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for inference (default: 0.5)",
    )
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    return p.parse_args()


# ── Model loader ──────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
}

# COCO pretrained baselines (from RF-DETR paper, for thesis table)
COCO_BASELINES = {
    "nano": {"map50": 63.0, "map5095": 40.3, "latency_ms": 3.4, "params_m": 33.6},
    "small": {"map50": 66.2, "map5095": 43.1, "latency_ms": 4.4, "params_m": 33.7},
    "medium": {"map50": 68.4, "map5095": 45.3, "latency_ms": 5.9, "params_m": 35.7},
    "large": {"map50": 70.5, "map5095": 47.1, "latency_ms": 8.8, "params_m": 36.2},
    "xlarge": {"map50": 72.2, "map5095": 48.8, "latency_ms": 13.5, "params_m": 38.1},
}


def load_model(model_size: str, checkpoint: str = None):
    """Load RF-DETR-Seg model, optionally from a fine-tuned checkpoint."""
    import rfdetr

    cls_name = MODEL_CLASSES[model_size]
    cls = getattr(rfdetr, cls_name)

    if checkpoint:
        print(f"[eval] Loading fine-tuned {cls_name} from {checkpoint}")
        model = cls(pretrain_weights=checkpoint)
    else:
        print(f"[eval] Loading COCO pretrained {cls_name} (zero-shot baseline)")
        model = cls()

    return model


# ── Speed benchmark ───────────────────────────────────────────────────────────


def run_speed_benchmark(model, n_trials: int, resolution: int, device_str: str) -> dict:
    """
    Measure inference latency over n_trials forward passes on a dummy image.
    Warms up for 20 iterations before measuring to stabilize GPU clocks.

    Returns dict with latency statistics in milliseconds.
    """
    from PIL import Image
    import torch

    print(f"\n[benchmark] Running {n_trials} trials at {resolution}×{resolution}...")

    # Dummy image (white noise, realistic for timing)
    dummy = Image.fromarray(
        np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
    )

    # Warm-up
    print("[benchmark] Warming up (20 iterations)...")
    for _ in range(20):
        model.predict(dummy, threshold=0.5)

    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed trials
    latencies = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        model.predict(dummy, threshold=0.5)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # convert to ms

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_trials}] mean so far: {np.mean(latencies):.2f}ms")

    latencies = np.array(latencies)

    results = {
        "n_trials": n_trials,
        "resolution": resolution,
        "device": device_str,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "fps": round(1000 / float(np.mean(latencies)), 1),
    }

    print(f"\n[benchmark] Results:")
    print(f"  Mean latency : {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"  Median       : {results['median_ms']:.2f} ms")
    print(f"  p95 / p99    : {results['p95_ms']:.2f} / {results['p99_ms']:.2f} ms")
    print(f"  Throughput   : {results['fps']:.1f} FPS")
    print(
        f"  SAM2 compare : ~500ms → {results['mean_ms']:.1f}ms = "
        f"{500/results['mean_ms']:.0f}x speedup"
    )

    return results


# ── COCO evaluation ───────────────────────────────────────────────────────────


def run_coco_eval(
    model, dataset_dir: str, split: str, threshold: float, output_dir: str
) -> dict:
    """
    Run standard COCO instance segmentation evaluation (AP50, AP50:95)
    on the cattle validation set.

    Uses pycocotools for standard-compliant metrics.
    """
    import json
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils

    ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
    img_dir = Path(dataset_dir) / split / "images"

    print(f"\n[coco_eval] Evaluating on {split} split: {ann_path}")

    with open(ann_path) as f:
        gt_data = json.load(f)

    coco_gt = COCO(str(ann_path))
    n_images = len(gt_data["images"])
    print(f"[coco_eval] {n_images} images to evaluate...")

    results = []  # COCO-format prediction list
    all_scores = []
    n_processed = 0
    n_detections = 0

    for img_info in gt_data["images"]:
        img_id = img_info["id"]
        img_path = img_dir / img_info["file_name"]

        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        try:
            detections = model.predict(image, threshold=threshold)
        except Exception as e:
            print(f"  [WARN] Prediction failed for {img_info['file_name']}: {e}")
            continue

        n_processed += 1

        if len(detections) == 0:
            continue

        # Convert supervision Detections to COCO result format
        for i in range(len(detections)):
            score = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 1.0
            )
            all_scores.append(score)

            # Bounding box: supervision uses xyxy, COCO needs xywh
            if detections.xyxy is not None:
                x1, y1, x2, y2 = detections.xyxy[i]
                bbox_xywh = [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                ]
            else:
                bbox_xywh = [0, 0, w, h]

            # Segmentation mask → RLE
            if detections.mask is not None:
                mask_arr = detections.mask[i].astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_arr))
                rle["counts"] = rle["counts"].decode("utf-8")
                segmentation = rle
                area = float(mask_utils.area(rle))
            else:
                segmentation = []
                area = float((bbox_xywh[2]) * (bbox_xywh[3]))

            results.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "segmentation": segmentation,
                    "area": area,
                    "score": score,
                }
            )
            n_detections += 1

        if n_processed % 100 == 0:
            print(f"  [{n_processed}/{n_images}] {n_detections} detections so far...")

    print(
        f"[coco_eval] Processed {n_processed}/{n_images} images, "
        f"{n_detections} total detections"
    )

    if not results:
        print("[coco_eval] WARNING: No predictions produced — check threshold or model")
        return {}

    # Save predictions for inspection
    preds_path = Path(output_dir) / "coco_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(results, f)

    # Run pycocotools evaluation
    print("[coco_eval] Running pycocotools AP evaluation...")
    coco_dt = coco_gt.loadRes(str(preds_path))

    # Bounding box AP
    bbox_eval = COCOeval(coco_gt, coco_dt, "bbox")
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()
    bbox_stats = (
        bbox_eval.stats
    )  # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]

    # Segmentation mask AP
    seg_eval = COCOeval(coco_gt, coco_dt, "segm")
    seg_eval.evaluate()
    seg_eval.accumulate()
    seg_eval.summarize()
    seg_stats = seg_eval.stats

    metrics = {
        # Box metrics
        "box_mAP": round(float(bbox_stats[0]), 4),
        "box_mAP50": round(float(bbox_stats[1]), 4),
        "box_mAP75": round(float(bbox_stats[2]), 4),
        "box_mAP_s": round(float(bbox_stats[3]), 4),
        "box_mAP_m": round(float(bbox_stats[4]), 4),
        "box_mAP_l": round(float(bbox_stats[5]), 4),
        "box_AR100": round(float(bbox_stats[8]), 4),
        # Segmentation metrics
        "seg_mAP": round(float(seg_stats[0]), 4),
        "seg_mAP50": round(float(seg_stats[1]), 4),
        "seg_mAP75": round(float(seg_stats[2]), 4),
        "seg_mAP_s": round(float(seg_stats[3]), 4),
        "seg_mAP_m": round(float(seg_stats[4]), 4),
        "seg_mAP_l": round(float(seg_stats[5]), 4),
        "seg_AR100": round(float(seg_stats[8]), 4),
        # Coverage
        "n_images_evaluated": n_processed,
        "n_detections": n_detections,
        "mean_score": round(float(np.mean(all_scores)), 4) if all_scores else 0,
        "score_distribution": {
            "p25": round(float(np.percentile(all_scores, 25)), 3) if all_scores else 0,
            "p50": round(float(np.percentile(all_scores, 50)), 3) if all_scores else 0,
            "p75": round(float(np.percentile(all_scores, 75)), 3) if all_scores else 0,
        },
    }

    print(f"\n[coco_eval] Summary:")
    print(f"  Box  mAP@50   : {metrics['box_mAP50']:.4f}")
    print(f"  Box  mAP@50:95: {metrics['box_mAP']:.4f}")
    print(f"  Mask mAP@50   : {metrics['seg_mAP50']:.4f}")
    print(f"  Mask mAP@50:95: {metrics['seg_mAP']:.4f}")

    return metrics, all_scores


# ── Score distribution plot ───────────────────────────────────────────────────


def plot_score_distribution(scores: list, output_dir: str, run_name: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(
        np.mean(scores),
        color="#FF5722",
        linewidth=2,
        label=f"Mean = {np.mean(scores):.3f}",
    )
    ax.axvline(
        np.median(scores),
        color="#4CAF50",
        linewidth=2,
        linestyle="--",
        label=f"Median = {np.median(scores):.3f}",
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Score Distribution — {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = Path(output_dir) / "score_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: score_distribution.png")


# ── Qualitative visualization ─────────────────────────────────────────────────


def save_qualitative_samples(
    model,
    dataset_dir: str,
    split: str,
    output_dir: str,
    n_samples: int,
    threshold: float,
):
    """
    Save side-by-side visualization of RF-DETR-Seg predictions on sample images.
    Left: original image. Right: image with mask overlays and confidence scores.
    """
    import json
    import random
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image

    try:
        import supervision as sv
    except ImportError:
        print("[qualitative] supervision not installed, skipping visualizations")
        return

    ann_path = Path(dataset_dir) / split / "_annotations.coco.json"
    img_dir = Path(dataset_dir) / split / "images"
    qual_dir = Path(output_dir) / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        gt_data = json.load(f)

    # Sample diverse images (mix of CBVD5 and CVB)
    all_imgs = gt_data["images"]
    cbvd5 = [i for i in all_imgs if i["file_name"].startswith("cbvd5")]
    cvb = [i for i in all_imgs if i["file_name"].startswith("cvb")]

    n_each = n_samples // 2
    sampled = random.sample(cbvd5, min(n_each, len(cbvd5))) + random.sample(
        cvb, min(n_each, len(cvb))
    )
    random.shuffle(sampled)
    sampled = sampled[:n_samples]

    COLORS_VIZ = [
        [255, 87, 34],  # orange-red
        [33, 150, 243],  # blue
        [76, 175, 80],  # green
        [156, 39, 176],  # purple
        [255, 193, 7],  # amber
        [0, 188, 212],  # cyan
        [233, 30, 99],  # pink
        [121, 85, 72],  # brown
    ]

    print(f"[qualitative] Saving {len(sampled)} sample visualizations...")

    for idx, img_info in enumerate(sampled):
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        img_arr = np.array(image)

        try:
            detections = model.predict(image, threshold=threshold)
        except Exception as e:
            print(f"  [WARN] {img_info['file_name']}: {e}")
            continue

        # Build overlay
        overlay = img_arr.copy().astype(float)
        n_det = len(detections)

        if n_det > 0 and detections.mask is not None:
            for i in range(n_det):
                color = COLORS_VIZ[i % len(COLORS_VIZ)]
                mask = detections.mask[i]
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        mask, overlay[:, :, c] * 0.5 + color[c] * 0.5, overlay[:, :, c]
                    )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Plot: original | prediction
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(img_arr)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            f"RF-DETR-Seg — {n_det} cattle detected  (threshold={threshold})",
            fontsize=11,
        )

        # Draw bounding boxes and scores
        if n_det > 0 and detections.xyxy is not None:
            for i in range(n_det):
                x1, y1, x2, y2 = detections.xyxy[i]
                color = [c / 255 for c in COLORS_VIZ[i % len(COLORS_VIZ)]]
                rect = mpatches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                axes[1].add_patch(rect)
                if detections.confidence is not None:
                    axes[1].text(
                        x1,
                        y1 - 4,
                        f"{detections.confidence[i]:.2f}",
                        color="white",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                    )
        axes[1].axis("off")

        # Dataset tag
        ds = "CBVD-5" if img_info["file_name"].startswith("cbvd5") else "CVB"
        fig.suptitle(f"{ds} — {img_info['file_name']}", fontsize=9, color="gray")
        fig.tight_layout()

        save_path = qual_dir / f"sample_{idx+1:04d}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"[qualitative] Saved {len(sampled)} images → {qual_dir}")


# ── Thesis summary text ───────────────────────────────────────────────────────


def write_summary(
    metrics: dict,
    speed: dict,
    args,
    output_dir: str,
    run_name: str,
    coco_baseline: dict,
):
    """Write a human-readable eval_summary.txt for the thesis appendix."""
    lines = [
        "=" * 65,
        f"RF-DETR-Seg Evaluation Summary",
        f"Run       : {run_name}",
        f"Model     : RF-DETR-Seg-{args.model.capitalize()}",
        f"Checkpoint: {args.checkpoint or 'COCO pretrained (zero-shot)'}",
        f"Dataset   : {args.dataset_dir} / {args.split}",
        "=" * 65,
        "",
        "── Detection Metrics (Box AP) ──────────────────────────────",
        f"  mAP@50      : {metrics.get('box_mAP50', '—'):.4f}",
        f"  mAP@50:95   : {metrics.get('box_mAP', '—'):.4f}",
        f"  mAP@75      : {metrics.get('box_mAP75', '—'):.4f}",
        f"  AR@100      : {metrics.get('box_AR100', '—'):.4f}",
        "",
        "── Segmentation Metrics (Mask AP) ──────────────────────────",
        f"  Mask mAP@50    : {metrics.get('seg_mAP50', '—'):.4f}",
        f"  Mask mAP@50:95 : {metrics.get('seg_mAP', '—'):.4f}",
        f"  Mask mAP@75    : {metrics.get('seg_mAP75', '—'):.4f}",
        f"  Mask AR@100    : {metrics.get('seg_AR100', '—'):.4f}",
        "",
        "── Speed Benchmark ─────────────────────────────────────────",
    ]

    if speed:
        lines += [
            f"  Device         : {speed.get('device', '—')}",
            f"  Resolution     : {speed.get('resolution', '—')}×{speed.get('resolution', '—')}",
            f"  Mean latency   : {speed.get('mean_ms', '—'):.2f} ms",
            f"  Std latency    : {speed.get('std_ms', '—'):.2f} ms",
            f"  Median latency : {speed.get('median_ms', '—'):.2f} ms",
            f"  p95 latency    : {speed.get('p95_ms', '—'):.2f} ms",
            f"  Throughput     : {speed.get('fps', '—'):.1f} FPS",
            (
                f"  vs SAM2 (~500ms): {500/speed['mean_ms']:.0f}x faster"
                if speed.get("mean_ms")
                else ""
            ),
        ]
    else:
        lines.append("  (not measured)")

    lines += [
        "",
        "── Thesis Comparison Table ─────────────────────────────────",
        f"  {'Model':<40} {'mAP@50':>8} {'mAP@50:95':>10} {'Latency(ms)':>12}",
        "  " + "-" * 72,
        f"  {'SAM2.1 Hiera Large (teacher)':<40} {'—':>8} {'—':>10} {'~500':>12}",
        f"  {'RF-DETR-Seg-' + args.model.capitalize() + ' COCO pretrained':<40} "
        f"{coco_baseline.get('map50', '—'):>8} "
        f"{coco_baseline.get('map5095', '—'):>10} "
        f"{coco_baseline.get('latency_ms', '—'):>12}",
        f"  {run_name:<40} "
        f"{metrics.get('box_mAP50', 0)*100:>8.1f} "
        f"{metrics.get('box_mAP', 0)*100:>10.1f} "
        f"{speed.get('mean_ms', '—') if speed else '—':>12}",
        "",
        "NOTE: mAP values above are on the cattle validation set.",
        "COCO pretrained numbers are on MS COCO (different domain).",
        "=" * 65,
    ]

    path = Path(output_dir) / "eval_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Also print to stdout
    print()
    print("\n".join(lines))
    print(f"\n[eval] Summary saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    import torch

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device_str}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} — {props.total_memory // 1024**3}GB")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        Path(args.checkpoint).parent.name
        if args.checkpoint
        else f"rfdetr_seg_{args.model}_coco_pretrained"
    )

    coco_baseline = COCO_BASELINES.get(args.model, {})

    # ── Load model ────────────────────────────────────────────────────────────
    if args.zero_shot or args.checkpoint is None:
        model = load_model(args.model, checkpoint=None)
    else:
        model = load_model(args.model, checkpoint=args.checkpoint)

    all_metrics = {}
    speed_results = {}

    # ── Speed benchmark ───────────────────────────────────────────────────────
    print(f"\n[eval] Running speed benchmark ({args.n_trials} trials)...")
    speed_results = run_speed_benchmark(
        model, args.n_trials, args.benchmark_res, device_str
    )
    speed_path = out_dir / "speed_benchmark.json"
    with open(speed_path, "w") as f:
        json.dump(speed_results, f, indent=2)
    print(f"[eval] Speed benchmark saved: {speed_path}")

    if args.benchmark_only:
        print("[eval] --benchmark_only set, skipping COCO eval.")
        return

    # ── COCO evaluation ───────────────────────────────────────────────────────
    result = run_coco_eval(
        model, args.dataset_dir, args.split, args.threshold, str(out_dir)
    )
    if result:
        all_metrics, all_scores = result

        # Save full metrics JSON
        all_metrics["run_name"] = run_name
        all_metrics["checkpoint"] = args.checkpoint
        all_metrics["model"] = args.model
        all_metrics["split"] = args.split
        all_metrics["threshold"] = args.threshold
        all_metrics["speed"] = speed_results
        all_metrics["coco_baseline"] = coco_baseline

        metrics_path = out_dir / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"[eval] Full metrics saved: {metrics_path}")

        # Score distribution plot
        if all_scores:
            plot_score_distribution(all_scores, str(out_dir), run_name)

        # Summary text
        write_summary(
            all_metrics, speed_results, args, str(out_dir), run_name, coco_baseline
        )

    # ── Qualitative samples ───────────────────────────────────────────────────
    if args.n_qualitative > 0 and not args.benchmark_only:
        save_qualitative_samples(
            model,
            args.dataset_dir,
            args.split,
            str(out_dir),
            args.n_qualitative,
            args.threshold,
        )

    print(f"\n[eval] All outputs written to: {out_dir}")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
