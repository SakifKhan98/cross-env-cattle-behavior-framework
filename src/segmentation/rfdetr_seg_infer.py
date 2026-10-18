"""
src/segmentation/rfdetr_seg_infer.py
Phase 3 — RF-DETR-Seg Instance Segmentation Inference

Runs the fine-tuned RF-DETR-Seg model (Config B, epoch 59 EMA) on raw
video frames and produces instance segmentation masks in the same
_masks.json format as SAM2, for downstream tracking compatibility.

INPUT:
    Raw frame images from data/raw/{cbvd5,cvb}/

OUTPUT:
    {paths.output_dir}/{dataset}/{video_id}_masks.json
    (default: data/processed/segmentation_rfdetr/{dataset}/{video_id}_masks.json — set in rfdetr_seg.yaml)

USAGE:
    python src/segmentation/rfdetr_seg_infer.py --config configs/segmentation/rfdetr_seg.yaml
    python src/segmentation/rfdetr_seg_infer.py --config configs/segmentation/rfdetr_seg.yaml --dataset cbvd5
    python src/segmentation/rfdetr_seg_infer.py --config configs/segmentation/rfdetr_seg.yaml --sanity
    python src/segmentation/rfdetr_seg_infer.py --config configs/segmentation/rfdetr_seg.yaml --video_id 618
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmentation.mask_utils import mask_to_rle, mask_area


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR-Seg Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation/rfdetr_seg.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cbvd5", "cvb", "both"],
        default="both",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Process only first 3 videos",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="Process only this specific video ID",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] Loaded: {config_path}")
    return cfg


def load_model(checkpoint_path: str):
    from rfdetr import RFDETRSegMedium

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[model] Loading RF-DETR-Seg from {checkpoint_path} ...")
    model = RFDETRSegMedium(pretrain_weights=str(ckpt))
    print(f"[model] Loaded successfully")
    return model


def xyxy_to_xywh(xyxy: np.ndarray) -> list:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [round(float(x1), 1), round(float(y1), 1),
            round(float(x2 - x1), 1), round(float(y2 - y1), 1)]


def frame_results_from_detections(detections) -> list:
    results = []
    if detections is None:
        return results

    boxes = detections.xyxy
    scores = detections.confidence
    masks = getattr(detections, "mask", None)

    for i in range(len(boxes)):
        r = {
            "bbox": xyxy_to_xywh(boxes[i]),
            "score": round(float(scores[i]), 4),
        }
        if masks is not None and i < len(masks):
            binary_mask = masks[i].astype(bool)
            r["mask_rle"] = mask_to_rle(binary_mask)
            r["mask_area"] = mask_area(binary_mask)
        else:
            r["mask_rle"] = None
            r["mask_area"] = 0
        results.append(r)

    return results


def frame_results_xyxy(detections) -> list:
    """Like frame_results_from_detections but bbox is [x1,y1,x2,y2] (xyxy)."""
    results = []
    if detections is None:
        return results

    boxes = detections.xyxy
    scores = detections.confidence
    masks = getattr(detections, "mask", None)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        r = {
            "bbox": [round(float(x1), 1), round(float(y1), 1),
                     round(float(x2), 1), round(float(y2), 1)],
            "score": round(float(scores[i]), 4),
        }
        if masks is not None and i < len(masks):
            binary_mask = masks[i].astype(bool)
            r["mask_rle"] = mask_to_rle(binary_mask)
            r["mask_area"] = mask_area(binary_mask)
        else:
            r["mask_rle"] = None
            r["mask_area"] = 0
        results.append(r)

    return results


def predict_frame(model, frame_bgr: np.ndarray, score_threshold: float,
                  max_size: int = 1280) -> list:
    """Run RF-DETR-Seg inference on a single BGR numpy frame.

    Accepts the frame in memory — no temp file is written.
    Returns a list of detection dicts with xyxy bboxes (in original frame
    coordinates), scores, and mask RLEs (at the inference resolution).

    High-resolution frames (e.g. 4480×2800) are downscaled to max_size on
    the longest edge before inference to avoid GPU OOM during mask
    upsampling. Bbox coordinates are then scaled back to original dimensions.
    """
    from PIL import Image
    import cv2 as _cv2

    frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
    orig_h, orig_w = frame_bgr.shape[:2]

    scale = 1.0
    if max(orig_h, orig_w) > max_size:
        scale = max_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        frame_rgb = _cv2.resize(frame_rgb, (new_w, new_h), interpolation=_cv2.INTER_AREA)

    pil_image = Image.fromarray(frame_rgb)
    detections = model.predict(pil_image, threshold=score_threshold)
    results = frame_results_xyxy(detections)

    if scale != 1.0:
        inv = 1.0 / scale
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            r["bbox"] = [
                round(x1 * inv, 1), round(y1 * inv, 1),
                round(x2 * inv, 1), round(y2 * inv, 1),
            ]

    return results


def enumerate_cbvd5_frames(frames_dir: Path):
    video_frames = {}
    for img_path in sorted(frames_dir.glob("*.jpg")):
        stem = img_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        video_id = parts[0]
        try:
            timestamp = int(parts[1])
        except ValueError:
            continue
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append((timestamp, str(img_path)))
    for video_id in video_frames:
        video_frames[video_id].sort(key=lambda x: x[0])
    return video_frames


def enumerate_cvb_frames(frames_root: Path):
    clips = {}
    for clip_dir in sorted(frames_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_id = clip_dir.name
        frames = []
        for img_path in sorted(clip_dir.glob("img_*.jpg")):
            try:
                frame_num = int(img_path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            frames.append((frame_num, str(img_path)))
        frames.sort(key=lambda x: x[0])
        clips[clip_id] = frames
    return clips


def process_dataset(model, cfg, dataset, args):
    ds_cfg = cfg["dataset"][dataset]
    frames_dir = Path(ds_cfg["frames_dir"])

    if not frames_dir.exists():
        print(f"  [WARNING] Frames directory not found: {frames_dir}")
        return

    out_dir = Path(cfg["paths"]["output_dir"]) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    score_threshold = cfg["rfdetr_seg"]["score_threshold"]

    if dataset == "cbvd5":
        video_frames = enumerate_cbvd5_frames(frames_dir)
    else:
        video_frames = enumerate_cvb_frames(frames_dir)

    video_ids = sorted(video_frames.keys())

    if args.video_id:
        matched = [v for v in video_ids if args.video_id in v]
        if not matched:
            print(f"  [WARN] No video matching '{args.video_id}' found in {dataset}")
            return
        video_ids = matched

    if args.sanity:
        video_ids = video_ids[:3]

    print(f"\n  {dataset.upper()}: {len(video_ids)} videos")

    t0 = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    total_masks = 0

    for vid_idx, video_id in enumerate(tqdm(video_ids, desc=f"  {dataset}")):
        out_path = out_dir / f"{video_id}_masks.json"
        if out_path.exists():
            n_skip += 1
            continue

        frames_list = video_frames[video_id]
        result = {"video_id": video_id, "dataset": dataset, "frames": {}}

        for frame_idx, img_path in frames_list:
            try:
                detections = model.predict(str(img_path), threshold=score_threshold)
                frame_dets = frame_results_from_detections(detections)
            except Exception as e:
                print(f"\n    [WARN] {video_id} frame {frame_idx}: {e}")
                frame_dets = []

            result["frames"][str(frame_idx)] = frame_dets
            if frame_dets:
                total_masks += len(frame_dets)

        try:
            with open(out_path, "w") as f:
                json.dump(result, f)
            n_ok += 1
        except Exception as e:
            print(f"\n    [ERROR] Failed to write {out_path}: {e}")
            n_fail += 1
            continue

    elapsed = time.time() - t0
    print(f"  [{dataset}] Done: {n_ok} written, {n_skip} skipped, "
          f"{n_fail} failed, {total_masks} masks, {elapsed:.1f}s")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    if device.type == "cuda":
        print(f"[device] GPU  : {torch.cuda.get_device_name(0)}")

    model = load_model(cfg["rfdetr_seg"]["checkpoint"])

    datasets = ["cbvd5", "cvb"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        process_dataset(model, cfg, dataset, args)

    print("\n[rfdetr_seg] Phase 3 segmentation complete!")


if __name__ == "__main__":
    main()
