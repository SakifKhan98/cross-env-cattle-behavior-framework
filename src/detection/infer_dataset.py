"""
src/detection/infer_dataset.py

Runs the trained RF-DETR cattle detector on ALL frames from both
datasets and saves per-video detection JSONs.

These JSONs are the input to:
  - src/segmentation/segment.py  (SAM2 prompting)
  - src/tracking/track.py        (OC-SORT tracking)

Output format — one JSON per video:
    {
      "video_id": "618",
      "dataset":  "cbvd5",
      "fps":      30,
      "width":    1920,
      "height":   1080,
      "frames": {
        "1": [
          {"bbox": [x, y, w, h], "score": 0.94},
          ...
        ],
        "2": [...],
        ...
      }
    }

Output location:
    data/processed/tracking/{dataset}/{video_id}.json

Usage:
    # Run on both datasets using the combined model
    python src/detection/infer_dataset.py --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth

    # Run on one dataset only
    python src/detection/infer_dataset.py --checkpoint <path> --dataset cbvd5

    # Override confidence threshold
    python src/detection/infer_dataset.py --checkpoint <path> --conf_thresh 0.4
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


# ── Dataset frame sources ─────────────────────────────────────────────────────
#
# For each dataset we need to know:
#   - Where the frame images live
#   - How to enumerate (video_id, frame_id) pairs
#   - What the image filename pattern is
#
# CBVD-5: labelframes/labelframes/{video_id}_{timestamp:05d}.jpg
# CVB:    raw_frames/{clip_id}/img_{frame:05d}.jpg

DATASET_CONFIGS = {
    "cbvd5": {
        "frames_dir": "data/raw/cbvd5/labelframes/labelframes",
        "pattern": "{video_id}_{timestamp:05d}.jpg",
        "fps": 30,
        "width": 1920,
        "height": 1080,
    },
    "cvb": {
        "frames_dir": "data/raw/cvb/raw_frames",
        "pattern": "img_{frame:05d}.jpg",
        "fps": 30,
        "width": 1920,
        "height": 1080,
    },
}


# ── Model loader ──────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
    """Load trained RF-DETR model from checkpoint."""
    try:
        from rfdetr import RFDETRMedium
    except ImportError:
        print("[ERROR] rfdetr is not installed. Run: pip install rfdetr")
        sys.exit(1)

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model = RFDETRMedium(pretrained_weights=str(ckpt))
    print(f"  [OK] Loaded checkpoint: {ckpt.name}")
    return model


# ── Normalize detections ──────────────────────────────────────────────────────


def normalize_detections(raw_output):
    """
    RF-DETR output format can vary across versions.
    This handles the most common cases and returns a consistent list of:
      [{"bbox": [x, y, w, h], "score": float}, ...]

    bbox is in [x_min, y_min, width, height] pixel coordinates.
    """
    detections = []

    # Case 1: supervision-style DetectionResult object
    if hasattr(raw_output, "xyxy") and hasattr(raw_output, "confidence"):
        boxes = raw_output.xyxy  # shape (N, 4) — x1,y1,x2,y2
        scores = raw_output.confidence  # shape (N,)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            detections.append(
                {
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                    "score": round(float(scores[i]), 4),
                }
            )
        return detections

    # Case 2: list of dicts
    if isinstance(raw_output, list):
        for det in raw_output:
            if isinstance(det, dict) and "bbox" in det:
                detections.append(
                    {
                        "bbox": [round(v, 1) for v in det["bbox"]],
                        "score": round(
                            float(det.get("score", det.get("confidence", 1.0))), 4
                        ),
                    }
                )
        return detections

    # Case 3: dict with "boxes" and "scores"
    if isinstance(raw_output, dict):
        boxes = raw_output.get("boxes", raw_output.get("xyxy", []))
        scores = raw_output.get("scores", raw_output.get("confidence", []))
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                {
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                    "score": round(float(scores[i]), 4),
                }
            )
        return detections

    return detections


# ── CBVD-5 inference ──────────────────────────────────────────────────────────


def infer_cbvd5(model, out_dir: Path, conf_thresh: float):
    """
    Run inference on all CBVD-5 labelframes.
    Groups frames by video_id and saves one JSON per video.

    CBVD-5 filename: {video_id}_{timestamp:05d}.jpg
    e.g. 618_00002.jpg → video_id=618, timestamp=2
    """
    frames_dir = Path(DATASET_CONFIGS["cbvd5"]["frames_dir"])
    if not frames_dir.exists():
        print(f"  [WARNING] CBVD-5 frames not found: {frames_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Group all image files by video_id
    video_frames: dict[str, list] = {}
    for img_path in sorted(frames_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. "618_00002"
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        video_id = parts[0]  # "618"
        timestamp = int(parts[1])  # 2
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append((timestamp, img_path))

    print(f"\n  CBVD-5: {len(video_frames)} videos found")

    for video_id, frames in tqdm(video_frames.items(), desc="  cbvd5"):
        frames.sort(key=lambda x: x[0])  # sort by timestamp

        out_file = out_dir / f"{video_id}.json"
        if out_file.exists():
            continue  # skip already processed

        result = {
            "video_id": video_id,
            "dataset": "cbvd5",
            "fps": DATASET_CONFIGS["cbvd5"]["fps"],
            "width": DATASET_CONFIGS["cbvd5"]["width"],
            "height": DATASET_CONFIGS["cbvd5"]["height"],
            "frames": {},
        }

        for timestamp, img_path in frames:
            try:
                raw_output = model.predict(str(img_path), threshold=conf_thresh)
                dets = normalize_detections(raw_output)
            except Exception as e:
                print(f"    [WARNING] Failed on {img_path.name}: {e}")
                dets = []

            result["frames"][str(timestamp)] = dets

        with open(out_file, "w") as f:
            json.dump(result, f)

    print(f"  [OK] CBVD-5 detections saved to {out_dir}")


# ── CVB inference ─────────────────────────────────────────────────────────────


def infer_cvb(model, out_dir: Path, conf_thresh: float):
    """
    Run inference on all CVB raw_frames.
    Each sub-folder in raw_frames/ is one clip (video).
    Saves one JSON per clip.

    CVB filename: img_{frame:05d}.jpg (frames 1–450)
    """
    frames_root = Path(DATASET_CONFIGS["cvb"]["frames_dir"])
    if not frames_root.exists():
        print(f"  [WARNING] CVB frames not found: {frames_root}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    print(f"\n  CVB: {len(clip_dirs)} clips found")

    for clip_dir in tqdm(clip_dirs, desc="  cvb"):
        clip_id = clip_dir.name
        out_file = out_dir / f"{clip_id}.json"

        if out_file.exists():
            continue

        frame_files = sorted(clip_dir.glob("img_*.jpg"))
        if not frame_files:
            continue

        result = {
            "video_id": clip_id,
            "dataset": "cvb",
            "fps": DATASET_CONFIGS["cvb"]["fps"],
            "width": DATASET_CONFIGS["cvb"]["width"],
            "height": DATASET_CONFIGS["cvb"]["height"],
            "frames": {},
        }

        for img_path in frame_files:
            # Extract frame number from img_00001.jpg → "1"
            frame_num = int(img_path.stem.split("_")[1])

            try:
                raw_output = model.predict(str(img_path), threshold=conf_thresh)
                dets = normalize_detections(raw_output)
            except Exception as e:
                print(f"    [WARNING] Failed on {img_path.name}: {e}")
                dets = []

            result["frames"][str(frame_num)] = dets

        with open(out_file, "w") as f:
            json.dump(result, f)

    print(f"  [OK] CVB detections saved to {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on all dataset videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on both datasets
  python src/detection/infer_dataset.py \\
      --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth

  # Run on cbvd5 only
  python src/detection/infer_dataset.py \\
      --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \\
      --dataset cbvd5

  # Lower confidence threshold to catch more boxes
  python src/detection/infer_dataset.py \\
      --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \\
      --conf_thresh 0.3
        """,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--dataset",
        choices=["cbvd5", "cvb", "both"],
        default="both",
        help="Which dataset to run inference on (default: both)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/tracking",
        help="Output directory for detection JSONs",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    args = parser.parse_args()

    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)

    out_dir = Path(args.out_dir)
    print(f"Output dir: {out_dir}")
    print(f"Confidence threshold: {args.conf_thresh}\n")

    if args.dataset in ("cbvd5", "both"):
        infer_cbvd5(model, out_dir / "cbvd5", args.conf_thresh)

    if args.dataset in ("cvb", "both"):
        infer_cvb(model, out_dir / "cvb", args.conf_thresh)

    print("\n[DONE] All detection JSONs saved.")
    print(f"       Next step: python src/segmentation/segment.py")


if __name__ == "__main__":
    main()
