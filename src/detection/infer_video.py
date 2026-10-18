"""
src/detection/infer_video.py

Runs the trained RF-DETR cattle detector on a SINGLE video file and
saves an annotated output video with bounding boxes drawn.

Use this for:
  - Qualitative inspection of detection quality
  - Generating demo videos for the thesis / paper figures
  - Debugging detection on a specific problem video

Output:
  - Annotated video: {output_dir}/{video_stem}_detected.mp4
  - Optional JSON:   {output_dir}/{video_stem}_detections.json
                     (same format as infer_dataset.py output)

Usage:
    python src/detection/infer_video.py \\
        --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \\
        --video data/raw/cbvd5/videos/videos/618.mp4

    # Save detection JSON alongside the annotated video
    python src/detection/infer_video.py \\
        --checkpoint <path> --video <path> --save_json

    # Override confidence threshold
    python src/detection/infer_video.py \\
        --checkpoint <path> --video <path> --conf_thresh 0.4
"""

import argparse
import json
import sys
import time
from pathlib import Path


# ── Model loader ──────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
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
    print(f"  [OK] Loaded: {ckpt.name}")
    return model


# ── Normalize detections ──────────────────────────────────────────────────────


def normalize_detections(raw_output):
    """
    Handles multiple possible RF-DETR output formats.
    Returns list of {"bbox": [x, y, w, h], "score": float}.
    """
    detections = []

    # supervision DetectionResult
    if hasattr(raw_output, "xyxy") and hasattr(raw_output, "confidence"):
        for i in range(len(raw_output.xyxy)):
            x1, y1, x2, y2 = raw_output.xyxy[i].tolist()
            detections.append(
                {
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                    "score": round(float(raw_output.confidence[i]), 4),
                }
            )
        return detections

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


# ── Drawing ───────────────────────────────────────────────────────────────────


def draw_detections(frame, detections: list, conf_thresh: float):
    """
    Draw bounding boxes and confidence scores on a frame (numpy array).
    Returns the annotated frame.
    """
    try:
        import cv2
    except ImportError:
        return frame  # return unmodified if cv2 not available

    BOX_COLOR = (0, 255, 0)  # green
    TEXT_COLOR = (0, 255, 0)
    THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55

    for det in detections:
        if det["score"] < conf_thresh:
            continue
        x, y, w, h = [int(v) for v in det["bbox"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, THICKNESS)
        label = f"{det['score']:.2f}"
        cv2.putText(
            frame, label, (x, max(y - 6, 10)), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS
        )

    return frame


# ── Main inference loop ───────────────────────────────────────────────────────


def infer_video(
    model,
    video_path: str,
    out_dir: Path,
    conf_thresh: float,
    save_json: bool,
):
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python is not installed. Run: pip install opencv-python")
        sys.exit(1)

    video = Path(video_path)
    if not video.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Video   : {video.name}")
    print(f"  Size    : {width}x{height} @ {fps:.1f} fps")
    print(f"  Frames  : {total}")
    print(f"  Output  : {out_dir}\n")

    # Output video writer
    out_video_path = out_dir / f"{video.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    all_detections = {
        "video_id": video.stem,
        "dataset": "unknown",
        "fps": fps,
        "width": width,
        "height": height,
        "frames": {},
    }

    frame_idx = 0
    total_dets = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame temporarily for RF-DETR predict()
        tmp_path = out_dir / "_tmp_frame.jpg"
        cv2.imwrite(str(tmp_path), frame)

        try:
            raw_output = model.predict(str(tmp_path), threshold=conf_thresh)
            dets = normalize_detections(raw_output)
        except Exception as e:
            print(f"  [WARNING] Frame {frame_idx}: {e}")
            dets = []

        all_detections["frames"][str(frame_idx)] = dets
        total_dets += len(dets)

        annotated = draw_detections(frame.copy(), dets, conf_thresh)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_idx / elapsed if elapsed > 0 else 0
            print(
                f"  Frame {frame_idx}/{total} | "
                f"{fps_processing:.1f} fps | "
                f"{total_dets} total detections"
            )

    cap.release()
    writer.release()

    # Clean up temp file
    tmp_path = out_dir / "_tmp_frame.jpg"
    if tmp_path.exists():
        tmp_path.unlink()

    elapsed = time.time() - start_time
    avg_dets = total_dets / max(frame_idx, 1)

    print(f"\n  [OK] Processed {frame_idx} frames in {elapsed:.1f}s")
    print(f"       Avg detections/frame: {avg_dets:.2f}")
    print(f"       Annotated video: {out_video_path}")

    if save_json:
        json_path = out_dir / f"{video.stem}_detections.json"
        with open(json_path, "w") as f:
            json.dump(all_detections, f)
        print(f"       Detection JSON : {json_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR on a single video, save annotated output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic — annotated video only
  python src/detection/infer_video.py \\
      --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \\
      --video data/raw/cbvd5/videos/videos/618.mp4

  # Also save detection JSON
  python src/detection/infer_video.py \\
      --checkpoint <path> --video <path> --save_json

  # Custom output folder
  python src/detection/infer_video.py \\
      --checkpoint <path> --video <path> --out_dir results/demo_videos
        """,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--video", required=True, help="Path to input video file (.mp4)"
    )
    parser.add_argument(
        "--out_dir",
        default="results/detection",
        help="Output directory for annotated video (default: results/detection)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Also save detections as JSON alongside the video",
    )
    args = parser.parse_args()

    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)

    infer_video(
        model=model,
        video_path=args.video,
        out_dir=Path(args.out_dir),
        conf_thresh=args.conf_thresh,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
