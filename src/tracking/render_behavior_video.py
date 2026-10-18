"""
Full-Pipeline Behavior Video Renderer — Cattle Vision Framework

Renders an annotated MP4 combining:
  - OC-SORT tracking bboxes + masks (from tracking_v2/ JSON)
  - VideoMAE behavior predictions (from results/behavior/predictions_sam2_tubelets/*.csv)

Each tracked cow is shown with its predicted behavior label (color-coded).
Overlapping tubelets are resolved per-frame by averaging logits then argmaxing.

Usage:
    # CVB video with combined-model predictions
    python src/tracking/render_behavior_video.py \
        --dataset cvb \
        --video_id 0089_arm01_gopro1_20200323_002948_beh7_ani2_ins1_cut_2 \
        --predictions results/behavior/predictions_sam2_tubelets/videomae_combined_v1_val.csv

    # Auto-select best CVB video
    python src/tracking/render_behavior_video.py \
        --dataset cvb --auto \
        --predictions results/behavior/predictions_sam2_tubelets/videomae_combined_v1_val.csv

    # Use a specific config's predictions
    python src/tracking/render_behavior_video.py \
        --dataset cvb --auto \
        --predictions results/behavior/predictions_sam2_tubelets/videomae_cvb_v1_val.csv
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _reencode_h264(src: Path, dst: Path) -> bool:
    """Re-encode src (any codec) to H.264 at dst using ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(dst),
        ],
        capture_output=True,
    )
    return result.returncode == 0

try:
    from pycocotools import mask as mask_utils
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[WARN] pycocotools not available — masks will not be rendered")

# Behavior colors (BGR) — distinct palette per class
BEHAVIOR_COLORS = {
    0: (80,  200, 80),   # Standing   — green
    1: (80,  130, 255),  # Lying      — blue
    2: (50,  220, 220),  # Foraging   — cyan
    3: (255, 200, 50),   # Drinking   — yellow
    4: (200, 80,  255),  # Ruminating — purple
    5: (255, 140, 50),   # Grooming   — orange
    6: (160, 160, 160),  # Other      — grey
}

LABEL_NAMES = {
    0: "Standing",
    1: "Lying",
    2: "Foraging",
    3: "Drinking",
    4: "Ruminating",
    5: "Grooming",
    6: "Other",
}

FALLBACK_COLOR = (200, 200, 200)


def get_color(label_id):
    return BEHAVIOR_COLORS.get(label_id, FALLBACK_COLOR)


def decode_rle(rle):
    if not HAS_COCO or rle is None:
        return None
    rle_copy = {
        "size": rle["size"],
        "counts": (
            rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
        ),
    }
    return mask_utils.decode(rle_copy).astype(np.uint8)


def load_predictions(pred_csv: Path, dataset: str, video_id: str):
    """
    Load predictions CSV and build:
        frame_labels[track_id][frame_idx] -> averaged logits array [7]

    Tubelet frames span [start_frame, end_frame) (both inclusive in labels.csv).
    Overlapping tubelets (stride 8, length 16) accumulate logits; argmax at render time.
    """
    import csv

    # track_id parsed from tubelet_dir:
    #   CVB  : .../track_0042/tubelet_0003  -> track_id=42
    #   CBVD5: .../kf50_instABCD            -> no track_id; use inst hash as proxy

    frame_logits = defaultdict(lambda: defaultdict(list))  # [track_id][frame_idx] -> [logits]

    with open(pred_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != dataset:
                continue
            if row["video_id"] != video_id:
                continue

            tdir = row["tubelet_dir"]
            # Parse track_id from tubelet_dir
            m = re.search(r"track_(\d+)", tdir)
            if m:
                track_id = int(m.group(1))
            else:
                # CBVD5 — use inst hash as pseudo-id
                m2 = re.search(r"inst([0-9a-f]+)", tdir)
                track_id = int(m2.group(1)[:6], 16) % 10000 if m2 else -1

            start = int(row["start_frame"])
            end   = int(row["end_frame"])

            # Parse logits (columns logit_0 ... logit_N)
            logit_keys = sorted(
                [k for k in row.keys() if k.startswith("logit_")],
                key=lambda k: int(k.split("_")[1]),
            )
            logits = np.array([float(row[k]) for k in logit_keys], dtype=np.float32)

            for frame_idx in range(start, end + 1):
                frame_logits[track_id][frame_idx].append(logits)

    # Average logits per (track, frame)
    frame_labels = defaultdict(dict)  # [track_id][frame_idx] -> label_id
    frame_conf   = defaultdict(dict)  # [track_id][frame_idx] -> confidence

    for track_id, fmap in frame_logits.items():
        for frame_idx, logit_list in fmap.items():
            avg_logits = np.mean(logit_list, axis=0)
            label_id   = int(np.argmax(avg_logits))
            # softmax confidence
            exp_l = np.exp(avg_logits - avg_logits.max())
            conf  = float(exp_l[label_id] / exp_l.sum())
            frame_labels[track_id][frame_idx] = label_id
            frame_conf[track_id][frame_idx]   = conf

    return frame_labels, frame_conf


def render_frame(img, detections, frame_labels, frame_conf, frame_id,
                 alpha=0.40, box_thickness=2, font_scale=0.60):
    overlay = img.copy()

    for det in detections:
        track_id = det["track_id"]
        bbox     = det["bbox"]
        rle      = det.get("mask_rle")

        label_id = frame_labels.get(track_id, {}).get(frame_id)
        conf     = frame_conf.get(track_id, {}).get(frame_id, 0.0)
        color    = get_color(label_id) if label_id is not None else FALLBACK_COLOR

        # Mask overlay
        if rle is not None:
            mask = decode_rle(rle)
            if mask is not None:
                if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                colored = np.zeros_like(img, dtype=np.uint8)
                colored[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        # Bounding box
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

        # Label text: behavior name + confidence, or "Unknown" for short-track animals
        if label_id is not None:
            label_str = f"{LABEL_NAMES.get(label_id, '?')} {conf:.0%}"
        else:
            label_str = "Unknown"

        (tw, th), baseline = cv2.getTextSize(
            label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        lx = max(x1, 0)
        ly = max(y1 - 6, th + baseline)
        cv2.rectangle(overlay, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline),
                      color, -1)
        cv2.putText(overlay, label_str, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

        # Small track ID in corner of bbox
        tid_str = f"#{track_id}"
        cv2.putText(overlay, tid_str, (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return overlay


def draw_legend(img, active_labels):
    """Draw a compact behavior legend on the right edge."""
    if not active_labels:
        return img
    h, w = img.shape[:2]
    pad   = 8
    lh    = 22
    lw    = 140
    total = len(active_labels) * lh + 2 * pad
    ly0   = h // 2 - total // 2
    for i, label_id in enumerate(sorted(active_labels)):
        color = get_color(label_id)
        name  = LABEL_NAMES.get(label_id, "?")
        y     = ly0 + i * lh + pad
        cv2.rectangle(img, (w - lw - pad, y), (w - pad, y + lh - 2), (20, 20, 20), -1)
        cv2.rectangle(img, (w - lw - pad, y), (w - lw - pad + 14, y + lh - 2), color, -1)
        cv2.putText(img, name, (w - lw - pad + 18, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def draw_hud(img, frame_id, total_frames, n_tracks, video_id, dataset, fps, model_tag):
    h, w  = img.shape[:2]
    bar_h = 44
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    progress = int((frame_id / max(total_frames, 1)) * w)
    cv2.rectangle(bar, (0, bar_h - 4), (progress, bar_h), (80, 160, 255), -1)

    left  = f"Cattle Vision Framework  |  {dataset.upper()}  |  {video_id}"
    right = f"Frame {frame_id:4d}/{total_frames}  |  Tracks: {n_tracks}  |  {model_tag}"

    cv2.putText(bar, left,  (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (200, 200, 200), 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(bar, right, (w - rw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (180, 220, 255), 1, cv2.LINE_AA)

    return np.vstack([bar, img])


def get_frame_path(project_root, dataset, video_id, frame_id):
    if dataset == "cbvd5":
        return (project_root / "data" / "raw" / "cbvd5" / "labelframes"
                / "labelframes" / f"{video_id}_{frame_id:05d}.jpg")
    return (project_root / "data" / "raw" / "cvb" / "raw_frames"
            / video_id / f"img_{frame_id:05d}.jpg")


def find_best_video(track_dir, min_tracks=6):
    best_path, best_score = None, -1
    for tf in sorted(track_dir.glob("*_tracks.json")):
        try:
            with open(tf) as f:
                d = json.load(f)
            n   = d["stats"]["total_unique_tracks"]
            nf  = d["stats"]["frames_with_tracks"]
            score = n * nf
            if n >= min_tracks and score > best_score:
                best_score = score
                best_path  = tf
        except Exception:
            continue
    return best_path


def render_inference_video(
    video_path,
    tracks_json: dict,
    predictions_csv,
    output_path,
    fps: float,
    job_id: str = "inference",
    max_dim: int = 1280,
) -> bool:
    """Render an annotated behavior video for a single-video inference job.

    Reads frames directly from video_path — no dataset-specific frame directories.
    Short-track animals (no predictions) are rendered with label "Unknown" in grey.

    Args:
        video_path:      Source video file (any OpenCV-readable format).
        tracks_json:     Tracks dict (run_tracking output / tracks.json).
        predictions_csv: Path to predictions.csv from classify_tubelets.
        output_path:     Destination MP4 path.
        fps:             Output video frame rate (from VideoIngestor header).
        job_id:          Used as video_id when loading predictions.
        max_dim:         Max output dimension in pixels.

    Returns:
        True on success, False on error.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    predictions_csv = Path(predictions_csv)

    frames_dict = tracks_json.get("frames", {})
    total_frames = int(tracks_json.get("stats", {}).get("total_frames", len(frames_dict)))

    print(f"  Loading predictions...", end=" ", flush=True)
    frame_labels, frame_conf = load_predictions(predictions_csv, "inference", job_id)
    print(f"done. {len(frame_labels)} tracks have predictions.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    ret, sample = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        cap.release()
        return False

    h, w = sample.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    out_w = int(w * scale)
    out_h = int(h * scale) + 44
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter")
        cap.release()
        return False

    model_tag = predictions_csv.stem
    active_labels_seen: set = set()
    written = 0
    frame_id = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        dets = frames_dict.get(str(frame_id), [])

        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            dets = [{**d, "bbox": [c * scale for c in d["bbox"]]} for d in dets]

        for det in dets:
            lbl = frame_labels.get(det["track_id"], {}).get(frame_id)
            if lbl is not None:
                active_labels_seen.add(lbl)

        annotated = render_frame(img, dets, frame_labels, frame_conf, frame_id)
        draw_legend(annotated, active_labels_seen)
        frame_with_hud = draw_hud(
            annotated, frame_id, total_frames, len(dets),
            job_id, "inference", fps, model_tag,
        )

        if frame_with_hud.shape[1] != out_w or frame_with_hud.shape[0] != out_h:
            frame_with_hud = cv2.resize(frame_with_hud, (out_w, out_h))

        writer.write(frame_with_hud)
        written += 1
        frame_id += 1

    writer.release()
    cap.release()
    print(f"  Render complete: {written} frames → {tmp_path}")

    # Re-encode to H.264 so browsers can play the video inline.
    print("  Re-encoding to H.264...", end=" ", flush=True)
    if _reencode_h264(tmp_path, output_path):
        tmp_path.unlink(missing_ok=True)
        print("done.")
    else:
        print("ffmpeg unavailable — keeping mp4v output.")
        tmp_path.rename(output_path)
    return True


def render_behavior_video(track_path, project_root, out_dir, dataset,
                          pred_csv, output_fps=30, cbvd5_fps=2, max_dim=1280):
    with open(track_path) as f:
        data = json.load(f)

    video_id    = data["video_id"]
    frames_dict = data["frames"]
    sorted_fids = sorted(frames_dict.keys(), key=lambda x: int(x))
    total_frames = len(sorted_fids)
    fps          = cbvd5_fps if dataset == "cbvd5" else output_fps

    model_tag = Path(pred_csv).stem  # e.g. videomae_combined_v1_val

    print(f"\n{'='*65}")
    print(f"Rendering behavior video: {video_id}")
    print(f"  Dataset   : {dataset.upper()}")
    print(f"  Frames    : {total_frames}")
    print(f"  Tracks    : {data['stats']['total_unique_tracks']}")
    print(f"  FPS       : {fps}")
    print(f"  Model     : {model_tag}")
    print(f"{'='*65}")

    print("Loading predictions...", end=" ", flush=True)
    frame_labels, frame_conf = load_predictions(pred_csv, dataset, video_id)
    covered_tracks = set(frame_labels.keys())
    print(f"done. {len(covered_tracks)} tracks have predictions.")

    if not covered_tracks:
        print("[WARN] No predictions found for this video. "
              "Run evaluate.py first, or check --predictions path and dataset/split.")

    # Determine output resolution from first readable frame
    out_w = out_h = None
    for fid_str in sorted_fids:
        img_path = get_frame_path(project_root, dataset, video_id, int(fid_str))
        if img_path.exists():
            sample = cv2.imread(str(img_path))
            if sample is not None:
                h, w = sample.shape[:2]
                scale = min(max_dim / w, max_dim / h, 1.0)
                out_w = int(w * scale)
                out_h = int(h * scale) + 44  # +44 HUD
                break

    if out_w is None:
        print("[ERROR] No readable frames found.")
        return False

    out_path = out_dir / f"{video_id}_behavior.mp4"
    tmp_path_b = out_path.with_suffix(".tmp.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(tmp_path_b), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter.")
        return False

    written, skipped = 0, 0
    active_labels_seen = set()

    for i, fid_str in enumerate(sorted_fids):
        frame_id = int(fid_str)
        dets     = frames_dict[fid_str]
        img_path = get_frame_path(project_root, dataset, video_id, frame_id)

        if not img_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w  = img.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            scaled_dets = []
            for det in dets:
                sd = det.copy()
                sd["bbox"] = [c * scale for c in det["bbox"]]
                scaled_dets.append(sd)
            dets = scaled_dets

        # Collect active behavior labels for legend
        for det in dets:
            tid = det["track_id"]
            lbl = frame_labels.get(tid, {}).get(frame_id)
            if lbl is not None:
                active_labels_seen.add(lbl)

        annotated = render_frame(img, dets, frame_labels, frame_conf, frame_id)
        draw_legend(annotated, active_labels_seen)

        frame_with_hud = draw_hud(
            annotated, frame_id, total_frames, len(dets), video_id, dataset,
            fps, model_tag)

        if frame_with_hud.shape[1] != out_w or frame_with_hud.shape[0] != out_h:
            frame_with_hud = cv2.resize(frame_with_hud, (out_w, out_h))

        writer.write(frame_with_hud)
        written += 1

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            pct = (i + 1) / total_frames * 100
            print(f"  [{i+1:4d}/{total_frames}]  {pct:.0f}%  written={written}  skipped={skipped}")

    writer.release()
    print(f"\nDone. Written: {written}  Skipped: {skipped}")

    print("Re-encoding to H.264...", end=" ", flush=True)
    if _reencode_h264(tmp_path_b, out_path):
        tmp_path_b.unlink(missing_ok=True)
        print("done.")
    else:
        print("ffmpeg unavailable — keeping mp4v output.")
        tmp_path_b.rename(out_path)
    print(f"Output: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Full-Pipeline Behavior Video Renderer")
    parser.add_argument("--dataset",     required=True, choices=["cbvd5", "cvb"])
    parser.add_argument("--video_id",    default=None)
    parser.add_argument("--auto",        action="store_true",
                        help="Auto-select best video by track density")
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions CSV from evaluate.py")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--max_dim",     type=int, default=1280)
    parser.add_argument("--track_dir",   default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    track_dir = (
        Path(args.track_dir) if args.track_dir
        else project_root / "data" / "processed" / "tracking_v2" / args.dataset
    )
    out_dir = project_root / "results" / "tracking" / "behavior_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_csv = Path(args.predictions)
    if not pred_csv.is_absolute():
        pred_csv = project_root / pred_csv
    if not pred_csv.exists():
        print(f"[ERROR] Predictions file not found: {pred_csv}")
        print("  Run evaluate.py first to generate predictions CSV.")
        sys.exit(1)

    if args.auto:
        track_path = find_best_video(track_dir)
        if track_path is None:
            print("[ERROR] No suitable video found.")
            sys.exit(1)
        print(f"Auto-selected: {track_path.stem}")
    elif args.video_id:
        matches = list(track_dir.glob(f"*{args.video_id}*_tracks.json"))
        if not matches:
            print(f"[ERROR] No track file for video_id: {args.video_id}")
            sys.exit(1)
        track_path = matches[0]
    else:
        print("[ERROR] Provide --video_id or --auto")
        sys.exit(1)

    render_behavior_video(
        track_path=track_path,
        project_root=project_root,
        out_dir=out_dir,
        dataset=args.dataset,
        pred_csv=pred_csv,
        output_fps=args.fps,
        max_dim=args.max_dim,
    )


if __name__ == "__main__":
    main()
