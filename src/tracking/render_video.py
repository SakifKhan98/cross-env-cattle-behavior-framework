"""
Phase 4 — Tracking Video Renderer
Cattle Vision Framework — Texas State University

Renders a tracking overlay video (MP4) for one video clip using
OpenCV's VideoWriter (no ffmpeg required).

Usage:
    # Render a specific CVB video
    python src/tracking/render_video.py --dataset cvb \
        --video_id 0089_arm01_gopro1_20200323_002948_beh7_ani2_ins1_cut_2

    # Render a specific CBVD-5 video (slow slideshow at 2fps)
    python src/tracking/render_video.py --dataset cbvd5 --video_id 329

    # Auto-pick the best video from each dataset
    python src/tracking/render_video.py --dataset cvb --auto
    python src/tracking/render_video.py --dataset cbvd5 --auto
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils

    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[WARN] pycocotools not available — masks will not be rendered")

# ── Reuse color palette and render_frame from visualize_tracks ────────────────
TRACK_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 130, 255),
    (255, 200, 50),
    (200, 80, 255),
    (50, 220, 220),
    (255, 140, 50),
    (255, 100, 180),
    (100, 255, 150),
    (180, 100, 255),
    (255, 255, 100),
    (80, 180, 255),
    (255, 80, 150),
    (150, 255, 80),
    (80, 255, 200),
    (255, 160, 80),
    (160, 80, 255),
    (255, 220, 150),
    (150, 220, 255),
    (220, 150, 255),
]


def get_color(track_id):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


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


def render_frame(img, detections, alpha=0.45, box_thickness=2, font_scale=0.65):
    overlay = img.copy()
    for det in detections:
        track_id = det["track_id"]
        color = get_color(track_id)
        bbox = det["bbox"]
        rle = det.get("mask_rle")

        if rle is not None:
            mask = decode_rle(rle)
            if mask is not None:
                # Resize mask to match scaled image if dimensions differ
                if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
                    mask = cv2.resize(
                        mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                colored = np.zeros_like(img, dtype=np.uint8)
                colored[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

        label = f"ID {track_id}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        lx = max(x1, 0)
        ly = max(y1 - 6, th + baseline)
        cv2.rectangle(
            overlay, (lx, ly - th - baseline), (lx + tw + 4, ly + baseline), color, -1
        )
        cv2.putText(
            overlay,
            label,
            (lx + 2, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_hud(img, frame_id, total_frames, n_tracks, video_id, dataset, fps):
    """Draw heads-up display bar at top of frame."""
    h, w = img.shape[:2]
    bar_h = 44
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)

    # Progress bar
    progress = int((frame_id / max(total_frames, 1)) * w)
    cv2.rectangle(bar, (0, bar_h - 4), (progress, bar_h), (80, 160, 255), -1)

    # Text
    left_text = f"Cattle Vision Framework  |  {dataset.upper()}  |  {video_id}"
    right_text = (
        f"Frame {frame_id:4d}/{total_frames}  |  Tracks: {n_tracks}  |  {fps:.0f} FPS"
    )

    cv2.putText(
        bar,
        left_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        bar,
        right_text,
        (w - rw - 10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([bar, img])


def get_frame_path(project_root, dataset, video_id, frame_id):
    if dataset == "cbvd5":
        return (
            project_root
            / "data"
            / "raw"
            / "cbvd5"
            / "labelframes"
            / "labelframes"
            / f"{video_id}_{frame_id:05d}.jpg"
        )
    else:
        return (
            project_root
            / "data"
            / "raw"
            / "cvb"
            / "raw_frames"
            / video_id
            / f"img_{frame_id:05d}.jpg"
        )


def find_best_video(track_dir, min_tracks=6):
    """Pick the video with most consistent tracks and all frames present."""
    best_path = None
    best_score = -1
    for tf in sorted(track_dir.glob("*_tracks.json")):
        try:
            with open(tf) as f:
                d = json.load(f)
            n_tracks = d["stats"]["total_unique_tracks"]
            n_frames_with_tracks = d["stats"]["frames_with_tracks"]
            score = n_tracks * n_frames_with_tracks
            if n_tracks >= min_tracks and score > best_score:
                best_score = score
                best_path = tf
        except Exception:
            continue
    return best_path


def render_video(
    track_path, project_root, out_dir, dataset, output_fps=30, cbvd5_fps=2, max_dim=1280
):
    with open(track_path) as f:
        data = json.load(f)

    video_id = data["video_id"]
    frames_dict = data["frames"]
    sorted_fids = sorted(frames_dict.keys(), key=lambda x: int(x))
    total_frames = len(sorted_fids)

    fps = cbvd5_fps if dataset == "cbvd5" else output_fps
    out_path = out_dir / f"{video_id}_tracked.mp4"

    print(f"\n{'='*60}")
    print(f"Rendering: {video_id}")
    print(f"  Dataset : {dataset.upper()}")
    print(f"  Frames  : {total_frames}")
    print(f"  Tracks  : {data['stats']['total_unique_tracks']}")
    print(f"  FPS     : {fps}")
    print(f"  Output  : {out_path}")
    print(f"{'='*60}")

    # ── Determine output resolution from first available frame ───────────────
    writer = None
    out_w, out_h = None, None

    for fid_str in sorted_fids:
        frame_id = int(fid_str)
        img_path = get_frame_path(project_root, dataset, video_id, frame_id)
        if img_path.exists():
            sample = cv2.imread(str(img_path))
            if sample is not None:
                h, w = sample.shape[:2]
                # Scale down if larger than max_dim
                scale = min(max_dim / w, max_dim / h, 1.0)
                out_w = int(w * scale)
                out_h = int(h * scale) + 44  # +44 for HUD bar
                break

    if out_w is None:
        print("[ERROR] No readable frames found.")
        return False

    # Try mp4v codec first, fallback to XVID
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("[WARN] mp4v failed, trying XVID...")
        out_path = out_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter.")
        return False

    # ── Render frames ────────────────────────────────────────────────────────
    written = 0
    skipped = 0

    for i, fid_str in enumerate(sorted_fids):
        frame_id = int(fid_str)
        dets = frames_dict[fid_str]
        img_path = get_frame_path(project_root, dataset, video_id, frame_id)

        if not img_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        # Scale if needed
        h, w = img.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            # Scale detection bboxes to match
            scaled_dets = []
            for det in dets:
                sd = det.copy()
                sd["bbox"] = [c * scale for c in det["bbox"]]
                # mask_rle size stays the same — we re-decode at original size
                # then resize the mask
                scaled_dets.append(sd)
            dets = scaled_dets

        # Render tracking overlay
        annotated = render_frame(img, dets)

        # Add HUD
        frame_with_hud = draw_hud(
            annotated, frame_id, total_frames, len(dets), video_id, dataset, fps
        )

        # Ensure exact output size (HUD addition may vary by 1px)
        if frame_with_hud.shape[1] != out_w or frame_with_hud.shape[0] != out_h:
            frame_with_hud = cv2.resize(frame_with_hud, (out_w, out_h))

        writer.write(frame_with_hud)
        written += 1

        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            pct = (i + 1) / total_frames * 100
            print(
                f"  [{i+1:4d}/{total_frames}]  {pct:.0f}%  written={written}  skipped={skipped}"
            )

    writer.release()
    print(f"\nDone. Written: {written} frames  |  Skipped: {skipped}")
    print(f"Output: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Tracking Video Renderer")
    parser.add_argument("--dataset", required=True, choices=["cbvd5", "cvb"])
    parser.add_argument("--video_id", default=None, help="Specific video ID to render")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-select best video by track count"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output FPS for CVB (CBVD-5 always uses 2fps)",
    )
    parser.add_argument(
        "--max_dim",
        type=int,
        default=1280,
        help="Max output dimension (width or height)",
    )
    parser.add_argument("--track_dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    track_dir = (
        Path(args.track_dir)
        if args.track_dir
        else project_root / "data" / "processed" / "tracking_v2" / args.dataset
    )
    out_dir = project_root / "results" / "tracking" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find target track file
    if args.auto:
        track_path = find_best_video(track_dir)
        if track_path is None:
            print("[ERROR] Could not find a suitable video.")
            sys.exit(1)
        print(f"Auto-selected: {track_path.stem}")
    elif args.video_id:
        matches = list(track_dir.glob(f"*{args.video_id}*_tracks.json"))
        if not matches:
            print(f"[ERROR] No track file found for video_id: {args.video_id}")
            sys.exit(1)
        track_path = matches[0]
    else:
        print("[ERROR] Provide --video_id or --auto")
        sys.exit(1)

    render_video(
        track_path=track_path,
        project_root=project_root,
        out_dir=out_dir,
        dataset=args.dataset,
        output_fps=args.fps,
        max_dim=args.max_dim,
    )


if __name__ == "__main__":
    main()
