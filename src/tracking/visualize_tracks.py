"""
Phase 4 — Track Visualization
Cattle Vision Framework — Texas State University

Renders color-coded instance masks with persistent track IDs overlaid on
original video frames. Produces per-video frame grids and optionally a
side-by-side comparison strip for thesis figures.

Usage:
    # Visualize 5 CBVD-5 videos (all frames)
    python src/tracking/visualize_tracks.py --dataset cbvd5 --n_videos 5

    # Visualize specific video
    python src/tracking/visualize_tracks.py --dataset cbvd5 --video_id 100

    # CVB — visualize keyframes only (frames 1,15,30... to keep output manageable)
    python src/tracking/visualize_tracks.py --dataset cvb --n_videos 5 --stride 30

    # Generate thesis figure strip (3 frames per video, side by side)
    python src/tracking/visualize_tracks.py --dataset cbvd5 --n_videos 8 --mode strip
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils

    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[WARN] pycocotools not available — masks will not be rendered")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Color palette — 20 visually distinct colors for track IDs
# ══════════════════════════════════════════════════════════════════════════════

TRACK_COLORS = [
    (255, 80, 80),  # red
    (80, 200, 80),  # green
    (80, 130, 255),  # blue
    (255, 200, 50),  # yellow
    (200, 80, 255),  # purple
    (50, 220, 220),  # cyan
    (255, 140, 50),  # orange
    (255, 100, 180),  # pink
    (100, 255, 150),  # mint
    (180, 100, 255),  # lavender
    (255, 255, 100),  # lime
    (80, 180, 255),  # sky blue
    (255, 80, 150),  # hot pink
    (150, 255, 80),  # yellow-green
    (80, 255, 200),  # teal
    (255, 160, 80),  # peach
    (160, 80, 255),  # violet
    (255, 220, 150),  # cream
    (150, 220, 255),  # light blue
    (220, 150, 255),  # lilac
]


def get_color(track_id: int) -> tuple:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Frame path resolvers
# ══════════════════════════════════════════════════════════════════════════════


def get_frame_path_cbvd5(project_root: Path, video_id: str, frame_id: int) -> Path:
    """CBVD-5: labelframes/labelframes/{video_id}_{frame_idx:05d}.jpg"""
    return (
        project_root
        / "data"
        / "raw"
        / "cbvd5"
        / "labelframes"
        / "labelframes"
        / f"{video_id}_{frame_id:05d}.jpg"
    )


def get_frame_path_cvb(project_root: Path, video_id: str, frame_id: int) -> Path:
    """CVB: raw_frames/{video_id}/img_{frame_idx:05d}.jpg"""
    return (
        project_root
        / "data"
        / "raw"
        / "cvb"
        / "raw_frames"
        / video_id
        / f"img_{frame_id:05d}.jpg"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Core rendering
# ══════════════════════════════════════════════════════════════════════════════


def decode_rle(rle: dict) -> np.ndarray | None:
    if not HAS_COCO or rle is None:
        return None
    rle_copy = {
        "size": rle["size"],
        "counts": (
            rle["counts"].encode() if isinstance(rle["counts"], str) else rle["counts"]
        ),
    }
    return mask_utils.decode(rle_copy).astype(np.uint8)


def render_frame(
    img: np.ndarray,
    detections: list,
    alpha: float = 0.45,
    box_thickness: int = 2,
    font_scale: float = 0.7,
) -> np.ndarray:
    """
    Overlay color-coded masks and track ID labels on one frame.

    Args:
        img:        BGR image (H×W×3)
        detections: list of {track_id, bbox:[x1,y1,x2,y2], score, mask_rle}
        alpha:      mask transparency (0=invisible, 1=opaque)

    Returns:
        Annotated BGR image
    """
    out = img.copy()
    overlay = img.copy()

    for det in detections:
        track_id = det["track_id"]
        color = get_color(track_id)
        bbox = det["bbox"]  # [x1, y1, x2, y2] pixels
        rle = det.get("mask_rle")

        # ── Draw mask ──────────────────────────────────────────────────────────
        if rle is not None:
            mask = decode_rle(rle)
            if mask is not None:
                colored = np.zeros_like(img, dtype=np.uint8)
                colored[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        # ── Draw bounding box ─────────────────────────────────────────────────
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

        # ── Draw track ID label ───────────────────────────────────────────────
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


def make_frame_grid(frames: list, cols: int = 3, target_w: int = 640) -> np.ndarray:
    """
    Arrange a list of BGR images into a grid.
    All images are resized to target_w wide (preserving aspect ratio).
    """
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to same width
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        new_h = int(h * target_w / w)
        resized.append(cv2.resize(f, (target_w, new_h)))

    cell_h = resized[0].shape[0]
    cell_w = target_w
    rows = (len(resized) + cols - 1) // cols

    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, img in enumerate(resized):
        r, c = divmod(i, cols)
        grid[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = img

    return grid


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Per-video visualization
# ══════════════════════════════════════════════════════════════════════════════


def visualize_video(
    track_path: Path,
    project_root: Path,
    out_dir: Path,
    dataset: str,
    stride: int = 1,
    mode: str = "grid",
    grid_cols: int = 3,
    target_w: int = 640,
) -> bool:
    """
    Render tracking overlays for one video.

    Saves:
      - {out_dir}/{video_id}_tracks_grid.jpg   (all frames in a grid)
      - {out_dir}/{video_id}_strip.jpg          (3-frame strip, mode='strip')

    Returns True if successful.
    """
    with open(track_path) as f:
        data = json.load(f)

    video_id = data["video_id"]
    frames = data["frames"]
    sorted_fids = sorted(frames.keys(), key=lambda x: int(x))

    # Select frames to render
    if stride > 1:
        sorted_fids = sorted_fids[::stride]

    rendered_frames = []
    missing_count = 0

    for fid_str in sorted_fids:
        frame_id = int(fid_str)
        dets = frames[fid_str]

        # Resolve image path
        if dataset == "cbvd5":
            img_path = get_frame_path_cbvd5(project_root, video_id, frame_id)
        else:
            img_path = get_frame_path_cvb(project_root, video_id, frame_id)

        if not img_path.exists():
            missing_count += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            missing_count += 1
            continue

        annotated = render_frame(img, dets)

        # Add frame label
        label = f"Frame {frame_id}  |  {len(dets)} tracks"
        cv2.putText(
            annotated,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        rendered_frames.append(annotated)

    if not rendered_frames:
        print(f"  [SKIP] {video_id} — no frames found (missing: {missing_count})")
        return False

    if mode == "strip":
        # Pick 3 evenly spaced frames for a compact thesis strip
        n = len(rendered_frames)
        indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        strip_frames = [rendered_frames[i] for i in indices]
        grid = make_frame_grid(strip_frames, cols=3, target_w=target_w)
        out_path = out_dir / f"{video_id}_strip.jpg"
    else:
        grid = make_frame_grid(rendered_frames, cols=grid_cols, target_w=target_w)
        out_path = out_dir / f"{video_id}_tracks_grid.jpg"

    # Add video ID header bar
    header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    cv2.putText(
        header,
        f"Video: {video_id}  |  Dataset: {dataset.upper()}  |  "
        f"Tracks: {data['stats']['total_unique_tracks']}",
        (10, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    final = np.vstack([header, grid])

    cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(
        f"  Saved: {out_path.name}  "
        f"({len(rendered_frames)} frames, {missing_count} missing)"
    )
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Track Visualization")
    parser.add_argument("--dataset", required=True, choices=["cbvd5", "cvb"])
    parser.add_argument(
        "--n_videos",
        type=int,
        default=5,
        help="Number of videos to visualize (random sample)",
    )
    parser.add_argument(
        "--video_id", default=None, help="Visualize a specific video ID only"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (e.g. 30 = every 30th frame for CVB)",
    )
    parser.add_argument(
        "--mode",
        default="grid",
        choices=["grid", "strip"],
        help="grid=all frames in grid, strip=3-frame thesis strip",
    )
    parser.add_argument("--cols", type=int, default=3, help="Columns in grid layout")
    parser.add_argument(
        "--width", type=int, default=640, help="Width of each frame cell in pixels"
    )
    parser.add_argument("--track_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    track_dir = (
        Path(args.track_dir)
        if args.track_dir
        else project_root / "data" / "processed" / "tracking_v2" / args.dataset
    )
    out_dir = project_root / "results" / "tracking" / "visualizations" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect track files
    track_files = sorted(track_dir.glob("*_tracks.json"))
    if not track_files:
        print(f"[ERROR] No track files found in {track_dir}")
        return

    # Filter or sample
    if args.video_id:
        track_files = [f for f in track_files if args.video_id in f.stem]
    else:
        random.seed(args.seed)
        # Prefer videos with tracks > 0
        with_tracks = []
        for tf in track_files:
            try:
                with open(tf) as f:
                    d = json.load(f)
                if d["stats"]["total_unique_tracks"] > 0:
                    with_tracks.append(tf)
            except Exception:
                pass
        pool = with_tracks if with_tracks else track_files
        track_files = random.sample(pool, min(args.n_videos, len(pool)))
        track_files = sorted(track_files)

    print(f"\n{'='*60}")
    print(f"Phase 4 — Track Visualization")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Videos    : {len(track_files)}")
    print(f"  Mode      : {args.mode}")
    print(f"  Stride    : {args.stride}")
    print(f"  Output    : {out_dir}")
    print(f"{'='*60}\n")

    success = 0
    for tf in track_files:
        ok = visualize_video(
            track_path=tf,
            project_root=project_root,
            out_dir=out_dir,
            dataset=args.dataset,
            stride=args.stride,
            mode=args.mode,
            grid_cols=args.cols,
            target_w=args.width,
        )
        if ok:
            success += 1

    print(f"\nDone. {success}/{len(track_files)} videos rendered.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
