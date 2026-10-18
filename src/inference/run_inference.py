"""CLI entry point for the Phase 9 single-video inference pipeline."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OCSORT_ROOT = _REPO_ROOT / "third_party" / "OC_SORT"

# Ensure repo root is on sys.path so `import src.*` works when run directly
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check_dependencies() -> None:
    if not _OCSORT_ROOT.exists():
        print("[ERROR] OC-SORT not found. Clone it with:")
        print(f"  git clone https://github.com/noahcao/OC_SORT.git {_OCSORT_ROOT}")
        sys.exit(1)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _frames_dir_to_video(frames_dir: Path, fps: float = 25.0) -> Path:
    """Write a temporary .mp4 from a sorted JPEG frame directory and return its path."""
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No .jpg/.png frames found in {frames_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise IOError(f"Cannot read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    tmp = Path(tempfile.mktemp(suffix=".mp4", prefix=f"{frames_dir.name}_"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"cv2.VideoWriter failed to open: {tmp}")

    for p in frame_paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"  [frames→video] wrote {len(frame_paths)} frames → {tmp}")
    return tmp


def _merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Overlay explicit CLI flags on top of the YAML config."""
    if getattr(args, "video", None):
        config.setdefault("input", {})["video_path"] = args.video
    if getattr(args, "frames_dir", None):
        config.setdefault("input", {})["frames_dir"] = args.frames_dir
    if args.output_dir:
        config.setdefault("output", {})["output_root"] = args.output_dir
    if args.conf_thresh is not None:
        config.setdefault("detection", {})["confidence_threshold"] = args.conf_thresh
    if args.cleanup:
        config.setdefault("output", {})["cleanup"] = True
    return config


def _make_printer():
    """Return a progress callback that prints to stdout."""
    _last_was_inline = False

    def callback(event: dict) -> None:
        nonlocal _last_was_inline
        stage = event["stage"]
        total = event["total_stages"]
        name = event["stage_name"]
        frame = event["frame"]
        total_frames = event["total_frames"]
        status = event["status"]

        if status == "running":
            print(
                f"\rStage {stage}/{total}: {name} — frame {frame}/{total_frames}",
                end="",
                flush=True,
            )
            _last_was_inline = True
        elif status == "done":
            if _last_was_inline:
                print()  # end the \r line
                _last_was_inline = False
            print(f"Stage {stage}/{total}: {name} — done ({total_frames} frames)")
        elif status == "error":
            if _last_was_inline:
                print()
                _last_was_inline = False
            print(f"Stage {stage}/{total}: {name} — ERROR", file=sys.stderr)

    return callback


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cattle Vision Framework — single-video inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", default=None, help="Path to input video file")
    src.add_argument(
        "--frames_dir",
        default=None,
        metavar="DIR",
        help="Path to a directory of sorted JPEG frames (used instead of --video)",
    )
    parser.add_argument("--frames_fps", type=float, default=25.0,
                        help="FPS to assign when assembling video from --frames_dir")
    parser.add_argument("--output_dir", default=None, help="Root directory for outputs")
    parser.add_argument(
        "--config",
        default="configs/inference/default.yaml",
        help="Path to inference YAML config",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate files after the run",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    _check_dependencies()

    args = parse_args(argv)

    config = _load_config(args.config)
    config = _merge_cli_overrides(config, args)

    # ── Resolve video path (converting frame directory if needed) ────────────
    tmp_video: Optional[Path] = None
    if args.frames_dir is not None:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.is_dir():
            print(f"[ERROR] frames_dir not found: {frames_dir}", file=sys.stderr)
            sys.exit(1)
        tmp_video = _frames_dir_to_video(frames_dir, fps=args.frames_fps)
        config.setdefault("input", {})["video_path"] = str(tmp_video)
        # Use the clip directory name as job_id so the output dir is named correctly
        config["input"]["job_id"] = frames_dir.name

    video_path = config["input"]["video_path"]
    if not Path(video_path).exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    from src.inference.pipeline import run_pipeline

    print(f"Cattle Vision — inference pipeline")
    print(f"  video : {video_path}")
    print(f"  output: {config['output']['output_root']}")
    print(f"  config: {args.config}")
    print()

    try:
        run_pipeline(config, progress=_make_printer())
    finally:
        if tmp_video is not None and tmp_video.exists():
            tmp_video.unlink()

    print("\nDone.")


if __name__ == "__main__":
    main()
