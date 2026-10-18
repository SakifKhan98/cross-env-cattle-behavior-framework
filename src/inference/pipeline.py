"""Pipeline Orchestrator — Phase 9 inference pipeline."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

from src.inference.video_ingestor import VideoIngestor

# Progress event shape emitted through the callback:
# {
#   "stage": int,          # 1-based stage index
#   "stage_name": str,
#   "total_stages": int,   # always 7 for the full pipeline
#   "frame": int,          # frames processed so far in this stage
#   "total_frames": int,   # total frames in the video
#   "status": str,         # "running" | "done" | "error"
# }

TOTAL_STAGES = 7

ProgressCallback = Callable[[Dict[str, Any]], None]


def _noop_callback(event: Dict[str, Any]) -> None:
    pass


def _emit(progress: ProgressCallback, stage: int, name: str,
          frame: int, total_frames: int, status: str) -> None:
    progress({
        "stage": stage,
        "stage_name": name,
        "total_stages": TOTAL_STAGES,
        "frame": frame,
        "total_frames": total_frames,
        "status": status,
    })


def run_pipeline(config: Dict[str, Any], progress: ProgressCallback = _noop_callback) -> None:
    """Run the full inference pipeline on a single video.

    Args:
        config:   Parsed contents of configs/inference/default.yaml (or equivalent dict).
        progress: Callback invoked with a progress-event dict at stage transitions
                  and periodically during long-running stages.
    """
    video_path = config["input"]["video_path"]
    job_id = config["input"].get("job_id") or Path(video_path).stem
    output_root = Path(config["output"]["output_root"]) / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()
    stage_times: Dict[str, float] = {}
    video_file = Path(video_path)
    file_size_mb = video_file.stat().st_size / (1024 * 1024)

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    t0 = time.time()
    ingestor = VideoIngestor(video_path)
    total_frames = ingestor.frame_count

    _emit(progress, 1, "Ingest", 0, total_frames, "running")
    _emit(progress, 1, "Ingest", total_frames, total_frames, "done")
    stage_times["ingest_s"] = time.time() - t0

    # ── Stage 2: Detect + Segment ────────────────────────────────────────────
    t0 = time.time()
    conf_thresh = config["detection"]["confidence_threshold"]
    ckpt = config["models"]["rfdetr_seg_checkpoint"]

    from src.segmentation.rfdetr_seg_infer import load_model, predict_frame

    model = load_model(ckpt)

    detections_out = {
        "video_path": str(video_path),
        "fps": ingestor.fps,
        "width": ingestor.width,
        "height": ingestor.height,
        "total_frames": total_frames,
        "frames": {},
    }

    _emit(progress, 2, "Detect", 0, total_frames, "running")

    for frame_idx, frame_bgr in ingestor.frames():
        dets = predict_frame(model, frame_bgr, conf_thresh)
        detections_out["frames"][str(frame_idx)] = dets
        _emit(progress, 2, "Detect", frame_idx + 1, total_frames, "running")

    det_path = output_root / "detections.json"
    with open(det_path, "w") as f:
        json.dump(detections_out, f)

    _emit(progress, 2, "Detect", total_frames, total_frames, "done")
    stage_times["detect_s"] = time.time() - t0

    # ── Stage 3: Track ───────────────────────────────────────────────────────
    t0 = time.time()
    from src.tracking.track import run_tracking

    ocsort_cfg = config.get("tracking", {})

    _emit(progress, 3, "Track", 0, total_frames, "running")

    def _on_frame(frame_idx: int, total: int) -> None:
        _emit(progress, 3, "Track", frame_idx, total, "running")

    tracks = run_tracking(
        detections_out["frames"],
        img_h=ingestor.height,
        img_w=ingestor.width,
        ocsort_cfg=ocsort_cfg,
        on_frame=_on_frame,
    )
    tracks["video_id"] = job_id

    tracks_path = output_root / "tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(tracks, f)

    _emit(progress, 3, "Track", total_frames, total_frames, "done")
    stage_times["track_s"] = time.time() - t0

    # ── Stage 4: Extract Tubelets ────────────────────────────────────────────
    t0 = time.time()
    from src.data.export_tubelets import export_tubelets_from_tracks

    tubelets_dir = output_root / "tubelets"
    _emit(progress, 4, "Extract", 0, total_frames, "running")

    tubelet_rows = export_tubelets_from_tracks(
        tracks, str(video_path), str(tubelets_dir),
    )

    _emit(progress, 4, "Extract", total_frames, total_frames, "done")
    stage_times["extract_s"] = time.time() - t0

    # ── Stage 5: Classify ────────────────────────────────────────────────────
    t0 = time.time()
    from src.behavior.classify import classify_tubelets

    videomae_ckpt = config["models"]["videomae_checkpoint"]
    n_tubelets = len(tubelet_rows)
    _emit(progress, 5, "Classify", 0, n_tubelets, "running")

    def _cls_progress(n_done: int, n_total: int) -> None:
        _emit(progress, 5, "Classify", n_done, n_total, "running")

    preds_csv = classify_tubelets(
        tubelet_rows,
        checkpoint=videomae_ckpt,
        output_dir=output_root,
        job_id=job_id,
        progress=_cls_progress,
    )

    _emit(progress, 5, "Classify", n_tubelets, n_tubelets, "done")
    stage_times["classify_s"] = time.time() - t0

    # ── Stage 6: Analyze ─────────────────────────────────────────────────────
    t0 = time.time()
    from src.analytics.timeline import run_timeline_analysis
    from src.analytics.budget import run_budget_analysis

    timelines_dir = output_root / "timelines"
    _emit(progress, 6, "Analyze", 0, total_frames, "running")

    run_timeline_analysis(preds_csv, timelines_dir, fps=ingestor.fps)
    run_budget_analysis(timelines_dir, output_root)

    _emit(progress, 6, "Analyze", total_frames, total_frames, "done")
    stage_times["analyze_s"] = time.time() - t0

    # ── Stage 7: Render ──────────────────────────────────────────────────────
    t0 = time.time()
    from src.tracking.render_behavior_video import render_inference_video

    annotated_path = output_root / "annotated.mp4"
    _emit(progress, 7, "Render", 0, total_frames, "running")

    render_inference_video(
        video_path=video_path,
        tracks_json=tracks,
        predictions_csv=preds_csv,
        output_path=annotated_path,
        fps=ingestor.fps,
        job_id=job_id,
    )

    _emit(progress, 7, "Render", total_frames, total_frames, "done")
    stage_times["render_s"] = time.time() - t0

    # ── Timing log ───────────────────────────────────────────────────────────
    total_s = time.time() - t_pipeline_start
    duration_s = total_frames / ingestor.fps if ingestor.fps else 0.0

    timing_row = {
        "video": video_file.name,
        "file_size_mb": round(file_size_mb, 2),
        "frame_count": total_frames,
        "fps": round(ingestor.fps, 3),
        "duration_s": round(duration_s, 1),
        **{k: round(v, 2) for k, v in stage_times.items()},
        "total_s": round(total_s, 2),
    }
    _TIMING_FIELDS = [
        "video", "file_size_mb", "frame_count", "fps", "duration_s",
        "ingest_s", "detect_s", "track_s", "extract_s", "classify_s",
        "analyze_s", "render_s", "total_s",
    ]

    # Per-video timing file
    per_video_timing = output_root / "timing.csv"
    with open(per_video_timing, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDS)
        writer.writeheader()
        writer.writerow(timing_row)

    # Shared timing log in the parent output dir (appends across videos)
    shared_timing = output_root.parent / "timing.csv"
    write_header = not shared_timing.exists()
    with open(shared_timing, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(timing_row)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if config["output"].get("cleanup"):
        import shutil as _shutil
        for p in [det_path, tracks_path, preds_csv]:
            try:
                p.unlink()
            except OSError:
                pass
        _shutil.rmtree(tubelets_dir, ignore_errors=True)
