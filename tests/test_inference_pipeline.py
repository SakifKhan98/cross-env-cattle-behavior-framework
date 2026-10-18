"""Integration test for the Phase 9 end-to-end inference pipeline.

Runs the full pipeline on a synthetic 3-second video with:
  - Mocked RF-DETR-Seg (returns stable fake detections on every frame)
  - Mocked VideoMAE classify_tubelets (writes a valid predictions.csv
    without loading GPU weights)
  - Real OC-SORT tracking, real tubelet export, real analytics, real render

Skip if OC-SORT is not cloned.
"""

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

_OCSORT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "OC_SORT"
pytestmark = pytest.mark.skipif(
    not _OCSORT_ROOT.exists(),
    reason="OC-SORT not cloned — run: git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT",
)

# ── synthetic video fixture ───────────────────────────────────────────────────

_FPS = 25.0
_W, _H = 320, 240
_N_FRAMES = 75  # 3 seconds

# Stable bbox for the fake "cow" — chosen to be well inside the frame
_COW_BBOX = [40.0, 40.0, 160.0, 200.0]


def _make_video(path: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, _FPS, (_W, _H))
    rng = np.random.default_rng(0)
    x1, y1, x2, y2 = [int(c) for c in _COW_BBOX]
    for _ in range(_N_FRAMES):
        frame = rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        writer.write(frame)
    writer.release()


# ── mock helpers ─────────────────────────────────────────────────────────────

class _FakeDetections:
    """Mimics the return shape of model.predict() in rfdetr_seg_infer."""
    def __init__(self):
        self.xyxy = np.array([_COW_BBOX], dtype=np.float32)
        self.confidence = np.array([0.92], dtype=np.float32)
        self.mask = None


class _FakeRFDETR:
    def predict(self, image, threshold=0.3):
        return _FakeDetections()


def _fake_classify_tubelets(tubelet_rows, checkpoint, output_dir, job_id, **kwargs):
    """Writes a minimal valid predictions.csv without loading VideoMAE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.csv"
    logit_cols = [f"logit_{i}" for i in range(7)]
    with open(preds_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "video_id", "tubelet_dir",
            "start_frame", "end_frame", "label_id", "pred_label_id",
        ] + logit_cols)
        for row in tubelet_rows:
            logits = ["1.0"] + ["0.0"] * 6  # predict "Standing"
            w.writerow([
                "inference", job_id, row["tubelet_dir"],
                row["start_frame"], row["end_frame"], -1, 0,
            ] + logits)
    return preds_path


# ── main integration test ─────────────────────────────────────────────────────

def test_full_pipeline(tmp_path):
    video_path = str(tmp_path / "test_video.mp4")
    _make_video(video_path)

    config = {
        "input": {"video_path": video_path, "job_id": "test_job"},
        "models": {
            "rfdetr_seg_checkpoint": "fake_ckpt.pth",
            "videomae_checkpoint": "fake_videomae.pt",
        },
        "detection": {"confidence_threshold": 0.3},
        "tracking": {
            "det_thresh": 0.3,
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "delta_t": 3,
            "inertia": 0.2,
        },
        "output": {
            "output_root": str(tmp_path / "results"),
            "cleanup": False,
        },
    }

    fake_model = _FakeRFDETR()

    with patch("src.segmentation.rfdetr_seg_infer.load_model", return_value=fake_model), \
         patch("src.segmentation.rfdetr_seg_infer.predict_frame",
               side_effect=lambda model, frame, thresh: [
                   {"bbox": _COW_BBOX, "score": 0.92, "mask_rle": None, "mask_area": 0}
               ]), \
         patch("src.behavior.classify.classify_tubelets",
               side_effect=_fake_classify_tubelets):

        from src.inference.pipeline import run_pipeline

        events = []
        run_pipeline(config, progress=events.append)

    job_dir = Path(config["output"]["output_root"]) / "test_job"

    # ── assertions ────────────────────────────────────────────────────────────

    # Stage 3: tracks.json with at least one track
    tracks_path = job_dir / "tracks.json"
    assert tracks_path.exists(), "tracks.json not written"
    with open(tracks_path) as f:
        tracks = json.load(f)
    n_tracks = tracks["stats"]["total_unique_tracks"]
    assert n_tracks >= 1, f"Expected ≥1 track, got {n_tracks}"

    # Stage 5: predictions.csv with at least one row
    preds_path = job_dir / "predictions.csv"
    assert preds_path.exists(), "predictions.csv not written"
    with open(preds_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, "predictions.csv is empty"

    # Stage 6: analytics CSVs
    assert (job_dir / "activity_budget.csv").exists(), "activity_budget.csv missing"
    assert (job_dir / "behavior_deviation.csv").exists(), "behavior_deviation.csv missing"

    # Stage 7: annotated.mp4 is non-empty
    annotated = job_dir / "annotated.mp4"
    assert annotated.exists(), "annotated.mp4 not written"
    assert annotated.stat().st_size > 0, "annotated.mp4 is empty"

    # Progress events: stages 1–7 each emit at least one event
    stage_ids = {e["stage"] for e in events}
    assert stage_ids == {1, 2, 3, 4, 5, 6, 7}, f"Missing stage events: {stage_ids}"


def test_cleanup_flag(tmp_path):
    """--cleanup removes detections.json, tracks.json, tubelets/, predictions.csv
    but keeps annotated.mp4 and analytics CSVs."""
    video_path = str(tmp_path / "test_video.mp4")
    _make_video(video_path)

    config = {
        "input": {"video_path": video_path, "job_id": "cleanup_job"},
        "models": {
            "rfdetr_seg_checkpoint": "fake.pth",
            "videomae_checkpoint": "fake.pt",
        },
        "detection": {"confidence_threshold": 0.3},
        "tracking": {
            "det_thresh": 0.3, "max_age": 30, "min_hits": 3,
            "iou_threshold": 0.3, "delta_t": 3, "inertia": 0.2,
        },
        "output": {
            "output_root": str(tmp_path / "results"),
            "cleanup": True,
        },
    }

    fake_model = _FakeRFDETR()

    with patch("src.segmentation.rfdetr_seg_infer.load_model", return_value=fake_model), \
         patch("src.segmentation.rfdetr_seg_infer.predict_frame",
               side_effect=lambda model, frame, thresh: [
                   {"bbox": _COW_BBOX, "score": 0.92, "mask_rle": None, "mask_area": 0}
               ]), \
         patch("src.behavior.classify.classify_tubelets",
               side_effect=_fake_classify_tubelets):

        from src.inference.pipeline import run_pipeline
        run_pipeline(config)

    job_dir = Path(config["output"]["output_root"]) / "cleanup_job"

    # Cleaned up
    assert not (job_dir / "detections.json").exists(), "detections.json not cleaned"
    assert not (job_dir / "tracks.json").exists(), "tracks.json not cleaned"
    assert not (job_dir / "tubelets").exists(), "tubelets/ not cleaned"

    # Kept
    assert (job_dir / "annotated.mp4").exists(), "annotated.mp4 should be kept"
    assert (job_dir / "activity_budget.csv").exists(), "activity_budget.csv should be kept"
