"""Unit tests for src/data/convert_mmcows_tubelets.py."""

import csv
import os
import sys

import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.convert_mmcows_tubelets import (
    CLIP_LEN,
    LABEL_MAP,
    MAJORITY_THRESH,
    convert,
    load_behavior_csvs,
    load_label_files,
)

IMG_W = 4480
IMG_H = 2800


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_raw_dir(
    tmp: Path,
    date: str = "0725",
    cam: str = "cam_4",
    n_frames: int = 20,
    n_cows: int = 2,
    behavior_per_cow: dict[int, list[int]] | None = None,
    cow_ids_per_frame: dict[int, list[int]] | None = None,
) -> Path:
    """
    Create a minimal mmCows raw directory tree for testing.

    Args:
        n_frames:          number of image/label files
        n_cows:            number of cows (IDs 1..n_cows)
        behavior_per_cow:  {cow_id: [code_per_frame, ...]}, defaults to code=2 (Standing→0)
        cow_ids_per_frame: {frame_idx: [cow_ids present]}, defaults to all cows present
    """
    import cv2

    label_dir = tmp / "labels" / "combined" / date / cam
    image_dir = tmp / "images" / date / cam
    behavior_dir = tmp / "behavior_labels" / "individual"
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    behavior_dir.mkdir(parents=True)

    # Generate unix timestamps 15 seconds apart starting at a round number
    base_ts = 1690271846
    timestamps = [base_ts + i * 15 for i in range(n_frames)]

    # Default: all cows present in every frame
    if cow_ids_per_frame is None:
        cow_ids_per_frame = {i: list(range(1, n_cows + 1)) for i in range(n_frames)}

    # Default behavior: code 2 (Standing → canonical 0) for all cows/frames
    if behavior_per_cow is None:
        behavior_per_cow = {c: [2] * n_frames for c in range(1, n_cows + 1)}

    # Write label files (YOLO: cow_id cx cy w h)
    for frame_idx, ts in enumerate(timestamps):
        stem = f"{ts}_00-00-00"
        present_cows = cow_ids_per_frame.get(frame_idx, list(range(1, n_cows + 1)))
        lines = []
        for cow_id in present_cows:
            lines.append(f"{cow_id} 0.5 0.5 0.1 0.1")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines))

        # Write dummy image (3-channel, valid JPEG-readable)
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), img)

    # Write behavior CSVs
    for cow_id in range(1, n_cows + 1):
        codes = behavior_per_cow.get(cow_id, [2] * n_frames)
        csv_path = behavior_dir / f"C{cow_id:02d}_{date}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "datetime", "behavior"])
            for frame_idx, ts in enumerate(timestamps):
                code = codes[frame_idx] if frame_idx < len(codes) else 2
                writer.writerow([ts, "00:00:00", code])

    return tmp


# ── load_behavior_csvs ────────────────────────────────────────────────────────


class TestLoadBehaviorCsvs:
    def test_loads_all_cows(self, tmp_path):
        behavior_dir = tmp_path / "behavior_labels" / "individual"
        behavior_dir.mkdir(parents=True)
        for cow_id in range(1, 4):
            p = behavior_dir / f"C{cow_id:02d}_0725.csv"
            with open(p, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "datetime", "behavior"])
                writer.writerow([1690261200, "0:00:00", cow_id])
        result = load_behavior_csvs(behavior_dir, "0725")
        assert set(result.keys()) == {1, 2, 3}
        assert result[1][1690261200] == 1
        assert result[3][1690261200] == 3

    def test_missing_csv_skipped(self, tmp_path):
        behavior_dir = tmp_path / "behavior_labels" / "individual"
        behavior_dir.mkdir(parents=True)
        (behavior_dir / "C01_0725.csv").write_text(
            "timestamp,datetime,behavior\n1690261200,0:00:00,2\n"
        )
        result = load_behavior_csvs(behavior_dir, "0725")
        assert 1 in result
        assert 2 not in result


# ── load_label_files ──────────────────────────────────────────────────────────


class TestLoadLabelFiles:
    def test_timestamps_sorted(self, tmp_path):
        label_dir = tmp_path
        (label_dir / "1690271876_02-57-56.txt").write_text("1 0.5 0.5 0.1 0.1\n")
        (label_dir / "1690271846_02-57-26.txt").write_text("2 0.5 0.5 0.1 0.1\n")
        sorted_images, _ = load_label_files(label_dir)
        assert sorted_images[0][0] < sorted_images[1][0]

    def test_cow_bboxes_parsed(self, tmp_path):
        label_dir = tmp_path
        (label_dir / "1690271846_02-57-26.txt").write_text("3 0.7 0.6 0.3 0.4\n")
        _, frame_bboxes = load_label_files(label_dir)
        assert 3 in frame_bboxes[1690271846]
        cx, cy, w, h = frame_bboxes[1690271846][3]
        assert abs(cx - 0.7) < 1e-6

    def test_empty_label_file_yields_no_cows(self, tmp_path):
        label_dir = tmp_path
        (label_dir / "1690271846_02-57-26.txt").write_text("")
        _, frame_bboxes = load_label_files(label_dir)
        assert frame_bboxes[1690271846] == {}


# ── LABEL_MAP ─────────────────────────────────────────────────────────────────


class TestLabelMap:
    def test_codes_3_and_4_both_map_to_2(self):
        assert LABEL_MAP[3] == 2
        assert LABEL_MAP[4] == 2

    def test_code_2_maps_to_standing(self):
        assert LABEL_MAP[2] == 0

    def test_code_7_maps_to_lying(self):
        assert LABEL_MAP[7] == 1

    def test_code_5_maps_to_grooming(self):
        assert LABEL_MAP[5] == 5

    def test_dropped_codes_are_none(self):
        assert LABEL_MAP[0] is None
        assert LABEL_MAP[1] is None
        assert LABEL_MAP[6] is None


# ── convert() — exclusion criteria ───────────────────────────────────────────


class TestConvertExclusionCriteria:
    def test_window_with_code_0_is_excluded(self, tmp_path):
        # Frame 0 has behavior code 0 (unknown) for cow 1
        behavior = {1: [0] + [2] * (CLIP_LEN - 1)}
        _make_raw_dir(tmp_path, n_frames=CLIP_LEN, n_cows=1, behavior_per_cow=behavior)
        rows = convert(tmp_path, None)
        assert len(rows) == 0

    def test_window_with_missing_bbox_is_excluded(self, tmp_path):
        # Frame 0 is missing cow 1's bbox
        cow_ids_per_frame = {0: []}  # no cows in frame 0
        _make_raw_dir(
            tmp_path,
            n_frames=CLIP_LEN,
            n_cows=1,
            cow_ids_per_frame=cow_ids_per_frame,
        )
        rows = convert(tmp_path, None)
        assert len(rows) == 0

    def test_insufficient_majority_excluded(self, tmp_path):
        # 11 frames code 2 (Standing→0) + 5 frames code 7 (Lying→1) → majority=11 < 12
        behavior = {1: [2] * 11 + [7] * 5 + [2] * 4}
        _make_raw_dir(tmp_path, n_frames=20, n_cows=1, behavior_per_cow=behavior)
        # Only check windows starting at index 0..4; window 0 has 11×Standing + 5×Lying
        # The sliding window at idx=0: [2,2,2,2,2,2,2,2,2,2,2,7,7,7,7,7] → majority=11
        rows = convert(tmp_path, None)
        # No window starting at 0 should pass (11 < 12). Verify at least idx=0 excluded.
        window0_rows = [r for r in rows if r["start_frame"] == 0]
        assert len(window0_rows) == 0

    def test_dropped_class_code_1_produces_no_tubelets(self, tmp_path):
        # All frames have code 1 (Walking → None)
        behavior = {1: [1] * 20}
        _make_raw_dir(tmp_path, n_frames=20, n_cows=1, behavior_per_cow=behavior)
        rows = convert(tmp_path, None)
        assert len(rows) == 0

    def test_dropped_class_code_6_produces_no_tubelets(self, tmp_path):
        # All frames have code 6 (Drinking → None)
        behavior = {1: [6] * 20}
        _make_raw_dir(tmp_path, n_frames=20, n_cows=1, behavior_per_cow=behavior)
        rows = convert(tmp_path, None)
        assert len(rows) == 0

    def test_mixed_code_3_and_4_produce_foraging_tubelet(self, tmp_path):
        # 8 frames code 3 (head-up Foraging→2) + 8 frames code 4 (head-down Foraging→2)
        # Majority = 16 (all map to canonical 2). Should produce one tubelet with label_id=2.
        behavior = {1: [3] * 8 + [4] * 8}
        _make_raw_dir(tmp_path, n_frames=CLIP_LEN, n_cows=1, behavior_per_cow=behavior)
        rows = convert(tmp_path, None)
        assert len(rows) == 1
        assert rows[0]["label_id"] == 2

    def test_majority_exactly_at_threshold_is_included(self, tmp_path):
        # 12 frames code 2 (Standing→0) + 4 frames code 7 (Lying→1) → majority=12 = threshold
        behavior = {1: [2] * 12 + [7] * 4}
        _make_raw_dir(tmp_path, n_frames=CLIP_LEN, n_cows=1, behavior_per_cow=behavior)
        rows = convert(tmp_path, None)
        assert len(rows) == 1
        assert rows[0]["label_id"] == 0


# ── convert() — output schema ─────────────────────────────────────────────────


class TestConvertOutputSchema:
    def test_labels_csv_columns(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "tubelets"
        _make_raw_dir(raw_dir, n_frames=CLIP_LEN, n_cows=1)
        rows = convert(raw_dir, out_dir)
        assert len(rows) > 0
        required = {"dataset", "video_id", "tubelet_dir", "start_frame", "end_frame", "label_id"}
        assert required.issubset(set(rows[0].keys()))

    def test_dataset_and_video_id_fields(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "tubelets"
        _make_raw_dir(raw_dir, n_frames=CLIP_LEN, n_cows=1)
        rows = convert(raw_dir, out_dir)
        for row in rows:
            assert row["dataset"] == "mmcows"
            assert row["video_id"] == "cam4_0725"

    def test_labels_csv_written_to_disk(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "tubelets"
        _make_raw_dir(raw_dir, n_frames=CLIP_LEN, n_cows=1)
        convert(raw_dir, out_dir)
        assert (out_dir / "labels.csv").exists()

    def test_frame_files_written(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "tubelets"
        _make_raw_dir(raw_dir, n_frames=CLIP_LEN, n_cows=1)
        rows = convert(raw_dir, out_dir)
        assert len(rows) > 0
        tubelet_path = Path(rows[0]["tubelet_dir"])
        frame_files = list(tubelet_path.glob("frame_*.jpg"))
        assert len(frame_files) == CLIP_LEN

    def test_dry_run_returns_rows_without_writing(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _make_raw_dir(raw_dir, n_frames=CLIP_LEN, n_cows=1)
        rows = convert(raw_dir, None)
        assert len(rows) > 0
        # Nothing should be written
        assert not (raw_dir.parent / "tubelets").exists()

    def test_multiple_windows_with_stride_1(self, tmp_path):
        # With 20 frames and stride=1, a single-cow fully-valid sequence
        # yields 20 - CLIP_LEN + 1 = 5 tubelets
        raw_dir = tmp_path / "raw"
        _make_raw_dir(raw_dir, n_frames=20, n_cows=1)
        rows = convert(raw_dir, None)
        assert len(rows) == 20 - CLIP_LEN + 1
