"""Unit tests for src/data/convert_mmcows_detection.py."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pathlib import Path
import tempfile

from src.data.convert_mmcows_detection import convert, yolo_to_coco_bbox

IMG_W = 4480
IMG_H = 2800


# ── yolo_to_coco_bbox ────────────────────────────────────────────────────────

class TestYoloToCocoBoxConversion:
    def test_known_values(self):
        # cow_id=1, cx=0.7385..., cy=0.6298..., w=0.3056..., h=0.4318
        cx, cy, w, h = 0.7385044642857143, 0.6298214285714285, 0.30558035714285714, 0.4317857142857143
        x, y, bw, bh = yolo_to_coco_bbox(cx, cy, w, h, IMG_W, IMG_H)
        # Expected: center→absolute, then shift to top-left
        # px = cx*4480 = 3308.5, py = cy*2800 = 1763.5
        # pw = w*4480  = 1369.0, ph = h*2800  = 1209.0
        # x1 = 3308.5 - 684.5 = 2624.0, y1 = 1763.5 - 604.5 = 1159.0
        assert abs(x - 2624.0) < 1.0
        assert abs(y - 1159.0) < 1.0
        assert abs(bw - 1369.0) < 2.0
        assert abs(bh - 1209.0) < 2.0

    def test_clamps_to_image_boundary(self):
        # Box centred near top-left; should not go negative
        x, y, bw, bh = yolo_to_coco_bbox(0.0, 0.0, 0.5, 0.5, IMG_W, IMG_H)
        assert x == 0.0
        assert y == 0.0
        assert bw > 0
        assert bh > 0

    def test_small_box_survives(self):
        x, y, bw, bh = yolo_to_coco_bbox(0.5, 0.5, 0.01, 0.01, IMG_W, IMG_H)
        assert bw >= 2 and bh >= 2


# ── convert() ────────────────────────────────────────────────────────────────

def _make_label_dir(tmp: Path, date="0725", cam="cam_4") -> tuple[Path, Path]:
    label_dir = tmp / "labels" / "combined" / date / cam
    image_dir = tmp / "images" / date / cam
    label_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    return label_dir, image_dir


class TestConvertFunction:
    def test_all_cow_ids_remapped_to_category_1(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        # Write labels for cow IDs 1, 8, 16
        for ts, cow_id in [("100_01-00-00", 1), ("200_01-00-20", 8), ("300_01-00-40", 16)]:
            (label_dir / f"{ts}.txt").write_text(
                f"{cow_id} 0.5 0.5 0.1 0.1\n"
            )
            (image_dir / f"{ts}.jpg").write_bytes(b"")  # dummy

        coco = convert(tmp_path, None)

        assert all(ann["category_id"] == 1 for ann in coco["annotations"])

    def test_image_count_equals_label_file_count(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        for i in range(5):
            ts = f"10000{i}_01-00-0{i}"
            (label_dir / f"{ts}.txt").write_text(f"1 0.5 0.5 0.1 0.1\n")
            (image_dir / f"{ts}.jpg").write_bytes(b"")

        coco = convert(tmp_path, None)

        assert len(coco["images"]) == 5

    def test_empty_label_file_produces_image_entry_with_zero_annotations(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        ts = "1000_01-00-00"
        (label_dir / f"{ts}.txt").write_text("")  # no annotations
        (image_dir / f"{ts}.jpg").write_bytes(b"")

        coco = convert(tmp_path, None)

        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 0

    def test_bbox_pixel_values_are_correct(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        cx, cy, w, h = 0.5, 0.5, 0.2, 0.2
        ts = "1000_01-00-00"
        (label_dir / f"{ts}.txt").write_text(f"3 {cx} {cy} {w} {h}\n")
        (image_dir / f"{ts}.jpg").write_bytes(b"")

        coco = convert(tmp_path, None)

        bbox = coco["annotations"][0]["bbox"]
        # Expected: x=4480*(0.5-0.1)=1792, y=2800*(0.5-0.1)=1120, w=4480*0.2=896, h=2800*0.2=560
        assert abs(bbox[0] - 1792.0) < 1.0
        assert abs(bbox[1] - 1120.0) < 1.0
        assert abs(bbox[2] - 896.0) < 1.0
        assert abs(bbox[3] - 560.0) < 1.0

    def test_multiple_cows_per_frame(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        ts = "1000_01-00-00"
        lines = "\n".join([f"{cow} 0.5 0.5 0.1 0.1" for cow in range(1, 17)])
        (label_dir / f"{ts}.txt").write_text(lines)
        (image_dir / f"{ts}.jpg").write_bytes(b"")

        coco = convert(tmp_path, None)

        assert len(coco["annotations"]) == 16
        assert all(ann["category_id"] == 1 for ann in coco["annotations"])

    def test_output_json_written_to_disk(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        label_dir, image_dir = _make_label_dir(raw_dir)
        ts = "1000_01-00-00"
        (label_dir / f"{ts}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
        (image_dir / f"{ts}.jpg").write_bytes(b"")

        convert(raw_dir, out_dir)

        ann_file = out_dir / "_annotations.coco.json"
        assert ann_file.exists()
        data = json.loads(ann_file.read_text())
        assert "images" in data and "annotations" in data and "categories" in data

    def test_coco_categories_correct(self, tmp_path):
        label_dir, image_dir = _make_label_dir(tmp_path)
        ts = "1000_01-00-00"
        (label_dir / f"{ts}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
        (image_dir / f"{ts}.jpg").write_bytes(b"")

        coco = convert(tmp_path, None)

        assert len(coco["categories"]) == 1
        assert coco["categories"][0]["name"] == "cattle"
        assert coco["categories"][0]["id"] == 1
