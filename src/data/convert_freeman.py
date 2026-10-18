"""
src/data/convert_freeman.py

Converts Freeman Center Multi-Behavior (CMB 2024) YOLO annotations to
COCO detection format for RF-DETR evaluation.

Input structure:
    data/raw/freeman-cmb-2024/
    └── CMB_dataset/CMB_dataset/
        ├── train/
        │   ├── images/  {video_id}_frame_{N:06d}.PNG
        │   └── labels/  {video_id}_frame_{N:06d}.txt  (YOLO: class_id cx cy w h)
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/

Output structure:
    data/processed/detection/freeman/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.PNG  (symlinked from raw)
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.PNG
    └── test/
        ├── _annotations.coco.json
        └── *.PNG

Notes:
    - Detection is class-agnostic: all 9 Freeman behavior classes → category "cattle" (id 1).
    - Behavior labels are preserved in annotation["attributes"]["freeman_class_id"] for
      downstream behavior evaluation use (not consumed by the detector).
    - YOLO bbox (cx, cy, w, h normalized) → COCO bbox (x, y, w, h pixels, top-left origin).
    - All 39,363 frames have a matching image on disk (verified in analysis notebook).

Usage:
    python src/data/convert_freeman.py
    python src/data/convert_freeman.py --raw_dir data/raw/freeman-cmb-2024 \
                                        --out_dir data/processed/detection/freeman
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

FRAME_W = 1920
FRAME_H = 1080

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]

# Freeman input split name → output split name (matches CBVD-5/CVB convention)
SPLIT_MAP = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized (cx, cy, w, h) to COCO pixel (x, y, w, h) top-left."""
    pw = w * img_w
    ph = h * img_h
    px = (cx - w / 2) * img_w
    py = (cy - h / 2) * img_h
    # Clamp to image bounds
    px = max(0.0, px)
    py = max(0.0, py)
    pw = min(pw, img_w - px)
    ph = min(ph, img_h - py)
    return px, py, pw, ph


def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a YOLO .txt label file. Returns list of {class_id, cx, cy, w, h}."""
    rows = []
    text = label_path.read_text().strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append(
            {
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
        )
    return rows


# ── Core conversion ───────────────────────────────────────────────────────────


def convert_split(
    cmb_split_dir: Path,
    out_dir: Path,
    split_name: str,
) -> None:
    """Convert one Freeman split to COCO format.

    Args:
        cmb_split_dir: path to CMB_dataset/{train|val|test}/
        out_dir:        path to output split dir (e.g. data/processed/detection/freeman/train/)
        split_name:     human-readable label for progress output
    """
    images_dir = cmb_split_dir / "images"
    labels_dir = cmb_split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped_missing = 0
    skipped_degenerate = 0

    for label_path in tqdm(label_files, desc=f"  {split_name}"):
        img_path = images_dir / (label_path.stem + ".PNG")

        if not img_path.exists():
            skipped_missing += 1
            continue

        rows = parse_yolo_label(label_path)
        if not rows:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": FRAME_W,
                "height": FRAME_H,
            }
        )

        # Symlink image into output split dir
        dst = out_dir / img_path.name
        if not dst.exists():
            os.symlink(img_path.resolve(), dst)

        for row in rows:
            px, py, pw, ph = yolo_to_coco_bbox(
                row["cx"], row["cy"], row["w"], row["h"], FRAME_W, FRAME_H
            )
            if pw < 2 or ph < 2:
                skipped_degenerate += 1
                continue

            anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(px, 2), round(py, 2), round(pw, 2), round(ph, 2)],
                    "area": round(pw * ph, 2),
                    "iscrowd": 0,
                    "attributes": {"freeman_class_id": row["class_id"]},
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": f"Freeman Center (CMB 2024) detection dataset — {split_name} split",
            "version": "1.0",
        },
        "categories": COCO_CATEGORIES,
        "images": images,
        "annotations": anns,
    }

    ann_file = out_dir / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"  ✓ {split_name}: {len(images)} images, {len(anns)} annotations"
        + (f", {skipped_missing} skipped (missing image)" if skipped_missing else "")
        + (
            f", {skipped_degenerate} skipped (degenerate box)"
            if skipped_degenerate
            else ""
        )
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main(raw_dir: str, out_dir: str) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    cmb = raw / "CMB_dataset" / "CMB_dataset"

    if not cmb.exists():
        raise FileNotFoundError(f"CMB_dataset not found: {cmb}")

    print(f"\nConverting Freeman Center → COCO detection format")
    print(f"  Source : {cmb}")
    print(f"  Output : {out}\n")

    for src_split, dst_split in SPLIT_MAP.items():
        split_dir = cmb / src_split
        if not split_dir.exists():
            print(f"  ⚠ Skipping {src_split}: directory not found")
            continue
        convert_split(
            cmb_split_dir=split_dir,
            out_dir=out / dst_split,
            split_name=dst_split,
        )

    print(f"\n✓ Freeman Center conversion complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Freeman Center YOLO annotations to COCO detection format"
    )
    parser.add_argument(
        "--raw_dir",
        default="data/raw/freeman-cmb-2024",
        help="Path to raw Freeman Center dataset folder",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/detection/freeman",
        help="Output folder for COCO-format dataset",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
