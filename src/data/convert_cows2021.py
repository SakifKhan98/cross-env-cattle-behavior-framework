"""
src/data/convert_cows2021.py

Converts Cows2021 detection_and_localisation annotations (Supervisely/DatasetNinja JSON)
to COCO detection format for RF-DETR OOD evaluation.

Input structure:
    data/raw/cows2021/
    ├── detection_and_localisation-train/ann/  *.jpg.json
    ├── detection_and_localisation-train/img/  *.jpg
    ├── detection_and_localisation-val/ann/
    ├── detection_and_localisation-val/img/
    ├── detection_and_localisation-test/ann/
    └── detection_and_localisation-test/img/

Output:
    data/processed/detection/cows2021/
    ├── test/
    │   ├── _annotations.coco.json
    │   └── *.jpg  (symlinked from raw)
    └── (train and val also converted if --all_splits is set)

Notes:
    - Default: test split only (OOD eval uses the held-out test set).
    - Annotation geometry: Supervisely rectangle exterior [[x1,y1],[x2,y2]].
    - Class "cattle torso" → category "cattle" (id 1) for class-agnostic detection.
    - Tracking evaluation (IDF1) is NOT possible: detection annotations carry no cow IDs.

Usage:
    python src/data/convert_cows2021.py
    python src/data/convert_cows2021.py --raw_dir data/raw/cows2021 \
                                         --out_dir data/processed/detection/cows2021
    python src/data/convert_cows2021.py --all_splits
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]

# Raw split folder suffix → output split name
SPLIT_MAP = {
    "detection_and_localisation-train": "train",
    "detection_and_localisation-val":   "val",
    "detection_and_localisation-test":  "test",
}


def exterior_to_coco_bbox(exterior: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert Supervisely rectangle exterior [[x1,y1],[x2,y2]] to COCO [x,y,w,h]."""
    x1, y1 = exterior[0]
    x2, y2 = exterior[1]
    x = max(0.0, float(min(x1, x2)))
    y = max(0.0, float(min(y1, y2)))
    w = float(abs(x2 - x1))
    h = float(abs(y2 - y1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h


def convert_split(raw_split_dir: Path, out_split_dir: Path, split_name: str) -> None:
    ann_dir = raw_split_dir / "ann"
    img_dir = raw_split_dir / "img"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    out_split_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped_missing = 0
    skipped_degenerate = 0

    for ann_path in tqdm(ann_files, desc=f"  {split_name}"):
        img_name = ann_path.stem  # e.g. "08271.jpg"
        img_path = img_dir / img_name

        if not img_path.exists():
            skipped_missing += 1
            continue

        with open(ann_path) as f:
            data = json.load(f)

        img_w = data["size"]["width"]
        img_h = data["size"]["height"]

        objects = [o for o in data.get("objects", []) if o.get("geometryType") == "rectangle"]
        if not objects:
            continue

        images.append({
            "id":        image_id,
            "file_name": img_name,
            "width":     img_w,
            "height":    img_h,
        })

        dst = out_split_dir / img_name
        if not dst.exists():
            os.symlink(img_path.resolve(), dst)

        for obj in objects:
            exterior = obj["points"]["exterior"]
            x, y, w, h = exterior_to_coco_bbox(exterior, img_w, img_h)
            if w < 2 or h < 2:
                skipped_degenerate += 1
                continue
            anns.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area":        round(w * h, 2),
                "iscrowd":     0,
            })
            ann_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": f"Cows2021 detection — {split_name} split — COCO format",
            "version": "1.0",
        },
        "categories": COCO_CATEGORIES,
        "images":     images,
        "annotations": anns,
    }

    ann_file = out_split_dir / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"  ✓ {split_name}: {len(images)} images, {len(anns)} annotations"
        + (f", {skipped_missing} skipped (missing)" if skipped_missing else "")
        + (f", {skipped_degenerate} skipped (degenerate)" if skipped_degenerate else "")
    )


def main(raw_dir: str, out_dir: str, all_splits: bool) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)

    splits_to_convert = list(SPLIT_MAP.items()) if all_splits else [
        ("detection_and_localisation-test", "test")
    ]

    print(f"\nConverting Cows2021 → COCO detection format")
    print(f"  Source : {raw}")
    print(f"  Output : {out}\n")

    for raw_suffix, split_name in splits_to_convert:
        raw_split = raw / raw_suffix
        if not raw_split.exists():
            print(f"  ⚠ Skipping {raw_suffix}: directory not found")
            continue
        convert_split(
            raw_split_dir=raw_split,
            out_split_dir=out / split_name,
            split_name=split_name,
        )

    print(f"\n✓ Cows2021 conversion complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Cows2021 Supervisely annotations to COCO detection format"
    )
    parser.add_argument("--raw_dir", default="data/raw/cows2021",
                        help="Path to cows2021 raw dataset folder")
    parser.add_argument("--out_dir", default="data/processed/detection/cows2021",
                        help="Output folder for COCO-format dataset")
    parser.add_argument("--all_splits", action="store_true",
                        help="Convert train/val/test instead of test only")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.all_splits)
