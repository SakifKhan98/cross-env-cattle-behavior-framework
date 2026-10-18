"""
src/data/convert_opencows2020.py

Converts OpenCows2020 detection_and_localisation annotations (Supervisely/DatasetNinja JSON)
to COCO detection format for RF-DETR OOD evaluation.

Input structure:
    data/raw/opencow2020/detection_and_localisation/
    ├── ann/  {image_name}.jpg.json   (Supervisely JSON, class "cow")
    └── img/  {image_name}.jpg

Output:
    data/processed/detection/opencows2020/
    ├── _annotations.coco.json
    └── *.jpg  (symlinked from raw)

Notes:
    - All 7,043 detection images treated as a single test split (OOD eval, no training done here).
    - Annotation geometry: rectangle exterior [[x1,y1],[x2,y2]] (top-left, bottom-right).
    - COCO bbox format: [x, y, width, height] in absolute pixels.

Usage:
    python src/data/convert_opencows2020.py
    python src/data/convert_opencows2020.py --raw_dir data/raw/opencow2020 \
                                             --out_dir data/processed/detection/opencows2020
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]


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


def convert(raw_dir: Path, out_dir: Path) -> None:
    ann_dir = raw_dir / "detection_and_localisation" / "ann"
    img_dir = raw_dir / "detection_and_localisation" / "img"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.json"))

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped_missing = 0
    skipped_degenerate = 0

    for ann_path in tqdm(ann_files, desc="opencows2020"):
        # Annotation filename: {img_name}.jpg.json
        img_name = ann_path.stem  # strips .json → e.g. "000001.jpg"
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

        dst = out_dir / img_name
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
            "description": "OpenCows2020 detection_and_localisation — COCO format for OOD eval",
            "version": "1.0",
        },
        "categories": COCO_CATEGORIES,
        "images":     images,
        "annotations": anns,
    }

    ann_file = out_dir / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nOpenCows2020 → COCO conversion complete.")
    print(f"  Images     : {len(images)}")
    print(f"  Annotations: {len(anns)}")
    if skipped_missing:
        print(f"  Skipped (missing image): {skipped_missing}")
    if skipped_degenerate:
        print(f"  Skipped (degenerate box): {skipped_degenerate}")
    print(f"  Output: {out_dir}")


def main(raw_dir: str, out_dir: str) -> None:
    convert(Path(raw_dir), Path(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OpenCows2020 Supervisely annotations to COCO detection format"
    )
    parser.add_argument("--raw_dir", default="data/raw/opencow2020",
                        help="Path to opencow2020 raw dataset folder")
    parser.add_argument("--out_dir", default="data/processed/detection/opencows2020",
                        help="Output folder for COCO-format dataset")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
