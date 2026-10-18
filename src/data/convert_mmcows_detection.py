"""
src/data/convert_mmcows_detection.py

Converts mmCows cam_4 YOLO-format bounding box annotations to COCO detection
format for RF-DETR OOD evaluation.

Input structure:
    data/raw/mmCows/labels/combined/0725/cam_4/  {timestamp}.txt  (YOLO: cow_id cx cy w h)
    data/raw/mmCows/images/0725/cam_4/           {timestamp}.jpg

YOLO format note: the first column is the per-cow integer ID (1–16), NOT a class
label. All cow IDs are remapped to category_id=1 ("cattle") in the COCO output.
Image dimensions are 4480×2800 pixels.

Output:
    data/processed/detection/mmcows/
    ├── _annotations.coco.json
    └── *.jpg  (symlinked from raw)

Usage:
    python src/data/convert_mmcows_detection.py
    python src/data/convert_mmcows_detection.py \
        --raw_dir  data/raw/mmCows \
        --out_dir  data/processed/detection/mmcows \
        --date     0725 --cam cam_4
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

IMG_W = 4480
IMG_H = 2800

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]


def yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert normalised YOLO cx/cy/w/h to absolute COCO [x, y, w, h]."""
    px = cx * img_w
    py = cy * img_h
    pw = w * img_w
    ph = h * img_h
    x1 = max(0.0, px - pw / 2)
    y1 = max(0.0, py - ph / 2)
    # Clamp to image boundary
    pw = min(pw, img_w - x1)
    ph = min(ph, img_h - y1)
    return round(x1, 2), round(y1, 2), round(pw, 2), round(ph, 2)


def convert(raw_dir: Path, out_dir: Path, date: str = "0725", cam: str = "cam_4") -> dict:
    """
    Convert mmCows YOLO labels to COCO JSON.

    Returns the COCO dict (useful for testing without file I/O side effects when
    out_dir is None).
    """
    label_dir = raw_dir / "labels" / "combined" / date / cam
    image_dir = raw_dir / "images" / date / cam

    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No .txt label files in {label_dir}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped_degenerate = 0
    images_no_ann = 0

    for label_path in tqdm(label_files, desc="mmcows-detection"):
        img_name = label_path.stem + ".jpg"
        img_path = image_dir / img_name

        images.append({
            "id":        image_id,
            "file_name": img_name,
            "width":     IMG_W,
            "height":    IMG_H,
        })

        if out_dir is not None:
            dst = out_dir / img_name
            if not dst.exists() and img_path.exists():
                os.symlink(img_path.resolve(), dst)

        lines = label_path.read_text().strip().splitlines()
        if not lines:
            images_no_ann += 1
            image_id += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            # parts[0] is cow_id (integer 1–16) — remapped to category_id=1
            cx, cy, w, h = map(float, parts[1:])
            x, y, bw, bh = yolo_to_coco_bbox(cx, cy, w, h, IMG_W, IMG_H)
            if bw < 2 or bh < 2:
                skipped_degenerate += 1
                continue
            anns.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [x, y, bw, bh],
                "area":        round(bw * bh, 2),
                "iscrowd":     0,
            })
            ann_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": f"mmCows {date} {cam} — COCO format for OOD detection eval",
            "version": "1.0",
        },
        "categories": COCO_CATEGORIES,
        "images":     images,
        "annotations": anns,
    }

    if out_dir is not None:
        ann_file = out_dir / "_annotations.coco.json"
        with open(ann_file, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"\nmmCows ({date}/{cam}) → COCO conversion complete.")
        print(f"  Images           : {len(images)}")
        print(f"  Annotations      : {len(anns)}")
        print(f"  Images w/o anns  : {images_no_ann}")
        if skipped_degenerate:
            print(f"  Skipped (degenerate box): {skipped_degenerate}")
        print(f"  Output           : {out_dir}")

    return coco


def main(raw_dir: str, out_dir: str, date: str = "0725", cam: str = "cam_4") -> None:
    convert(Path(raw_dir), Path(out_dir), date=date, cam=cam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert mmCows YOLO annotations to COCO detection format"
    )
    parser.add_argument("--raw_dir", default="data/raw/mmCows",
                        help="Path to mmCows raw dataset folder")
    parser.add_argument("--out_dir", default="data/processed/detection/mmcows",
                        help="Output folder for COCO-format dataset")
    parser.add_argument("--date", default="0725",
                        help="Date subfolder (default: 0725)")
    parser.add_argument("--cam", default="cam_4",
                        help="Camera subfolder (default: cam_4)")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, date=args.date, cam=args.cam)
