"""
src/data/convert_cattleeyeview.py

Converts CattleEyeView YOLO-format annotations to COCO detection + segmentation
format for RF-DETR OOD evaluation and Mask IoU evaluation.

Input structure (test split only — OOD eval uses the held-out test set):
    data/raw/cattle-eye-view/dataset/
    ├── detect/
    │   ├── images/test/{video_id}/{frame_id}.jpg
    │   └── labels/test/{video_id}/{frame_id}.txt   # class cx cy w h (normalized)
    └── segment/
        └── labels/test/{video_id}/{frame_id}.txt   # class x1 y1 x2 y2 ... xn yn (normalized)

Output:
    data/processed/detection/cattleeyeview/
    └── test/
        ├── _annotations.coco.json          # bbox + segmentation polygon annotations
        └── {video_id}__{frame_id}.jpg      # symlinked images (flat dir)

Notes:
    - Images are 1920×1080 px.
    - Detect labels are the primary source; segment polygons are merged in when available.
    - Single category: "cattle" (id 1).
    - Tracking evaluation (IDF1) is NOT possible: no ground-truth track IDs in annotations.

Usage:
    python src/data/convert_cattleeyeview.py
    python src/data/convert_cattleeyeview.py \
        --raw_dir  data/raw/cattle-eye-view \
        --out_dir  data/processed/detection/cattleeyeview
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

IMAGE_W = 1920
IMAGE_H = 1080

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]


def yolo_bbox_to_coco(cx: float, cy: float, w: float, h: float,
                      img_w: int, img_h: int) -> list[float]:
    """Convert normalised YOLO bbox → COCO [x, y, w, h] in pixels."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    bw = min(bw, img_w - x1)
    bh = min(bh, img_h - y1)
    return [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)]


def yolo_poly_to_coco(coords: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert flat normalised YOLO polygon coords → flat pixel COCO segmentation."""
    out = []
    for i in range(0, len(coords), 2):
        out.append(round(coords[i] * img_w, 2))
        out.append(round(coords[i + 1] * img_h, 2))
    return out


def poly_to_bbox(seg: list[float]) -> list[float]:
    """Compute COCO bbox [x, y, w, h] from a flat pixel polygon."""
    xs = seg[0::2]
    ys = seg[1::2]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def poly_area(seg: list[float]) -> float:
    """Shoelace area of a polygon given as flat [x1,y1,x2,y2,...]."""
    xs = seg[0::2]
    ys = seg[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def convert_split(raw_dir: Path, out_dir: Path) -> None:
    detect_img_dir = raw_dir / "dataset" / "detect" / "images" / "test"
    detect_lbl_dir = raw_dir / "dataset" / "detect" / "labels" / "test"
    seg_lbl_dir    = raw_dir / "dataset" / "segment" / "labels" / "test"

    for d in [detect_img_dir, detect_lbl_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    out_dir.mkdir(parents=True, exist_ok=True)

    images_coco = []
    annotations  = []
    img_id  = 0
    ann_id  = 0

    video_ids = sorted(os.listdir(detect_img_dir))

    for vid in tqdm(video_ids, desc="videos"):
        vid_img_dir = detect_img_dir / vid
        vid_lbl_dir = detect_lbl_dir / vid
        vid_seg_dir = seg_lbl_dir / vid if (seg_lbl_dir / vid).exists() else None

        if not vid_img_dir.is_dir() or not vid_lbl_dir.is_dir():
            continue

        for img_file in sorted(vid_img_dir.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            stem = img_file.stem
            lbl_file = vid_lbl_dir / f"{stem}.txt"

            if not lbl_file.exists():
                continue

            img_id += 1
            flat_name = f"{vid}__{img_file.name}"

            # Symlink image into output dir
            symlink_path = out_dir / flat_name
            if not symlink_path.exists():
                symlink_path.symlink_to(img_file.resolve())

            images_coco.append({
                "id":        img_id,
                "file_name": flat_name,
                "width":     IMAGE_W,
                "height":    IMAGE_H,
            })

            # Load detect labels (primary)
            detect_lines = lbl_file.read_text().strip().splitlines()

            # Load segment labels if available (same number of lines, same order)
            seg_lines = []
            if vid_seg_dir is not None:
                seg_file = vid_seg_dir / f"{stem}.txt"
                if seg_file.exists():
                    seg_lines = seg_file.read_text().strip().splitlines()

            for inst_idx, det_line in enumerate(detect_lines):
                parts = det_line.split()
                if len(parts) < 5:
                    continue

                cx, cy, w, h = map(float, parts[1:5])
                coco_bbox = yolo_bbox_to_coco(cx, cy, w, h, IMAGE_W, IMAGE_H)
                bw, bh = coco_bbox[2], coco_bbox[3]
                area_bbox = round(bw * bh, 2)

                # Segmentation polygon (from seg labels, if present and matching)
                segmentation = []
                area = area_bbox
                if inst_idx < len(seg_lines):
                    seg_parts = seg_lines[inst_idx].split()
                    if len(seg_parts) >= 7:  # class + at least 3 points
                        coords = list(map(float, seg_parts[1:]))
                        if len(coords) % 2 == 0:
                            seg_poly = yolo_poly_to_coco(coords, IMAGE_W, IMAGE_H)
                            segmentation = [seg_poly]
                            # Use polygon-derived bbox and area when mask is available
                            coco_bbox = poly_to_bbox(seg_poly)
                            area = round(poly_area(seg_poly), 2)

                ann_id += 1
                annotations.append({
                    "id":            ann_id,
                    "image_id":      img_id,
                    "category_id":   1,
                    "bbox":          coco_bbox,
                    "area":          area,
                    "segmentation":  segmentation,
                    "iscrowd":       0,
                })

    coco = {
        "info":        {"description": "CattleEyeView test split — detection + segmentation"},
        "categories":  COCO_CATEGORIES,
        "images":      images_coco,
        "annotations": annotations,
    }

    ann_path = out_dir / "_annotations.coco.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"\n[convert] CattleEyeView test split → {out_dir}")
    print(f"  Images      : {len(images_coco)}")
    print(f"  Annotations : {len(annotations)}")
    print(f"  COCO file   : {ann_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Convert CattleEyeView to COCO format")
    p.add_argument("--raw_dir",
                   default="data/raw/cattle-eye-view",
                   help="Root of the raw CattleEyeView download")
    p.add_argument("--out_dir",
                   default="data/processed/detection/cattleeyeview/test",
                   help="Output directory for COCO annotations + image symlinks")
    return p.parse_args()


def main():
    args = parse_args()
    convert_split(Path(args.raw_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
