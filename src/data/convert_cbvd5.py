"""
src/data/convert_cbvd5.py

Converts CBVD-5 AVA-format CSV annotations to COCO detection format
for RF-DETR training.

Input structure:
    data/raw/cbvd5/
    ├── annotations/
    │   ├── ava_train_v2.1.csv
    │   ├── ava_val_v2.1.csv
    │   └── ava_test_v2.1.csv
    └── labelframes/labelframes/
        └── {video_id}_{timestamp:05d}.jpg

Output structure (RF-DETR COCO format):
    data/processed/detection/cbvd5/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg   (symlinked from labelframes)
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg

Usage:
    python src/data/convert_cbvd5.py
    python src/data/convert_cbvd5.py --raw_dir data/raw/cbvd5 --out_dir data/processed/detection/cbvd5
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

# CBVD-5 AVA CSV has no header. Columns in order:
CSV_COLS = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "target_id"]

# Frame resolution (Dahua surveillance cameras)
FRAME_W = 1920
FRAME_H = 1080

# COCO detection uses a single class for cattle detection
COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]

# Official splits → CSV filenames
SPLIT_FILES = {
    "train": "ava_train_v2.1.csv",
    "valid": "ava_val_v2.1.csv",
    "test": "ava_test_v2.1.csv",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CBVD-5 AVA CSV file (no header)."""
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS)
    return df


def normalized_to_coco_bbox(x1n, y1n, x2n, y2n, width, height):
    """
    Convert normalized [x1,y1,x2,y2] to COCO [x,y,w,h] in pixels.
    Clamps to frame boundaries.
    """
    x1 = max(0.0, x1n * width)
    y1 = max(0.0, y1n * height)
    x2 = min(float(width), x2n * width)
    y2 = min(float(height), y2n * height)
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def get_frame_size(image_path: Path):
    """Read actual frame dimensions from image file."""
    with Image.open(image_path) as img:
        return img.width, img.height


def build_image_filename(video_id, timestamp):
    """
    CBVD-5 image naming: {video_id}_{timestamp:05d}.jpg
    e.g. video_id=618, timestamp=2 → '618_00002.jpg'
    """
    return f"{video_id}_{int(timestamp):05d}.jpg"


# ── Core conversion ──────────────────────────────────────────────────────────


def convert_split(
    df: pd.DataFrame,
    labelframes_dir: Path,
    out_dir: Path,
    split_name: str,
):
    """
    Convert one split (train/valid/test) to COCO format.

    - Deduplicates boxes: same (video_id, timestamp, box) with multiple
      action_ids → kept as one annotation (detection is class-agnostic)
    - Skips rows where the source image does not exist on disk
    - Images are SYMLINKED (not copied) to save disk space
    - Images are placed FLAT in out_dir (RF-DETR requirement)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: (video_id, timestamp, x1, y1, x2, y2) → keep one row
    # For detection we only need unique boxes, not individual action_ids
    box_key_cols = ["video_id", "timestamp", "x1", "y1", "x2", "y2"]
    df_unique = df.drop_duplicates(subset=box_key_cols).reset_index(drop=True)

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped = 0

    # Build image id map: filename → image_id (avoid duplicate image entries)
    filename_to_image_id = {}

    # Pre-build the image dict once (not inside the loop) for performance
    for _, row in tqdm(
        df_unique.iterrows(), total=len(df_unique), desc=f"  {split_name}"
    ):

        video_id = int(row["video_id"])
        timestamp = int(row["timestamp"])
        filename = build_image_filename(video_id, timestamp)
        src_path = labelframes_dir / filename

        # Skip if image doesn't exist on disk
        if not src_path.exists():
            skipped += 1
            continue

        # Register image if not already seen
        if filename not in filename_to_image_id:
            # Use actual image size (handles any non-standard frames)
            try:
                w, h = get_frame_size(src_path)
            except Exception:
                w, h = FRAME_W, FRAME_H

            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": w,
                    "height": h,
                }
            )
            filename_to_image_id[filename] = image_id
            image_id += 1

            # Symlink image into output split folder
            dst_path = out_dir / filename
            if not dst_path.exists():
                os.symlink(src_path.resolve(), dst_path)

        # Create annotation
        img_id = filename_to_image_id[filename]
        w = images[img_id - 1]["width"]  # fast lookup (image_id is 1-indexed)
        h = images[img_id - 1]["height"]

        # Find the correct image dimensions from the images list
        img_entry = next(im for im in images if im["id"] == img_id)
        frame_w, frame_h = img_entry["width"], img_entry["height"]

        bx, by, bw, bh = normalized_to_coco_bbox(
            row["x1"], row["y1"], row["x2"], row["y2"], frame_w, frame_h
        )

        # Skip degenerate boxes
        if bw < 2 or bh < 2:
            continue

        anns.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,  # single class: cattle
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco = {
        "info": {
            "description": f"CBVD-5 detection dataset — {split_name} split",
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
        f"  ✓ {split_name}: {len(images)} images, "
        f"{len(anns)} annotations, {skipped} rows skipped (missing images)"
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main(raw_dir: str, out_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir)

    ann_dir = raw / "annotations"
    labelframes_dir = raw / "labelframes" / "labelframes"

    if not labelframes_dir.exists():
        raise FileNotFoundError(f"Labelframes not found: {labelframes_dir}")

    print(f"\nConverting CBVD-5 → COCO detection format")
    print(f"  Source : {raw}")
    print(f"  Output : {out}\n")

    for split_name, csv_file in SPLIT_FILES.items():
        csv_path = ann_dir / csv_file
        if not csv_path.exists():
            print(f"  ⚠ Skipping {split_name}: {csv_file} not found")
            continue

        print(f"Loading {csv_file} ...")
        df = load_csv(csv_path)
        print(f"  {len(df)} rows loaded")

        convert_split(
            df=df,
            labelframes_dir=labelframes_dir,
            out_dir=out / split_name,
            split_name=split_name,
        )

    print("\n✓ CBVD-5 conversion complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CBVD-5 AVA annotations to COCO detection format"
    )
    parser.add_argument(
        "--raw_dir", default="data/raw/cbvd5", help="Path to raw CBVD-5 dataset folder"
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/detection/cbvd5",
        help="Output folder for COCO-format dataset",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
