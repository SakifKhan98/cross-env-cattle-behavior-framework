"""
src/data/convert_cvb.py

Converts CVB (Cattle Visual Behaviors) AVA-format CSV annotations to
COCO detection format for RF-DETR training.

Input structure:
    data/raw/cvb/
    ├── cvb_in_ava_format/
    │   ├── ava_train_set.csv
    │   ├── ava_val_set.csv
    │   ├── train_set_tasklist.csv
    │   └── val_set_tasklist.csv
    └── raw_frames/
        └── {clip_id}/
            └── img_{frame:05d}.jpg

Output structure (RF-DETR COCO format):
    data/processed/detection/cvb/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg   (symlinked from raw_frames)
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/        (same as valid for CVB — no separate test set)
        ├── _annotations.coco.json
        └── *.jpg

Notes:
    - CVB has no separate test set. val_set is used for both valid and test.
    - Dropped labels: none(1), walking(3), hidden(11), running(12)
    - Column H is the cow's track ID (used later for tracking evaluation)
    - Each row = one cow in one frame (not keyframe-only like CBVD-5)

Usage:
    python src/data/convert_cvb.py
    python src/data/convert_cvb.py --raw_dir data/raw/cvb --out_dir data/processed/detection/cvb
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

CSV_COLS = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "animal_id"]

# GoPro resolution
FRAME_W = 1920
FRAME_H = 1080

COCO_CATEGORIES = [{"id": 1, "name": "cattle", "supercategory": "animal"}]

# CVB action_id → canonical label (from behaviour_list.pbtx)
# DROP = exclude this label entirely
CVB_ACTION_MAP = {
    1: "DROP",  # none
    2: "Foraging",  # grazing
    3: "DROP",  # walking
    4: "Ruminating",  # ruminating-standing
    5: "Ruminating",  # ruminating-lying
    6: "Standing",  # resting-standing
    7: "Lying",  # resting-lying
    8: "Drinking",  # drinking
    9: "Grooming",  # grooming
    10: "Other",  # other
    11: "DROP",  # hidden
    12: "DROP",  # running
}

# Official split CSVs (used for detection only — lists which clip IDs belong to each split)
SPLIT_CONFIGS = {
    "train": {
        "csv": "ava_train_set.csv",
        "tasklist": "train_set_tasklist.csv",
    },
    "valid": {
        "csv": "ava_val_set.csv",
        "tasklist": "val_set_tasklist.csv",
    },
    "test": {
        "csv": "ava_val_set.csv",  # CVB has no separate test set
        "tasklist": "val_set_tasklist.csv",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CVB AVA CSV file (no header)."""
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS)
    return df


def normalized_to_coco_bbox(x1n, y1n, x2n, y2n, width, height):
    """Convert normalized [x1,y1,x2,y2] to COCO [x,y,w,h] in pixels."""
    x1 = max(0.0, x1n * width)
    y1 = max(0.0, y1n * height)
    x2 = min(float(width), x2n * width)
    y2 = min(float(height), y2n * height)
    return x1, y1, x2 - x1, y2 - y1


def get_frame_size(image_path: Path):
    """Read actual frame dimensions."""
    with Image.open(image_path) as img:
        return img.width, img.height


def build_image_filename(clip_id: str, timestamp: int) -> str:
    """
    CVB frame naming: img_{timestamp:05d}.jpg inside the clip folder.
    We flatten to: {clip_id}__img_{timestamp:05d}.jpg in the output folder.
    Double underscore separates clip_id from frame name.
    """
    return f"{clip_id}__img_{int(timestamp):05d}.jpg"


def get_frame_source_path(raw_frames_dir: Path, clip_id: str, timestamp: int) -> Path:
    """
    Source path: raw_frames/{clip_id}/img_{timestamp:05d}.jpg
    """
    return raw_frames_dir / clip_id / f"img_{int(timestamp):05d}.jpg"


# ── Core conversion ──────────────────────────────────────────────────────────


def convert_split(
    df: pd.DataFrame,
    raw_frames_dir: Path,
    out_dir: Path,
    split_name: str,
):
    """
    Convert one split to COCO format.

    Key differences vs CBVD-5:
    - Drop rows where action_id maps to DROP
    - animal_id (column H) is preserved as a note for tracking eval
    - One unique image = (clip_id, timestamp) combination
    - Flattened filename uses double underscore: clip_id__img_00001.jpg
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out dropped labels
    df = df[df["action_id"].map(CVB_ACTION_MAP) != "DROP"].copy()
    df = df.reset_index(drop=True)

    # Deduplicate boxes: same (clip, timestamp, box coords) → one annotation
    # (same cow can have multiple action_ids; for detection we need one box)
    box_key_cols = ["video_id", "timestamp", "x1", "y1", "x2", "y2"]
    df_unique = df.drop_duplicates(subset=box_key_cols).reset_index(drop=True)

    images = []
    anns = []
    image_id = 1
    ann_id = 1
    skipped = 0

    filename_to_image_id = {}

    for _, row in tqdm(
        df_unique.iterrows(), total=len(df_unique), desc=f"  {split_name}"
    ):

        clip_id = str(row["video_id"])
        timestamp = int(row["timestamp"])

        src_path = get_frame_source_path(raw_frames_dir, clip_id, timestamp)
        flat_name = build_image_filename(clip_id, timestamp)

        # Skip missing frames
        if not src_path.exists():
            skipped += 1
            continue

        # Register image if not already seen
        if flat_name not in filename_to_image_id:
            try:
                w, h = get_frame_size(src_path)
            except Exception:
                w, h = FRAME_W, FRAME_H

            images.append(
                {
                    "id": image_id,
                    "file_name": flat_name,
                    "width": w,
                    "height": h,
                }
            )
            filename_to_image_id[flat_name] = image_id
            image_id += 1

            # Symlink flat into output split folder
            dst_path = out_dir / flat_name
            if not dst_path.exists():
                os.symlink(src_path.resolve(), dst_path)

        img_id = filename_to_image_id[flat_name]
        img_entry = images[img_id - 1]
        frame_w, frame_h = img_entry["width"], img_entry["height"]

        bx, by, bw, bh = normalized_to_coco_bbox(
            row["x1"], row["y1"], row["x2"], row["y2"], frame_w, frame_h
        )

        if bw < 2 or bh < 2:
            continue

        anns.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            }
        )
        ann_id += 1

    coco = {
        "info": {
            "description": f"CVB detection dataset — {split_name} split",
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
        f"{len(anns)} annotations, {skipped} rows skipped (missing frames)"
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main(raw_dir: str, out_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir)

    ava_dir = raw / "cvb_in_ava_format"
    raw_frames_dir = raw / "raw_frames"

    if not raw_frames_dir.exists():
        raise FileNotFoundError(f"raw_frames not found: {raw_frames_dir}")
    if not ava_dir.exists():
        raise FileNotFoundError(f"cvb_in_ava_format not found: {ava_dir}")

    print(f"\nConverting CVB → COCO detection format")
    print(f"  Source : {raw}")
    print(f"  Output : {out}\n")

    for split_name, config in SPLIT_CONFIGS.items():
        csv_path = ava_dir / config["csv"]
        if not csv_path.exists():
            print(f"  ⚠ Skipping {split_name}: {config['csv']} not found")
            continue

        print(f"Loading {config['csv']} ...")
        df = load_csv(csv_path)
        print(f"  {len(df)} rows loaded (before filtering)")

        convert_split(
            df=df,
            raw_frames_dir=raw_frames_dir,
            out_dir=out / split_name,
            split_name=split_name,
        )

    print("\n✓ CVB conversion complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CVB AVA annotations to COCO detection format"
    )
    parser.add_argument(
        "--raw_dir", default="data/raw/cvb", help="Path to raw CVB dataset folder"
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/detection/cvb",
        help="Output folder for COCO-format dataset",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
