"""
src/data/merge_coco.py

Merges CBVD-5 and CVB COCO detection datasets into a single combined
dataset for joint RF-DETR training.

Input:
    data/processed/detection/cbvd5/{train,valid,test}/
    data/processed/detection/cvb/{train,valid,test}/

Output:
    data/processed/detection/combined/{train,valid,test}/
        ├── _annotations.coco.json  (merged, all IDs remapped globally)
        └── *.jpg                   (symlinks from source datasets)

Key rules:
    - All image_ids and annotation_ids are remapped to avoid collisions
    - Filenames are prefixed by dataset: cbvd5__<name> and cvb__<name>
    - Images are symlinked (not copied) to save disk space
    - The combined dataset uses the same single-class "cattle" schema

Usage:
    python src/data/merge_coco.py
    python src/data/merge_coco.py --det_dir data/processed/detection --out_dir data/processed/detection/combined
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


# ── Source datasets to merge ─────────────────────────────────────────────────

DATASETS = ["cbvd5", "cvb"]
SPLITS = ["train", "valid", "test"]


# ── Core merge ───────────────────────────────────────────────────────────────


def merge_split(
    det_dir: Path,
    out_dir: Path,
    split: str,
):
    """
    Merge one split (train/valid/test) across all datasets.

    Strategy:
    - Iterate datasets in order (cbvd5 first, then cvb)
    - Remap each dataset's image_ids starting from current max+1
    - Remap each dataset's annotation_ids starting from current max+1
    - Prefix filenames: cbvd5__618_00002.jpg, cvb__1321_...jpg
    - Symlink prefixed filename → original symlink target
    """
    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    merged_images = []
    merged_annotations = []
    merged_categories = None  # taken from first dataset, same for all
    global_image_id = 1
    global_ann_id = 1
    total_skipped = 0

    for dataset in DATASETS:
        ann_file = det_dir / dataset / split / "_annotations.coco.json"

        if not ann_file.exists():
            print(f"  ⚠ {dataset}/{split}: _annotations.coco.json not found, skipping")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        # Use categories from first dataset found
        if merged_categories is None:
            merged_categories = coco["categories"]

        # Build local → global id mapping for images
        local_to_global_image_id = {}

        src_dir = det_dir / dataset / split

        print(
            f"  Merging {dataset}/{split}: "
            f"{len(coco['images'])} images, {len(coco['annotations'])} anns"
        )

        for img in tqdm(coco["images"], desc=f"    {dataset}/{split} images"):
            # New prefixed filename
            prefixed_name = f"{dataset}__{img['file_name']}"

            # Symlink: combined/split/prefixed_name → original file
            src_path = src_dir / img["file_name"]
            dst_path = split_out / prefixed_name

            if not src_path.exists():
                total_skipped += 1
                continue

            if not dst_path.exists():
                # Resolve the symlink chain to get the real file
                real_path = src_path.resolve()
                os.symlink(real_path, dst_path)

            # Record id mapping
            local_to_global_image_id[img["id"]] = global_image_id

            merged_images.append(
                {
                    "id": global_image_id,
                    "file_name": prefixed_name,
                    "width": img["width"],
                    "height": img["height"],
                }
            )
            global_image_id += 1

        # Remap annotations
        for ann in coco["annotations"]:
            local_img_id = ann["image_id"]
            if local_img_id not in local_to_global_image_id:
                # Image was skipped above
                continue

            merged_annotations.append(
                {
                    "id": global_ann_id,
                    "image_id": local_to_global_image_id[local_img_id],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                }
            )
            global_ann_id += 1

    if merged_categories is None:
        merged_categories = [{"id": 1, "name": "cattle", "supercategory": "animal"}]

    merged_coco = {
        "info": {
            "description": f"Combined CBVD-5 + CVB detection dataset — {split} split",
            "version": "1.0",
            "datasets": DATASETS,
        },
        "categories": merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    out_file = split_out / "_annotations.coco.json"
    with open(out_file, "w") as f:
        json.dump(merged_coco, f, indent=2)

    print(
        f"  ✓ combined/{split}: {len(merged_images)} images, "
        f"{len(merged_annotations)} annotations "
        f"({total_skipped} images skipped)\n"
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main(det_dir: str, out_dir: str):
    det = Path(det_dir)
    out = Path(out_dir)

    print(f"\nMerging datasets: {DATASETS}")
    print(f"  Source: {det}")
    print(f"  Output: {out}\n")

    for split in SPLITS:
        merge_split(det_dir=det, out_dir=out, split=split)

    print("✓ Merge complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge CBVD-5 and CVB COCO datasets into combined dataset"
    )
    parser.add_argument(
        "--det_dir",
        default="data/processed/detection",
        help="Parent folder containing cbvd5/ and cvb/ detection datasets",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed/detection/combined",
        help="Output folder for merged combined dataset",
    )
    args = parser.parse_args()
    main(args.det_dir, args.out_dir)
