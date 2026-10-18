"""
src/data/make_mini.py

Creates a small subset of the combined COCO dataset for fast sanity-check
training runs. Use this constantly during development before committing to
a full training run.

Default: 50 images per split (train/valid/test), sampled evenly across
both source datasets (cbvd5 and cvb).

Input:
    data/processed/detection/combined/{train,valid,test}/

Output:
    data/processed/detection/combined_mini/{train,valid,test}/
        ├── _annotations.coco.json
        └── *.jpg (symlinks)

Usage:
    python src/data/make_mini.py
    python src/data/make_mini.py --n_images 50 --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm


SPLITS = ["train", "valid", "test"]


def make_mini_split(
    src_dir: Path,
    out_dir: Path,
    split: str,
    n_images: int,
    seed: int,
):
    ann_file = src_dir / split / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"  ⚠ {split}: _annotations.coco.json not found, skipping")
        return

    with open(ann_file) as f:
        coco = json.load(f)

    all_images = coco["images"]

    # Sample evenly from cbvd5 and cvb prefixes if possible
    cbvd5_imgs = [im for im in all_images if im["file_name"].startswith("cbvd5__")]
    cvb_imgs = [im for im in all_images if im["file_name"].startswith("cvb__")]
    other_imgs = [
        im
        for im in all_images
        if not im["file_name"].startswith("cbvd5__")
        and not im["file_name"].startswith("cvb__")
    ]

    rng = random.Random(seed)

    # Try to get n_images/2 from each; fall back if one dataset has fewer
    half = n_images // 2
    cbvd5_sample = rng.sample(cbvd5_imgs, min(half, len(cbvd5_imgs)))
    cvb_sample = rng.sample(cvb_imgs, min(half, len(cvb_imgs)))
    other_sample = rng.sample(other_imgs, min(n_images, len(other_imgs)))

    # If we don't have enough from both datasets, top up from the other
    sampled = cbvd5_sample + cvb_sample + other_sample
    shortfall = n_images - len(sampled)
    if shortfall > 0:
        remainder = [im for im in all_images if im not in sampled]
        sampled += rng.sample(remainder, min(shortfall, len(remainder)))

    sampled = sampled[:n_images]
    sampled_ids = {im["id"] for im in sampled}

    # Filter annotations to sampled images only
    sampled_anns = [a for a in coco["annotations"] if a["image_id"] in sampled_ids]

    # Remap IDs to be sequential
    id_map = {old_id: new_id for new_id, old_id in enumerate(sampled_ids, start=1)}
    new_images = []
    for im in sampled:
        new_im = dict(im)
        new_im["id"] = id_map[im["id"]]
        new_images.append(new_im)

    new_anns = []
    for i, ann in enumerate(sampled_anns, start=1):
        new_ann = dict(ann)
        new_ann["id"] = i
        new_ann["image_id"] = id_map[ann["image_id"]]
        new_anns.append(new_ann)

    # Write output
    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Symlink images
    src_split = src_dir / split
    skipped = 0
    for im in tqdm(new_images, desc=f"  Symlinking {split}"):
        src = src_split / im["file_name"]
        dst = out_split / im["file_name"]
        if not src.exists():
            skipped += 1
            continue
        if not dst.exists():
            os.symlink(src.resolve(), dst)

    mini_coco = {
        "info": {
            "description": f"Mini combined dataset — {split} split ({n_images} images)",
            "version": "1.0",
        },
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_anns,
    }

    with open(out_split / "_annotations.coco.json", "w") as f:
        json.dump(mini_coco, f, indent=2)

    print(
        f"  ✓ mini/{split}: {len(new_images)} images, "
        f"{len(new_anns)} annotations ({skipped} skipped)"
    )


def main(src_dir: str, out_dir: str, n_images: int, seed: int):
    src = Path(src_dir)
    out = Path(out_dir)

    print(f"\nCreating mini dataset ({n_images} images per split, seed={seed})")
    print(f"  Source: {src}")
    print(f"  Output: {out}\n")

    for split in SPLITS:
        make_mini_split(src, out, split, n_images, seed)

    print("\n✓ Mini dataset created.\n")
    print(f"Run a sanity-check training with:")
    print(
        f"  python src/detection/train.py --config configs/detection/rfdetr_combined.yaml \\"
    )
    print(f"    --dataset_dir {out} --epochs 2\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a mini subset of the combined dataset for fast sanity checks"
    )
    parser.add_argument("--src_dir", default="data/processed/detection/combined")
    parser.add_argument("--out_dir", default="data/processed/detection/combined_mini")
    parser.add_argument(
        "--n_images", type=int, default=50, help="Number of images per split"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.src_dir, args.out_dir, args.n_images, args.seed)
