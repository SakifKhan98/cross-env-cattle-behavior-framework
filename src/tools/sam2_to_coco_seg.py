"""
src/tools/sam2_to_coco_seg.py

PURPOSE:
    Convert SAM2 mask JSONs (Phase 3 output) into a COCO instance segmentation
    dataset that RF-DETR-Seg can train on directly.

WHAT THIS DOES:
    SAM2 already outputs masks in COCO RLE format. This script:
    1. Walks every mask JSON in data/processed/segmentation/{cbvd5,cvb}/
    2. For each frame with masks, copies (or symlinks) the source image
    3. Writes a COCO-format _annotations.coco.json for train and val splits

OUTPUT STRUCTURE (RF-DETR-Seg expects this exactly):
    data/rfdetr_seg/cattle/
        train/
            images/           ← source frame images
            _annotations.coco.json
        valid/
            images/
            _annotations.coco.json

COCO FORMAT RECAP:
    {
      "images":      [{"id": int, "file_name": str, "width": int, "height": int}],
      "annotations": [{"id": int, "image_id": int, "category_id": int,
                        "bbox": [x,y,w,h], "segmentation": RLE or polygon,
                        "area": float, "iscrowd": 0}],
      "categories":  [{"id": 1, "name": "cattle", "supercategory": "animal"}]
    }

USAGE:
    # Default: 80/20 split, both datasets, max 3 frames per CVB clip
    python src/tools/sam2_to_coco_seg.py

    # Custom options:
    python src/tools/sam2_to_coco_seg.py \\
        --cbvd5_masks data/processed/segmentation/cbvd5 \\
        --cvb_masks   data/processed/segmentation/cvb \\
        --cbvd5_imgs  data/raw/cbvd5/labelframes/labelframes \\
        --cvb_imgs    data/raw/cvb/raw_frames \\
        --output_dir  data/rfdetr_seg/cattle \\
        --val_ratio   0.2 \\
        --cvb_stride  15 \\
        --seed        42

NOTES ON cvb_stride:
    CVB clips have 450 frames. Using every frame would produce ~225,000 nearly
    identical images. --cvb_stride 15 samples one frame per re-prompt window,
    giving diverse, non-redundant training examples. This matches the K=15
    interval used during SAM2 segmentation — every sampled frame is a 'fresh
    prompt' frame with high-confidence masks.

NOTES ON image copying:
    Images are hard-copied (not symlinked) for maximum compatibility with
    RF-DETR's data loader. This uses more disk space but avoids symlink issues
    on network filesystems.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert SAM2 mask JSONs to COCO segmentation dataset"
    )
    p.add_argument("--cbvd5_masks", default="data/processed/segmentation/cbvd5")
    p.add_argument("--cvb_masks", default="data/processed/segmentation/cvb")
    p.add_argument("--cbvd5_imgs", default="data/raw/cbvd5/labelframes/labelframes")
    p.add_argument("--cvb_imgs", default="data/raw/cvb/raw_frames")
    p.add_argument("--output_dir", default="data/rfdetr_seg/cattle")
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of images to use for validation (default: 0.2)",
    )
    p.add_argument(
        "--cvb_stride",
        type=int,
        default=15,
        help="Sample every Nth frame from CVB clips (default: 15 = one per reprompt window)",
    )
    p.add_argument(
        "--cbvd5_only", action="store_true", help="Only include CBVD-5 (skip CVB)"
    )
    p.add_argument(
        "--cvb_only", action="store_true", help="Only include CVB (skip CBVD-5)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min_area",
        type=int,
        default=100,
        help="Skip masks smaller than this many pixels (default: 100)",
    )
    return p.parse_args()


# ── Image path helpers ────────────────────────────────────────────────────────


def cbvd5_img_path(img_root: str, video_id: str, frame_idx: int) -> str:
    return os.path.join(img_root, f"{video_id}_{int(frame_idx):05d}.jpg")


def cvb_img_path(img_root: str, video_id: str, frame_idx: int) -> str:
    return os.path.join(img_root, video_id, f"img_{int(frame_idx):05d}.jpg")


# ── Frame collector ───────────────────────────────────────────────────────────


def collect_frames(
    mask_dir: str, img_root: str, dataset: str, cvb_stride: int, min_area: int
) -> list:
    """
    Walk all mask JSONs in mask_dir and return a list of frame records.

    Each record:
    {
        "img_path":    str,          full path to source image
        "video_id":    str,
        "frame_key":   str,
        "dataset":     str,
        "instances":   list of {bbox, mask_rle, mask_area}
    }

    Frames are included only if:
    - The source image file exists
    - At least one instance passes the min_area filter
    """
    records = []
    mask_files = sorted(Path(mask_dir).glob("*.json"))

    print(f"[collect] {dataset.upper()}: scanning {len(mask_files)} mask JSONs...")

    n_skipped_noimg = 0
    n_skipped_nomask = 0

    for mask_file in mask_files:
        with open(mask_file) as f:
            data = json.load(f)

        video_id = data["video_id"]
        frames = data.get("frames", {})

        # For CVB: sample every cvb_stride-th frame to avoid redundancy
        frame_keys = sorted(frames.keys(), key=lambda k: int(k))
        if dataset == "cvb" and cvb_stride > 1:
            frame_keys = [k for i, k in enumerate(frame_keys) if i % cvb_stride == 0]

        for frame_key in frame_keys:
            instances = frames[frame_key]

            # Filter by min_area
            valid_instances = [
                inst for inst in instances if inst.get("mask_area", 0) >= min_area
            ]
            if not valid_instances:
                n_skipped_nomask += 1
                continue

            # Check image exists
            if dataset == "cbvd5":
                img_path = cbvd5_img_path(img_root, video_id, int(frame_key))
            else:
                img_path = cvb_img_path(img_root, video_id, int(frame_key))

            if not os.path.isfile(img_path):
                n_skipped_noimg += 1
                continue

            records.append(
                {
                    "img_path": img_path,
                    "video_id": video_id,
                    "frame_key": frame_key,
                    "dataset": dataset,
                    "instances": valid_instances,
                }
            )

    print(
        f"[collect] {dataset.upper()}: {len(records)} usable frames "
        f"({n_skipped_noimg} missing images, {n_skipped_nomask} no valid masks)"
    )
    return records


# ── COCO JSON builder ─────────────────────────────────────────────────────────


def build_coco_json(records: list, split_dir: Path) -> dict:
    """
    Build a COCO instance segmentation JSON dict from a list of frame records.
    Copies source images into split_dir/images/.

    Returns the COCO dict (caller writes it to disk).
    """
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "Cattle Vision Framework — SAM2 pseudo-labels",
            "version": "1.0",
            "year": 2026,
            "contributor": "Sakif Khan, Texas State University",
        },
        "categories": [{"id": 1, "name": "cattle", "supercategory": "animal"}],
        "images": [],
        "annotations": [],
    }

    image_id = 1
    annotation_id = 1

    for rec in records:
        # ── Copy image ────────────────────────────────────────────────────────
        # Unique filename: {dataset}_{video_id}_frame{frame_key}.jpg
        dst_name = f"{rec['dataset']}_{rec['video_id']}_frame{rec['frame_key']}.jpg"
        dst_path = images_dir / dst_name

        if not dst_path.exists():
            shutil.copy2(rec["img_path"], dst_path)

        # ── Get image dimensions from the mask RLE size ───────────────────────
        # RLE size is [height, width]
        first_inst = rec["instances"][0]
        h, w = first_inst["mask_rle"]["size"]

        coco["images"].append(
            {"id": image_id, "file_name": dst_name, "width": w, "height": h}
        )

        # ── Add one annotation per instance ──────────────────────────────────
        for inst in rec["instances"]:
            rle = inst["mask_rle"]
            bbox = inst["bbox"]  # already COCO [x, y, w, h] float
            area = inst["mask_area"]

            # RF-DETR-Seg accepts RLE segmentation directly in COCO format.
            # The counts field from pycocotools is a byte string when loaded
            # from JSON it's already a regular string — pass it as-is.
            segmentation = {"counts": rle["counts"], "size": rle["size"]}

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(v, 2) for v in bbox],
                    "segmentation": segmentation,
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    return coco


# ── Stats printer ─────────────────────────────────────────────────────────────


def print_stats(split: str, records: list, coco: dict):
    n_images = len(coco["images"])
    n_annots = len(coco["annotations"])
    areas = [a["area"] for a in coco["annotations"]]
    avg_area = sum(areas) / len(areas) if areas else 0
    avg_per_img = n_annots / n_images if n_images else 0

    datasets = {}
    for r in records:
        datasets[r["dataset"]] = datasets.get(r["dataset"], 0) + 1

    print(f"\n  [{split}]")
    print(f"    Images      : {n_images:,}")
    print(f"    Annotations : {n_annots:,}")
    print(f"    Avg/image   : {avg_per_img:.1f} instances")
    print(f"    Avg area    : {avg_area:,.0f} px")
    for ds, cnt in sorted(datasets.items()):
        print(f"    {ds.upper():10s}: {cnt:,} frames")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all usable frames ─────────────────────────────────────────
    all_records = []

    if not args.cvb_only:
        cbvd5_records = collect_frames(
            args.cbvd5_masks,
            args.cbvd5_imgs,
            "cbvd5",
            cvb_stride=1,
            min_area=args.min_area,
        )
        all_records.extend(cbvd5_records)

    if not args.cbvd5_only:
        cvb_records = collect_frames(
            args.cvb_masks,
            args.cvb_imgs,
            "cvb",
            cvb_stride=args.cvb_stride,
            min_area=args.min_area,
        )
        all_records.extend(cvb_records)

    if not all_records:
        print("[ERROR] No usable frames found. Check paths and mask files.")
        sys.exit(1)

    print(f"\n[split] Total usable frames: {len(all_records):,}")

    # ── 2. Train / val split ─────────────────────────────────────────────────
    # Shuffle at video level to prevent frames from same video leaking across splits
    # Group by video_id first
    by_video = {}
    for rec in all_records:
        key = f"{rec['dataset']}_{rec['video_id']}"
        by_video.setdefault(key, []).append(rec)

    video_keys = sorted(by_video.keys())
    random.shuffle(video_keys)

    n_val_videos = max(1, int(len(video_keys) * args.val_ratio))
    val_keys = set(video_keys[:n_val_videos])
    train_keys = set(video_keys[n_val_videos:])

    train_records = [r for k in train_keys for r in by_video[k]]
    val_records = [r for k in val_keys for r in by_video[k]]

    # Shuffle frame order within each split
    random.shuffle(train_records)
    random.shuffle(val_records)

    print(
        f"[split] Train: {len(train_records):,} frames from {len(train_keys):,} videos"
    )
    print(f"[split] Val:   {len(val_records):,} frames from {len(val_keys):,} videos")
    print(f"[split] Val ratio (actual): {len(val_records)/len(all_records):.1%}")

    # ── 3. Build and write COCO JSONs ─────────────────────────────────────────
    for split, records in [("train", train_records), ("valid", val_records)]:
        print(f"\n[build] Building {split} split ({len(records):,} frames)...")
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        coco = build_coco_json(records, split_dir)

        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(coco, f)

        print(f"[build] Wrote: {json_path}")
        print_stats(split, records, coco)

    # ── 4. Print training command ─────────────────────────────────────────────
    print(
        f"""
{'='*60}
[done] Dataset ready at: {out_dir}

To train RF-DETR-Seg-Medium:

    from rfdetr import RFDETRSegMedium
    model = RFDETRSegMedium()
    model.train(
        dataset_dir="{out_dir}",
        epochs=100,
        batch_size=2,
        grad_accum_steps=8,   # effective batch = 16
        lr=1e-4,
        output_dir="runs/segmentation/rfdetr_seg_cattle_v1"
    )

Or via shell script:
    bash scripts/08_train_rfdetr_seg.sh
{'='*60}
"""
    )


if __name__ == "__main__":
    main()
