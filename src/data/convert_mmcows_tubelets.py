"""
src/data/convert_mmcows_tubelets.py

Generates VideoMAE tubelets from mmCows cam_4 ground-truth bounding boxes and
per-cow behavior CSVs, bypassing the RF-DETR → OC-SORT pipeline.

Input structure:
    data/raw/mmCows/images/0725/cam_4/         {ts}_{HH-MM-SS}.jpg  (4480×2800)
    data/raw/mmCows/labels/combined/0725/cam_4/ {ts}_{HH-MM-SS}.txt  (YOLO: cow_id cx cy w h)
    data/raw/mmCows/behavior_labels/individual/ C{NN}_0725.csv        (timestamp,datetime,behavior)

mmCows behavior code → canonical label_id mapping:
    0 → DROP  (unknown/unlabeled)
    1 → DROP  (walking — insufficient support)
    2 → 0     Standing
    3 → 2     Foraging/Grazing (feeding head-up)
    4 → 2     Foraging/Grazing (feeding head-down)
    5 → 5     Grooming (licking)
    6 → DROP  (drinking — insufficient support)
    7 → 1     Lying

Tubelet parameters: window=16 consecutive frames, stride=1, crop=224×224 px.

Inclusion criteria (all must hold):
    1. Target cow bbox present in all 16 frames.
    2. No frame has behavior code 0 (unknown).
    3. Majority canonical label_id appears in ≥12 of 16 frames.
    4. Majority canonical label_id ∈ {0, 1, 2, 5}.

Output:
    data/processed/tubelets/mmcows/{cow_id}/{start_ts}/frame_{i:02d}.jpg
    data/processed/tubelets/mmcows/labels.csv

Usage:
    python src/data/convert_mmcows_tubelets.py
    python src/data/convert_mmcows_tubelets.py \\
        --raw_dir  data/raw/mmCows \\
        --out_dir  data/processed/tubelets/mmcows \\
        --date     0725 --cam cam_4
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

IMG_W = 4480
IMG_H = 2800
CROP_SIZE = 224
CLIP_LEN = 16
STRIDE = 1
MAJORITY_THRESH = 12

# mmCows behavior code → canonical label_id (None = DROP)
LABEL_MAP: dict[int, int | None] = {
    0: None,
    1: None,
    2: 0,
    3: 2,
    4: 2,
    5: 5,
    6: None,
    7: 1,
}

EVAL_CLASSES: frozenset[int] = frozenset({0, 1, 2, 5})


def load_behavior_csvs(
    behavior_dir: Path, date: str
) -> dict[int, dict[int, int]]:
    """Load per-cow behavior CSVs.

    Returns {cow_id (1-16): {unix_timestamp: behavior_code}}.
    """
    result: dict[int, dict[int, int]] = {}
    for cow_id in range(1, 17):
        csv_path = behavior_dir / f"C{cow_id:02d}_{date}.csv"
        if not csv_path.exists():
            continue
        ts_map: dict[int, int] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_map[int(row["timestamp"])] = int(float(row["behavior"]))
        result[cow_id] = ts_map
    return result


def load_label_files(
    label_dir: Path,
) -> tuple[list[tuple[int, str]], dict[int, dict[int, tuple[float, float, float, float]]]]:
    """Load YOLO label files for a camera.

    Returns:
        sorted_images: sorted list of (unix_timestamp, stem) for all label files
        frame_bboxes:  {unix_timestamp: {cow_id: (cx, cy, w, h)}}
    """
    frame_bboxes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    stems: list[tuple[int, str]] = []

    for label_path in label_dir.glob("*.txt"):
        stem = label_path.stem
        try:
            ts = int(stem.split("_")[0])
        except (ValueError, IndexError):
            continue

        stems.append((ts, stem))
        cow_bboxes: dict[int, tuple[float, float, float, float]] = {}

        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cow_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            cow_bboxes[cow_id] = (cx, cy, w, h)

        frame_bboxes[ts] = cow_bboxes

    stems.sort()
    return stems, frame_bboxes


def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float
) -> tuple[int, int, int, int]:
    """Convert normalised YOLO cx/cy/w/h to absolute pixel xyxy (clamped)."""
    px = cx * IMG_W
    py = cy * IMG_H
    pw = w * IMG_W
    ph = h * IMG_H
    x1 = max(0, int(px - pw / 2))
    y1 = max(0, int(py - ph / 2))
    x2 = min(IMG_W, int(px + pw / 2))
    y2 = min(IMG_H, int(py + ph / 2))
    return x1, y1, x2, y2


def _crop_and_resize(img, x1: int, y1: int, x2: int, y2: int):
    """Crop xyxy region from img and resize to CROP_SIZE×CROP_SIZE."""
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


def convert(
    raw_dir: Path,
    out_dir: Path | None,
    date: str = "0725",
    cam: str = "cam_4",
) -> list[dict]:
    """Generate mmCows tubelets and write labels.csv.

    Args:
        raw_dir: path to mmCows raw dataset root
        out_dir: output directory (None = dry-run, return rows without writing)
        date:    date subfolder (default: 0725)
        cam:     camera subfolder (default: cam_4)

    Returns:
        list of label row dicts with keys:
        dataset, video_id, tubelet_dir, start_frame, end_frame, label_id, split
    """
    label_dir = raw_dir / "labels" / "combined" / date / cam
    image_dir = raw_dir / "images" / date / cam
    behavior_dir = raw_dir / "behavior_labels" / "individual"

    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not behavior_dir.exists():
        raise FileNotFoundError(f"Behavior dir not found: {behavior_dir}")

    sorted_images, frame_bboxes = load_label_files(label_dir)
    if not sorted_images:
        raise FileNotFoundError(f"No .txt label files in {label_dir}")

    behavior = load_behavior_csvs(behavior_dir, date)
    if not behavior:
        raise FileNotFoundError(f"No behavior CSVs found in {behavior_dir}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    video_id = f"{cam.replace('_', '')}_{date}"  # e.g. cam4_0725
    n_frames = len(sorted_images)
    labels_rows: list[dict] = []

    for cow_id in tqdm(range(1, 17), desc="mmcows-tubelets"):
        cow_behavior = behavior.get(cow_id)
        if cow_behavior is None:
            continue

        # Slide a 16-frame window with stride=1 over the sorted image list
        for start_idx in range(n_frames - CLIP_LEN + 1):
            window = sorted_images[start_idx: start_idx + CLIP_LEN]

            # --- Criterion 1: cow bbox present in all 16 frames ---
            if any(cow_id not in frame_bboxes.get(ts, {}) for ts, _ in window):
                continue

            # --- Criterion 2: no code-0 frame ---
            codes = [cow_behavior.get(ts, 0) for ts, _ in window]
            if any(c == 0 for c in codes):
                continue

            # --- Map codes to canonical labels (drop unknown-mapped) ---
            canonical = [LABEL_MAP.get(c) for c in codes]
            if any(lbl is None for lbl in canonical):
                continue

            # --- Criterion 3: majority label ≥ MAJORITY_THRESH ---
            counts = Counter(canonical)
            majority_label, majority_count = counts.most_common(1)[0]
            if majority_count < MAJORITY_THRESH:
                continue

            # --- Criterion 4: majority label in eval classes ---
            if majority_label not in EVAL_CLASSES:
                continue

            # --- Build tubelet ---
            start_ts = window[0][0]
            cow_dir_name = f"C{cow_id:02d}"
            tubelet_subdir = f"{start_ts}"

            if out_dir is not None:
                tubelet_path = out_dir / cow_dir_name / tubelet_subdir
                tubelet_path.mkdir(parents=True, exist_ok=True)

                save_ok = True
                for frame_i, (ts, stem) in enumerate(window):
                    img_path = image_dir / f"{stem}.jpg"
                    img = cv2.imread(str(img_path))
                    if img is None:
                        save_ok = False
                        break

                    cx, cy, w, h = frame_bboxes[ts][cow_id]
                    x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h)
                    crop = _crop_and_resize(img, x1, y1, x2, y2)
                    if crop is None:
                        save_ok = False
                        break

                    cv2.imwrite(
                        str(tubelet_path / f"frame_{frame_i:02d}.jpg"),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )

                if not save_ok:
                    shutil.rmtree(tubelet_path, ignore_errors=True)
                    continue

                tubelet_dir_str = str(tubelet_path)
            else:
                tubelet_dir_str = str(
                    Path("data/processed/tubelets/mmcows") / cow_dir_name / tubelet_subdir
                )

            labels_rows.append({
                "dataset":     "mmcows",
                "video_id":    video_id,
                "tubelet_dir": tubelet_dir_str,
                "start_frame": start_idx,
                "end_frame":   start_idx + CLIP_LEN,
                "label_id":    majority_label,
                "split":       "eval",
            })

    if out_dir is not None and labels_rows:
        csv_path = out_dir / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "video_id", "tubelet_dir",
                            "start_frame", "end_frame", "label_id", "split"],
            )
            writer.writeheader()
            writer.writerows(labels_rows)

        print(f"\nmmCows tubelet conversion complete.")
        print(f"  Tubelets generated : {len(labels_rows)}")
        print(f"  Labels CSV         : {csv_path}")

    return labels_rows


def main(
    raw_dir: str,
    out_dir: str,
    date: str = "0725",
    cam: str = "cam_4",
) -> None:
    convert(Path(raw_dir), Path(out_dir), date=date, cam=cam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mmCows tubelets from ground-truth bounding boxes"
    )
    parser.add_argument("--raw_dir", default="data/raw/mmCows")
    parser.add_argument("--out_dir", default="data/processed/tubelets/mmcows")
    parser.add_argument("--date", default="0725")
    parser.add_argument("--cam", default="cam_4")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, date=args.date, cam=args.cam)
