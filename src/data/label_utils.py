"""Label mapping utilities for CVB, CBVD-5, and Freeman Center datasets."""

LABEL_NAMES = {
    0: "Standing",
    1: "Lying",
    2: "Foraging",
    3: "Drinking",
    4: "Ruminating",
    5: "Grooming",
    6: "Other",
}

_CVB_BEHAVIOR_MAP = {
    "resting-standing": 0,
    "resting-lying": 1,
    "grazing": 2,
    "drinking": 3,
    "ruminating-standing": 4,
    "ruminating-lying": 4,
    "grooming": 5,
    "other": 6,
    # SKIP behaviors map to None
    "hidden": None,
    "walking": None,
    "running": None,
    "none": None,
}

# Priority order: higher index = higher priority wins
# 4 (Drinking) > 3 (Foraging) > 5 (Ruminating) > 2 (Lying) > 1 (Standing)
_CBVD5_ACTION_MAP = {
    1: 0,  # stand → Standing
    2: 1,  # lying down → Lying
    3: 2,  # foraging → Foraging
    4: 3,  # drinking water → Drinking
    5: 4,  # rumination → Ruminating
}

_CBVD5_PRIORITY = [1, 2, 5, 3, 4]  # ascending priority (last = highest)

# Freeman Center (CMB 2024) — 9-class YOLO taxonomy → canonical 7-class IDs.
# 'normal' (2) resolved to Standing (0): upright alert posture with no active behavior.
# Classes 3,5,7,8 map to Other (6); no Lying or Drinking in this dataset.
_FREEMAN_LABEL_MAP = {
    0: 2,     # hay feeding   → Foraging/Grazing
    1: 2,     # grazing       → Foraging/Grazing
    2: 0,     # normal        → Standing  (RESOLVED: Option A)
    3: 6,     # dominance assertion → Other
    4: 4,     # ruminating    → Ruminating
    5: 6,     # fear response → Other
    6: 5,     # grooming      → Grooming
    7: 6,     # vocalizing    → Other
    8: 6,     # sniffing      → Other
}


def cvb_behavior_to_label(behavior_str: str) -> int | None:
    """Map CVB behavior string to canonical label ID. Returns None for SKIP behaviors."""
    return _CVB_BEHAVIOR_MAP.get(behavior_str.lower().strip())


def cbvd5_actions_to_label(action_ids: list[int]) -> int:
    """Apply priority rule to multi-label CBVD-5 action IDs and return canonical label ID.

    Priority (highest wins): Drinking(4) > Foraging(3) > Ruminating(5) > Lying(2) > Standing(1)
    Unknown action IDs are ignored.
    """
    best = None
    for action_id in action_ids:
        if action_id not in _CBVD5_PRIORITY:
            continue
        if best is None or _CBVD5_PRIORITY.index(action_id) > _CBVD5_PRIORITY.index(best):
            best = action_id
    if best is None:
        raise ValueError(f"No known action IDs in: {action_ids}")
    return _CBVD5_ACTION_MAP[best]


def freeman_label_to_canonical(freeman_id: int) -> int:
    """Map a Freeman Center class ID (0–8) to the canonical 7-class label ID."""
    if freeman_id not in _FREEMAN_LABEL_MAP:
        raise ValueError(f"Unknown Freeman class ID: {freeman_id}")
    return _FREEMAN_LABEL_MAP[freeman_id]


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def match_predicted_to_gt(
    pred_bboxes: list[list[float]],
    gt_bboxes: list[list[float]],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Hungarian match predicted to GT boxes; return {pred_idx: gt_idx} for IoU >= threshold."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not pred_bboxes or not gt_bboxes:
        return {}

    n, m = len(pred_bboxes), len(gt_bboxes)
    cost = np.zeros((n, m), dtype=np.float64)
    for i, pb in enumerate(pred_bboxes):
        for j, gb in enumerate(gt_bboxes):
            cost[i, j] = 1.0 - bbox_iou(pb, gb)

    row_ind, col_ind = linear_sum_assignment(cost)
    result = {}
    for r, c in zip(row_ind, col_ind):
        if (1.0 - cost[r, c]) >= iou_threshold:
            result[int(r)] = int(c)
    return result


def load_cvb_gt(ann_json_path: str) -> dict[int, list[dict]]:
    """Load CVB COCO annotation JSON and return {frame_int: [{"bbox_xyxy", "label_id"}]}.

    Skips annotations whose behavior maps to None (SKIP behaviors).
    Converts COCO [x, y, w, h] bbox to [x1, y1, x2, y2].
    frame_int == image_id (verified: image_id matches img_{frame:05d}.jpg numbering).
    """
    import json

    with open(ann_json_path) as f:
        data = json.load(f)

    result: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        behavior = ann["attributes"]["behavior"]
        label_id = cvb_behavior_to_label(behavior)
        if label_id is None:
            continue
        x, y, w, h = ann["bbox"]
        bbox_xyxy = [x, y, x + w, y + h]
        frame_int = ann["image_id"]
        result.setdefault(frame_int, []).append({"bbox_xyxy": bbox_xyxy, "label_id": label_id})

    return result


def load_cbvd5_annotations() -> list[dict]:
    """Read all three CBVD-5 AVA CSVs and return one entry per unique (video_id, timestamp, bbox).

    Multi-label rows sharing the same (video_id, timestamp, bbox_rounded) are grouped and the
    priority rule (§3.4) is applied to select a single label_id.
    """
    import csv
    import os

    base = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "cbvd5", "annotations"
    )

    # group action_ids by (video_id, timestamp, bbox_key); track split per key
    from collections import defaultdict
    groups: dict[tuple, list[int]] = defaultdict(list)
    group_meta: dict[tuple, tuple] = {}  # key → (video_id, timestamp, bbox_norm, split)

    for split, fname in (
        ("train", "ava_train_v2.1.csv"),
        ("val", "ava_val_v2.1.csv"),
        ("test", "ava_test_v2.1.csv"),
    ):
        with open(os.path.join(base, fname)) as f:
            for row in csv.reader(f):
                video_id = row[0]
                timestamp = float(row[1])
                bbox_norm = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                action_id = int(row[6])
                bbox_key = tuple(round(x, 4) for x in bbox_norm)
                key = (video_id, timestamp, bbox_key)
                groups[key].append(action_id)
                if key not in group_meta:
                    group_meta[key] = (video_id, timestamp, bbox_norm, split)

    result = []
    for key, action_ids in groups.items():
        video_id, timestamp, bbox_norm, split = group_meta[key]
        label_id = cbvd5_actions_to_label(action_ids)
        result.append(
            {
                "video_id": video_id,
                "timestamp": timestamp,
                "frame_center": int(timestamp * 25),
                "bbox_norm": bbox_norm,
                "label_id": label_id,
                "split": split,
            }
        )
    return result


def extract_cbvd5_frames(video_path: str, start_frame: int, count: int = 16) -> list:
    """Open an .mp4, seek to start_frame, read count consecutive BGR frames.

    Returns list of numpy arrays (H, W, 3) at original resolution.
    Raises ValueError if fewer than count frames can be read.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < count:
        raise ValueError(
            f"Expected {count} frames from {video_path} at start_frame={start_frame}, "
            f"got {len(frames)}"
        )
    return frames


def load_cvb_splits() -> dict[str, str]:
    """Return {video_id → "train"|"val"} from the official CVB AVA-format split CSVs."""
    import os

    base = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "cvb", "cvb_in_ava_format"
    )
    result: dict[str, str] = {}
    for split, fname in (("train", "ava_train_set.csv"), ("val", "ava_val_set.csv")):
        with open(os.path.join(base, fname)) as f:
            for line in f:
                video_id = line.split(",")[0]
                result[video_id] = split
    return result
