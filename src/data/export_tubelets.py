"""Tubelet export: CVB and CBVD-5."""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.label_utils import (
    LABEL_NAMES,
    load_cvb_gt,
    load_cbvd5_annotations,
    extract_cbvd5_frames,
    match_predicted_to_gt,
)

_ONE_DAY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CVB_TRACKING = os.path.join(_ONE_DAY, "data", "processed", "tracking_v2", "cvb")
_CVB_FRAMES = os.path.join(_ONE_DAY, "data", "raw", "cvb", "raw_frames")
_CVB_ANN = os.path.join(_ONE_DAY, "data", "raw", "cvb", "annotations")
_CBVD5_TRACKING = os.path.join(_ONE_DAY, "data", "processed", "tracking_v2", "cbvd5")
_CBVD5_VIDEOS = os.path.join(_ONE_DAY, "data", "raw", "cbvd5", "videos", "videos")


def _ann_path(video_id: str) -> str | None:
    """Return annotation JSON path for a CVB video_id, or None if missing."""
    p = os.path.join(_CVB_ANN, video_id, "annotations", "instances_default.json")
    return p if os.path.isfile(p) else None


def export_tubelets_from_tracks(
    tracks_json: dict,
    video_path: str,
    output_dir: str,
    clip_len: int = 16,
    stride: int = 8,
    pad: int = 20,
) -> list:
    """Extract tubelet clips from any video file using tracking results.

    No dataset-specific paths or constants. Works with any OpenCV-readable video.

    Args:
        tracks_json: Tracks dict from run_tracking (CLAUDE.md §6 schema).
                     bbox values must be [x1,y1,x2,y2] (xyxy).
        video_path:  Path to the source video file.
        output_dir:  Directory under which to write tubelet frame crops.
                     Structure: {output_dir}/track_{id:04d}/tubelet_{idx:04d}/frame_{i:02d}.jpg
        clip_len:    Frames per tubelet (default 16).
        stride:      Sliding-window step size (default 8).
        pad:         Pixel padding around crop bbox (default 20).

    Returns:
        List of dicts: {tubelet_dir, track_id, start_frame, end_frame}.
        Tracks with fewer than clip_len frames produce no entry.
    """
    import shutil as _shutil
    from pathlib import Path as _Path

    frames_dict = tracks_json.get("frames", {})
    output_dir = _Path(output_dir)

    # Build per-track data structures
    track_frames: dict = {}
    track_bboxes: dict = {}
    for fid_str, dets in frames_dict.items():
        fi = int(fid_str)
        for det in dets:
            tid = det["track_id"]
            track_frames.setdefault(tid, []).append(fi)
            track_bboxes.setdefault(tid, {})[fi] = det["bbox"]

    for tid in track_frames:
        track_frames[tid].sort()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # ── Collect all frame indices needed across every tubelet window ──────────
    needed_frames: set = set()
    valid_tracks = {}
    for tid, frames in sorted(track_frames.items()):
        if len(frames) < clip_len:
            continue
        frame_set = set(frames)
        min_f, max_f = frames[0], frames[-1]
        windows = []
        start = min_f
        while start + clip_len <= max_f + 1:
            window = list(range(start, start + clip_len))
            present = sum(1 for f in window if f in frame_set)
            if present >= (clip_len * 3 // 4):
                windows.append(window)
                needed_frames.update(window)
            start += stride
        if windows:
            valid_tracks[tid] = (frame_set, windows)

    # ── One sequential pass — grab() skips without decode, read() decodes ────
    frame_cache: dict = {}
    if needed_frames:
        needed_sorted = sorted(needed_frames)
        needed_set = set(needed_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, needed_sorted[0])
        current = needed_sorted[0]
        for target in needed_sorted:
            while current < target:
                if not cap.grab():
                    break
                current += 1
            if current == target:
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    frame_cache[current] = frame
                if not cap.grab():
                    break
                current += 1

    cap.release()

    # ── Build tubelets from cached frames ─────────────────────────────────────
    tubelet_rows = []
    for tid, (frame_set, windows) in valid_tracks.items():
        bbox_keys = sorted(track_bboxes[tid].keys())
        tubelet_idx = 0
        for window in windows:
            out_dir = output_dir / f"track_{tid:04d}" / f"tubelet_{tubelet_idx:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            save_ok = True
            for i, fi in enumerate(window):
                frame = frame_cache.get(fi)
                if frame is None:
                    save_ok = False
                    break

                if fi in track_bboxes[tid]:
                    bbox = track_bboxes[tid][fi]
                else:
                    nearest = min(bbox_keys, key=lambda k: abs(k - fi))
                    bbox = track_bboxes[tid][nearest]

                crop = _crop_frame(frame, bbox, pad)
                if crop is None or crop.size == 0:
                    save_ok = False
                    break

                cv2.imwrite(
                    str(out_dir / f"frame_{i:02d}.jpg"),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            if save_ok:
                tubelet_rows.append({
                    "tubelet_dir": str(out_dir),
                    "track_id": tid,
                    "start_frame": window[0],
                    "end_frame": window[-1] + 1,
                })
                tubelet_idx += 1
            else:
                _shutil.rmtree(out_dir, ignore_errors=True)

    return tubelet_rows


def _crop_frame(img: any, bbox: list[float], pad: int = 20) -> any:
    """Crop img ([H,W,C] numpy) using [x1,y1,x2,y2] bbox + padding, clamped to image bounds."""
    H, W = img.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(W, int(bbox[2]) + pad)
    y2 = min(H, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def export_cvb_tubelets(
    output_root: str,
    labels_rows: list,
    split_map: dict[str, str],
    max_videos: int | None = None,
) -> None:
    """Export CVB tubelets, appending label dicts to labels_rows.

    Args:
        output_root: root directory for tubelets (e.g. data/processed/tubelets)
        labels_rows: list to append label row dicts into
        split_map: {video_id → "train"|"val"} from load_cvb_splits()
        max_videos: if set, stop after this many videos (for testing)
    """
    tracking_files = sorted(os.listdir(_CVB_TRACKING))
    processed = 0

    for fname in tracking_files:
        if max_videos is not None and processed >= max_videos:
            break
        if not fname.endswith("_tracks.json"):
            continue

        video_id = fname[: -len("_tracks.json")]
        if video_id not in split_map:
            continue

        ann_path = _ann_path(video_id)
        if ann_path is None:
            continue

        split = split_map[video_id]

        # Step 1 — load tracking: {frame_int → [{track_id, bbox}]}
        with open(os.path.join(_CVB_TRACKING, fname)) as f:
            track_data = json.load(f)
        predicted: dict[int, list[dict]] = {
            int(k): v for k, v in track_data["frames"].items()
        }

        # Step 2 — load GT annotations
        gt = load_cvb_gt(ann_path)

        # Step 3 — per-frame label lookup: {frame_int → {track_id → label_id}}
        frame_labels: dict[int, dict[int, int]] = {}
        for frame_int in predicted:
            if frame_int not in gt:
                continue
            pred_dets = predicted[frame_int]
            gt_entries = gt[frame_int]
            pred_bboxes = [det["bbox"] for det in pred_dets]
            gt_bboxes = [e["bbox_xyxy"] for e in gt_entries]
            matches = match_predicted_to_gt(pred_bboxes, gt_bboxes)
            if not matches:
                continue
            for pred_idx, gt_idx in matches.items():
                tid = pred_dets[pred_idx]["track_id"]
                label_id = gt_entries[gt_idx]["label_id"]
                frame_labels.setdefault(frame_int, {})[tid] = label_id

        # Step 4 — build tracks dict: {track_id → sorted frame list}
        tracks: dict[int, list[int]] = {}
        for frame_int, dets in predicted.items():
            for det in dets:
                tid = det["track_id"]
                tracks.setdefault(tid, []).append(frame_int)
        for tid in tracks:
            tracks[tid].sort()

        # Step 5 — slide tubelet windows
        frames_dir = os.path.join(_CVB_FRAMES, video_id)
        cvb_out = os.path.join(output_root, "cvb", video_id)

        for tid, track_frames in tracks.items():
            track_frame_set = set(track_frames)
            min_frame = track_frames[0]
            max_frame = track_frames[-1]

            # build {frame_int → bbox} lookup for this track
            track_bbox: dict[int, list[float]] = {}
            for fi, dets in predicted.items():
                for det in dets:
                    if det["track_id"] == tid:
                        track_bbox[fi] = det["bbox"]

            tubelet_idx = 0
            start = min_frame
            while start + 16 <= max_frame + 1:
                window = list(range(start, start + 16))
                # Require at least 12 of 16 frames present in track (allows short gaps)
                present_count = sum(1 for f in window if f in track_frame_set)
                if present_count >= 12:
                    # Collect labels for frames that have a match
                    window_labels = [
                        frame_labels[f][tid]
                        for f in window
                        if f in frame_labels and tid in frame_labels[f]
                    ]
                    if len(window_labels) >= 8:
                        # Majority vote
                        from collections import Counter
                        tubelet_label = Counter(window_labels).most_common(1)[0][0]

                        # Step 6 — crop and save
                        out_dir = os.path.join(
                            cvb_out,
                            f"track_{tid:04d}",
                            f"tubelet_{tubelet_idx:04d}",
                        )
                        os.makedirs(out_dir, exist_ok=True)

                        save_ok = True
                        sorted_bbox_frames = sorted(track_bbox.keys())
                        for i, frame_int in enumerate(window):
                            img_path = os.path.join(
                                frames_dir, f"img_{frame_int:05d}.jpg"
                            )
                            img = cv2.imread(img_path)
                            if img is None:
                                save_ok = False
                                break
                            # Use exact bbox if available, else nearest-neighbor interpolation
                            if frame_int in track_bbox:
                                bbox = track_bbox[frame_int]
                            else:
                                nearest = min(sorted_bbox_frames, key=lambda f: abs(f - frame_int))
                                bbox = track_bbox[nearest]
                            crop = _crop_frame(img, bbox)
                            if crop is None or crop.size == 0:
                                save_ok = False
                                break
                            cv2.imwrite(
                                os.path.join(out_dir, f"frame_{i:02d}.jpg"),
                                crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95],
                            )
                        if save_ok:
                            tubelet_dir = os.path.relpath(out_dir, _ONE_DAY)
                            labels_rows.append(
                                {
                                    "dataset": "cvb",
                                    "video_id": video_id,
                                    "tubelet_dir": tubelet_dir,
                                    "start_frame": start,
                                    "end_frame": start + 16,
                                    "label_id": tubelet_label,
                                    "label_name": LABEL_NAMES[tubelet_label],
                                    "split": split,
                                }
                            )
                            tubelet_idx += 1
                        else:
                            import shutil
                            shutil.rmtree(out_dir, ignore_errors=True)

                start += 8

        processed += 1
        if processed % 50 == 0:
            print(f"[CVB] processed {processed} videos, {len(labels_rows)} tubelets so far")


def export_cbvd5_tubelets(
    output_root: str,
    labels_rows: list,
    max_videos: int | None = None,
) -> None:
    """Export CBVD-5 tubelets, appending label dicts to labels_rows.

    Args:
        output_root: root directory for tubelets (e.g. data/processed/tubelets)
        labels_rows: list to append label row dicts into
        max_videos: if set, stop after this many videos (for testing)
    """
    import hashlib
    from collections import defaultdict

    # Step 1 — load all annotations grouped by video_id
    annotations = load_cbvd5_annotations()
    by_video: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        by_video[ann["video_id"]].append(ann)

    processed = 0
    for video_id in sorted(by_video):
        if max_videos is not None and processed >= max_videos:
            break

        video_path = os.path.join(_CBVD5_VIDEOS, f"{video_id}.mp4")
        if not os.path.isfile(video_path):
            continue

        tracking_path = os.path.join(_CBVD5_TRACKING, f"{video_id}_tracks.json")
        if not os.path.isfile(tracking_path):
            continue

        with open(tracking_path) as f:
            track_data = json.load(f)
        predicted_by_ts: dict[str, list[dict]] = track_data.get("frames", {})

        for ann in by_video[video_id]:
            timestamp = ann["timestamp"]
            frame_center = ann["frame_center"]
            # Step 2 — compute window, clamp, require exactly 16
            start = max(0, frame_center - 8)
            end = min(250, start + 16)
            if end - start != 16:
                continue

            bbox_norm = ann["bbox_norm"]
            label_id = ann["label_id"]
            split = ann["split"]

            # Step 3 — label already resolved by priority rule in load_cbvd5_annotations

            # Step 4 — convert GT bbox to pixels, match to predicted
            x1_gt = bbox_norm[0] * 1920
            y1_gt = bbox_norm[1] * 1080
            x2_gt = bbox_norm[2] * 1920
            y2_gt = bbox_norm[3] * 1080
            gt_bbox_px = [x1_gt, y1_gt, x2_gt, y2_gt]

            crop_bbox = gt_bbox_px  # fallback if no match
            ts_key = str(int(timestamp))
            predicted_dets = predicted_by_ts.get(ts_key, [])
            if predicted_dets:
                pred_bboxes = [det["bbox"] for det in predicted_dets]
                matches = match_predicted_to_gt(pred_bboxes, [gt_bbox_px])
                if matches:
                    best_pred_idx = next(iter(matches))
                    crop_bbox = predicted_dets[best_pred_idx]["bbox"]

            # Deterministic 6-char hex from GT bbox coords
            key = f"{round(bbox_norm[0], 3):.3f}_{round(bbox_norm[1], 3):.3f}"
            bbox_hash = hashlib.md5(key.encode()).hexdigest()[:6]

            out_dir = os.path.join(
                output_root, "cbvd5", video_id,
                f"kf{int(timestamp)}_inst{bbox_hash}",
            )
            os.makedirs(out_dir, exist_ok=True)

            # Step 5 — extract 16 frames from raw video and save crops
            try:
                frames = extract_cbvd5_frames(video_path, start)
            except ValueError:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
                continue

            crops = [_crop_frame(f, crop_bbox) for f in frames]
            if any(c is None or c.size == 0 for c in crops):
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
                continue
            for i, crop in enumerate(crops):
                cv2.imwrite(
                    os.path.join(out_dir, f"frame_{i:02d}.jpg"),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            # Step 6 — append labels row
            tubelet_dir = os.path.relpath(out_dir, _ONE_DAY)
            labels_rows.append(
                {
                    "dataset": "cbvd5",
                    "video_id": video_id,
                    "tubelet_dir": tubelet_dir,
                    "start_frame": start,
                    "end_frame": end,
                    "label_id": label_id,
                    "label_name": LABEL_NAMES[label_id],
                    "split": split,
                }
            )

        processed += 1
        if processed % 50 == 0:
            print(f"[CBVD-5] processed {processed} videos, {len(labels_rows)} tubelets so far")


def main() -> None:
    import argparse
    import csv

    from src.data.label_utils import load_cvb_splits

    global _CVB_TRACKING, _CBVD5_TRACKING

    parser = argparse.ArgumentParser(description="Export CVB and CBVD-5 tubelets to disk.")
    parser.add_argument("--output", required=True, help="Root output dir for tubelets")
    parser.add_argument("--cvb_tracking", default=None, help="Override CVB tracking dir")
    parser.add_argument("--cbvd5_tracking", default=None, help="Override CBVD-5 tracking dir")
    parser.add_argument("--cvb_only", action="store_true",
                        help="Re-export CVB only; merge with existing CBVD-5 rows from labels.csv")
    # Hidden args for testing partial runs
    parser.add_argument("--max_cvb_videos", type=int, default=None)
    parser.add_argument("--max_cbvd5_videos", type=int, default=None)
    args = parser.parse_args()

    if args.cvb_tracking:
        _CVB_TRACKING = args.cvb_tracking
    if args.cbvd5_tracking:
        _CBVD5_TRACKING = args.cbvd5_tracking

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    labels_rows: list[dict] = []

    if args.cvb_only:
        # Load existing CBVD-5 rows from labels.csv, re-export CVB only
        existing_csv = os.path.join(output_root, "labels.csv")
        if os.path.isfile(existing_csv):
            with open(existing_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["dataset"] == "cbvd5":
                        row["label_id"] = int(row["label_id"])
                        labels_rows.append(row)
            print(f"[CBVD-5] loaded {len(labels_rows)} existing rows from {existing_csv}")
        else:
            print("[WARN] --cvb_only set but no existing labels.csv found; exporting CVB fresh")
        split_map = load_cvb_splits()
        print(f"[CVB] loaded {len(split_map)} video splits")
        cvb_start = len(labels_rows)
        export_cvb_tubelets(output_root, labels_rows, split_map, max_videos=args.max_cvb_videos)
        print(f"[CVB] done — {len(labels_rows) - cvb_start} new CVB tubelets")
    else:
        # CVB export
        split_map = load_cvb_splits()
        print(f"[CVB] loaded {len(split_map)} video splits")
        export_cvb_tubelets(output_root, labels_rows, split_map, max_videos=args.max_cvb_videos)
        print(f"[CVB] done — {len(labels_rows)} tubelets total")

        # CBVD-5 export
        cbvd5_start = len(labels_rows)
        export_cbvd5_tubelets(output_root, labels_rows, max_videos=args.max_cbvd5_videos)
        print(f"[CBVD-5] done — {len(labels_rows) - cbvd5_start} tubelets added")

    # Write labels.csv
    csv_path = os.path.join(output_root, "labels.csv")
    fieldnames = [
        "dataset", "video_id", "tubelet_dir",
        "start_frame", "end_frame",
        "label_id", "label_name", "split",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels_rows)

    print(f"Wrote {len(labels_rows)} rows → {csv_path}")


if __name__ == "__main__":
    main()
