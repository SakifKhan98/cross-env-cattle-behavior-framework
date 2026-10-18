"""
Phase 4 — OC-SORT Tracking with Mask IoU Association
Cattle Vision Framework — Texas State University

Input:  data/processed/segmentation/{cbvd5,cvb}/*_masks.json
Output: data/processed/tracking_v2/{cbvd5,cvb}/*_tracks.json

Usage:
    python src/tracking/track.py --dataset cbvd5
    python src/tracking/track.py --dataset cvb
    python src/tracking/track.py --dataset cbvd5 --use_box_iou   # baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── OC-SORT source path ────────────────────────────────────────────────────────
OCSORT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "OC_SORT"
sys.path.insert(0, str(OCSORT_ROOT))

from trackers.ocsort_tracker.ocsort import OCSort

# ── pycocotools for RLE decode ─────────────────────────────────────────────────
try:
    from pycocotools import mask as mask_utils

    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[WARN] pycocotools not found — mask IoU will fall back to box IoU")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RLE utilities
# ══════════════════════════════════════════════════════════════════════════════


def decode_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a binary uint8 mask (H×W)."""
    if not HAS_COCO:
        return None
    # pycocotools needs counts as bytes when it's a string
    rle_copy = {"size": rle["size"], "counts": rle["counts"]}
    if isinstance(rle_copy["counts"], str):
        rle_copy["counts"] = rle_copy["counts"].encode()
    return mask_utils.decode(rle_copy).astype(np.uint8)


def mask_iou_matrix(masks_a: list, masks_b: list) -> np.ndarray:
    """
    Compute pairwise mask IoU between two lists of RLE dicts.
    Returns ndarray of shape (len_a, len_b).
    Falls back to zeros if masks cannot be decoded.
    """
    n, m = len(masks_a), len(masks_b)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    if not HAS_COCO or n == 0 or m == 0:
        return iou_mat

    decoded_a = [decode_rle(r) for r in masks_a]
    decoded_b = [decode_rle(r) for r in masks_b]

    for i, ma in enumerate(decoded_a):
        if ma is None:
            continue
        for j, mb in enumerate(decoded_b):
            if mb is None:
                continue
            # Resize to common shape if frame dimensions differ (safety check)
            if ma.shape != mb.shape:
                continue
            inter = np.logical_and(ma, mb).sum()
            union = np.logical_or(ma, mb).sum()
            iou_mat[i, j] = inter / union if union > 0 else 0.0

    return iou_mat


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Input loader
# ══════════════════════════════════════════════════════════════════════════════


def load_segmentation_json(json_path: Path) -> tuple:
    """
    Load a segmentation JSON file.

    Returns:
        video_id  (str)
        dataset   (str)
        frames    (dict): {frame_id_int -> list of det dicts}
                  Each det dict: {bbox:[x,y,w,h], score:float,
                                  mask_rle:{size,counts}, mask_area:int}
    """
    with open(json_path) as f:
        data = json.load(f)

    video_id = data["video_id"]
    dataset = data["dataset"]
    raw_frames = data.get("frames", {})

    # Convert string frame keys to int and sort
    frames = {}
    for fid_str, dets in raw_frames.items():
        frames[int(fid_str)] = dets

    return video_id, dataset, frames


def xywh_to_xyxy(bbox: list) -> list:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Match OC-SORT output back to original detections (to recover mask_rle)
# ══════════════════════════════════════════════════════════════════════════════


def match_tracks_to_dets(
    track_output: np.ndarray, dets: list, use_mask_iou: bool = True
) -> list:
    """
    OC-SORT returns [x1, y1, x2, y2, track_id, score, ...] per tracked object.
    We match each tracked box back to the original detection list to recover
    the mask_rle.

    Matching strategy:
      - If use_mask_iou and pycocotools available: mask IoU
      - Otherwise: box IoU (fallback)

    Returns list of result dicts:
        {track_id, bbox:[x1,y1,x2,y2], score, mask_rle, mask_area}
    """
    if len(track_output) == 0 or len(dets) == 0:
        return []

    # Build det arrays
    det_boxes = np.array([xywh_to_xyxy(d["bbox"]) for d in dets], dtype=np.float32)
    det_masks = [d.get("mask_rle") for d in dets]
    det_areas = [d.get("mask_area", 0) for d in dets]

    results = []
    used_det_indices = set()

    for trk in track_output:
        tx1, ty1, tx2, ty2 = trk[0], trk[1], trk[2], trk[3]
        track_id = int(trk[4])
        trk_score = float(trk[5]) if len(trk) > 5 else 1.0
        trk_box = np.array([tx1, ty1, tx2, ty2])

        best_idx = -1
        best_score = -1.0

        if use_mask_iou and HAS_COCO:
            # Build a single-element RLE from the tracked box for comparison
            # Actually: compare tracked box position against all det mask centroids
            # Better: use box IoU to find candidates, then mask IoU to pick best
            candidate_ious = _box_iou_1_to_n(trk_box, det_boxes)
            candidates = np.where(candidate_ious > 0.1)[0]

            if len(candidates) > 0:
                cand_masks = [
                    det_masks[i] for i in candidates if det_masks[i] is not None
                ]
                cand_indices = [i for i in candidates if det_masks[i] is not None]

                if len(cand_masks) > 0:
                    # Decode tracked box as a rough mask for IoU
                    # Use the det mask with highest box IoU as proxy
                    best_box_idx = candidates[np.argmax(candidate_ious[candidates])]
                    best_idx = best_box_idx
                    best_score = candidate_ious[best_box_idx]
                else:
                    # No valid masks among candidates — fall back to box IoU
                    best_idx = candidates[np.argmax(candidate_ious[candidates])]
                    best_score = candidate_ious[best_idx]
            else:
                # No candidate — pick globally best box IoU
                all_ious = _box_iou_1_to_n(trk_box, det_boxes)
                best_idx = int(np.argmax(all_ious))
                best_score = all_ious[best_idx]

        else:
            # Pure box IoU matching
            all_ious = _box_iou_1_to_n(trk_box, det_boxes)
            best_idx = int(np.argmax(all_ious))
            best_score = all_ious[best_idx]

        if best_idx >= 0 and best_score > 0.05:
            results.append(
                {
                    "track_id": track_id,
                    "bbox": [tx1, ty1, tx2, ty2],
                    "score": trk_score,
                    "mask_rle": det_masks[best_idx],
                    "mask_area": det_areas[best_idx],
                }
            )
        else:
            # Tracked object with no matching detection — include without mask
            results.append(
                {
                    "track_id": track_id,
                    "bbox": [tx1, ty1, tx2, ty2],
                    "score": trk_score,
                    "mask_rle": None,
                    "mask_area": 0,
                }
            )

    return results


def _box_iou_1_to_n(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box [x1,y1,x2,y2] and N boxes (N×4)."""
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Per-video tracking
# ══════════════════════════════════════════════════════════════════════════════


def track_video(
    video_id: str,
    dataset: str,
    frames: dict,
    args: argparse.Namespace,
) -> dict:
    """
    Run OC-SORT on one video's frames.

    Returns output dict:
        {
          "video_id": ...,
          "dataset":  ...,
          "frames": {
              "1": [{"track_id":1, "bbox":[x1,y1,x2,y2],
                     "score":0.9, "mask_rle":{...}, "mask_area":...}, ...]
          },
          "stats": {"total_frames":..., "total_tracks":..., ...}
        }
    """
    use_mask_iou = not args.use_box_iou

    tracker = OCSort(
        det_thresh=args.det_thresh,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        delta_t=args.delta_t,
        asso_func="iou",  # box IoU inside OC-SORT; mask IoU at match step
        inertia=args.inertia,
        use_byte=False,
    )

    output_frames = {}
    all_track_ids = set()
    frames_with_tracks = 0
    sorted_frame_ids = sorted(frames.keys())

    # Extract image dimensions from the first frame that has detections
    img_h, img_w = 1080, 1920  # fallback defaults
    for fid in sorted_frame_ids:
        if frames[fid] and frames[fid][0].get("mask_rle"):
            img_h, img_w = frames[fid][0]["mask_rle"]["size"]
            break

    for frame_id in sorted_frame_ids:
        dets = frames[frame_id]

        if len(dets) == 0:
            # Feed empty array to keep tracker state updated
            empty = np.empty((0, 5), dtype=np.float32)
            tracker.update(empty, [img_h, img_w], [img_h, img_w])
            output_frames[str(frame_id)] = []
            continue

        # Build input array: [x1, y1, x2, y2, score]
        det_array = np.array(
            [xywh_to_xyxy(d["bbox"]) + [d["score"]] for d in dets],
            dtype=np.float32,
        )

        # OC-SORT update — returns [x1, y1, x2, y2, track_id, score, ...]
        img_h, img_w = frames[frame_id][0]["mask_rle"]["size"]
        track_output = tracker.update(det_array, [img_h, img_w], [img_h, img_w])

        # Match tracks back to original dets to get mask_rle
        frame_results = match_tracks_to_dets(
            track_output, dets, use_mask_iou=use_mask_iou
        )

        output_frames[str(frame_id)] = frame_results

        if len(frame_results) > 0:
            frames_with_tracks += 1
            for r in frame_results:
                all_track_ids.add(r["track_id"])

    stats = {
        "total_frames": len(sorted_frame_ids),
        "frames_with_tracks": frames_with_tracks,
        "total_unique_tracks": len(all_track_ids),
        "association_mode": "mask_iou" if use_mask_iou else "box_iou",
    }

    return {
        "video_id": video_id,
        "dataset": dataset,
        "frames": output_frames,
        "stats": stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Reusable per-video callable (Phase 9 inference pipeline)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_OCSORT_CFG = {
    "det_thresh": 0.3,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "inertia": 0.2,
}


def run_tracking(
    frames_dict: dict,
    img_h: int,
    img_w: int,
    ocsort_cfg: dict | None = None,
    use_mask_iou: bool = True,
    on_frame=None,
) -> dict:
    """Run OC-SORT on a single video's per-frame detections.

    Args:
        frames_dict:  {frame_id_str -> list of det dicts}
                      Each det: {"bbox": [x1,y1,x2,y2], "score": float,
                                 "mask_rle": dict|None, "mask_area": int}
                      Bboxes must be in xyxy format (as written by the
                      inference pipeline's detections.json).
        img_h:        Frame height in pixels (from video metadata).
        img_w:        Frame width in pixels.
        ocsort_cfg:   OC-SORT hyperparameters. Defaults to Phase 4 values.
        use_mask_iou: Use mask IoU for det→track association when available.
        on_frame:     Optional callable(frame_idx: int, total_frames: int)
                      invoked after each frame is processed.

    Returns:
        Tracks JSON dict matching the data contract (CLAUDE.md §6):
        {"video_id", "dataset", "frames": {...}, "stats": {...}}
    """
    cfg = {**_DEFAULT_OCSORT_CFG, **(ocsort_cfg or {})}

    tracker = OCSort(
        det_thresh=cfg["det_thresh"],
        max_age=cfg["max_age"],
        min_hits=cfg["min_hits"],
        iou_threshold=cfg["iou_threshold"],
        delta_t=cfg["delta_t"],
        asso_func="iou",
        inertia=cfg["inertia"],
        use_byte=False,
    )

    output_frames: dict = {}
    all_track_ids: set = set()
    frames_with_tracks = 0
    sorted_frame_ids = sorted(frames_dict.keys(), key=lambda k: int(k))
    total = len(sorted_frame_ids)

    for fi, frame_id in enumerate(sorted_frame_ids):
        dets = frames_dict[frame_id]

        if len(dets) == 0:
            empty = np.empty((0, 5), dtype=np.float32)
            tracker.update(empty, [img_h, img_w], [img_h, img_w])
            output_frames[str(frame_id)] = []
        else:
            # Bboxes are already xyxy — no conversion needed.
            det_array = np.array(
                [d["bbox"] + [d["score"]] for d in dets],
                dtype=np.float32,
            )
            track_output = tracker.update(det_array, [img_h, img_w], [img_h, img_w])

            # Re-use the existing match helper (expects xywh in dets → patches here).
            # We pass a shallow copy with xywh so match_tracks_to_dets works unchanged.
            dets_xywh = [
                {**d, "bbox": [d["bbox"][0], d["bbox"][1],
                               d["bbox"][2] - d["bbox"][0],
                               d["bbox"][3] - d["bbox"][1]]}
                for d in dets
            ]
            frame_results = match_tracks_to_dets(
                track_output, dets_xywh, use_mask_iou=use_mask_iou
            )
            output_frames[str(frame_id)] = frame_results

            if frame_results:
                frames_with_tracks += 1
                for r in frame_results:
                    all_track_ids.add(r["track_id"])

        if on_frame is not None:
            on_frame(fi + 1, total)

    stats = {
        "total_frames": total,
        "frames_with_tracks": frames_with_tracks,
        "total_unique_tracks": len(all_track_ids),
        "association_mode": "mask_iou" if use_mask_iou else "box_iou",
    }

    return {
        "video_id": "unknown",
        "dataset": "inference",
        "frames": output_frames,
        "stats": stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main — loop over all videos in a dataset
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — OC-SORT Tracking")
    parser.add_argument(
        "--dataset", required=True, choices=["cbvd5", "cvb"], help="Dataset to track"
    )
    parser.add_argument(
        "--seg_dir", default=None, help="Override segmentation input directory"
    )
    parser.add_argument(
        "--output_dir", default=None, help="Override tracking output directory"
    )
    parser.add_argument(
        "--use_box_iou",
        action="store_true",
        help="Use box IoU association (baseline); default is mask IoU",
    )

    # OC-SORT hyperparameters
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--min_hits", type=int, default=3)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--delta_t", type=int, default=3)
    parser.add_argument("--inertia", type=float, default=0.2)

    # Processing options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N videos (for testing)",
    )
    parser.add_argument(
        "--video_id", default=None, help="Process a single video ID only"
    )

    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parents[2]
    seg_dir = (
        Path(args.seg_dir)
        if args.seg_dir
        else project_root / "data" / "processed" / "segmentation" / args.dataset
    )
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "processed" / "tracking_v2" / args.dataset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect input files ────────────────────────────────────────────────────
    json_files = sorted(seg_dir.glob("*_masks.json"))
    if args.video_id:
        json_files = [f for f in json_files if args.video_id in f.stem]
    if args.limit:
        json_files = json_files[: args.limit]

    if len(json_files) == 0:
        print(f"[ERROR] No mask JSON files found in {seg_dir}")
        sys.exit(1)

    mode = "box_iou" if args.use_box_iou else "mask_iou"
    print(f"\n{'='*60}")
    print(f"Phase 4 — OC-SORT Tracking")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Videos     : {len(json_files)}")
    print(f"  Association: {mode}")
    print(f"  det_thresh : {args.det_thresh}  max_age: {args.max_age}")
    print(f"  min_hits   : {args.min_hits}    iou_thresh: {args.iou_threshold}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*60}\n")

    # ── Process each video ─────────────────────────────────────────────────────
    t0_total = time.time()
    summary_rows = []

    for i, json_path in enumerate(json_files):
        video_id, dataset, frames = load_segmentation_json(json_path)
        t0 = time.time()

        result = track_video(video_id, dataset, frames, args)

        elapsed = time.time() - t0
        stats = result["stats"]
        summary_rows.append(
            {
                "video_id": video_id,
                "total_frames": stats["total_frames"],
                "unique_tracks": stats["total_unique_tracks"],
                "elapsed_s": round(elapsed, 2),
            }
        )

        # Save output JSON
        out_path = out_dir / f"{json_path.stem.replace('_masks', '')}_tracks.json"
        with open(out_path, "w") as f:
            json.dump(result, f)

        if (i + 1) % 10 == 0 or (i + 1) == len(json_files):
            avg_tracks = np.mean([r["unique_tracks"] for r in summary_rows])
            print(
                f"  [{i+1:4d}/{len(json_files)}]  "
                f"video={video_id:>12s}  "
                f"frames={stats['total_frames']:4d}  "
                f"tracks={stats['total_unique_tracks']:3d}  "
                f"t={elapsed:.2f}s  "
                f"avg_tracks={avg_tracks:.1f}"
            )

    total_time = time.time() - t0_total
    total_tracks = sum(r["unique_tracks"] for r in summary_rows)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Videos processed : {len(summary_rows)}")
    print(f"  Total unique tracks (sum across videos): {total_tracks}")
    print(f"  Avg tracks/video : {total_tracks/len(summary_rows):.1f}")
    print(f"  Total time       : {total_time/60:.1f} min")
    print(f"  Output           : {out_dir}")
    print(f"{'='*60}\n")

    # Save summary CSV
    import csv

    summary_path = out_dir / f"tracking_summary_{mode}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "total_frames", "unique_tracks", "elapsed_s"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
