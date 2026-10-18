"""
Phase 4 — CVB Tracking Evaluation
Cattle Vision Framework — Texas State University

Evaluates OC-SORT tracking output against CVB ground truth animal IDs.
GT source: ava_train_set.csv + ava_val_set.csv (AVA format)
GT format: video_id, frame_id, x1n, y1n, x2n, y2n, behavior, animal_id
           where bboxes are normalized [0,1] and frame_ids are '01'..'15'

Metrics computed per video, then aggregated:
  - MOTA  (Multi-Object Tracking Accuracy)
  - MOTP  (Multi-Object Tracking Precision)
  - IDF1  (Identity F1 — primary metric)
  - ID Switches
  - Precision, Recall

Usage:
    python src/tracking/eval_tracking.py
    python src/tracking/eval_tracking.py --split val
    python src/tracking/eval_tracking.py --split all --iou_thresh 0.4
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GT loader
# ══════════════════════════════════════════════════════════════════════════════


def load_gt(csv_paths: list) -> dict:
    gt = defaultdict(lambda: defaultdict(list))
    already_seen = set()
    for csv_path in csv_paths:
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 8:
                    continue
                video_id = row[0]
                frame_id = int(row[1])
                x1n, y1n = float(row[2]), float(row[3])
                x2n, y2n = float(row[4]), float(row[5])
                animal_id = int(row[7])
                key = (video_id, frame_id, animal_id)
                if key in already_seen:
                    continue
                already_seen.add(key)
                gt[video_id][frame_id].append(
                    {
                        "animal_id": animal_id,
                        "bbox_norm": [x1n, y1n, x2n, y2n],
                    }
                )
    return gt


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Box IoU utility
# ══════════════════════════════════════════════════════════════════════════════


def box_iou(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU: returns (len_a x len_b) ndarray."""
    m = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            m[i, j] = box_iou(a, b)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Hungarian matching (simple greedy — sufficient for ≤20 objects)
# ══════════════════════════════════════════════════════════════════════════════


def greedy_match(iou_mat, thresh):
    """
    Greedy matching: repeatedly pick highest IoU pair above thresh.
    Returns list of (row_idx, col_idx) matched pairs.
    """
    matched = []
    mat = iou_mat.copy()
    while True:
        idx = np.argmax(mat)
        r, c = divmod(idx, mat.shape[1])
        if mat[r, c] < thresh:
            break
        matched.append((r, c))
        mat[r, :] = -1
        mat[:, c] = -1
    return matched


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Per-video evaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_video(video_id, track_data, gt_frames, iou_thresh=0.5):
    """
    Evaluate one video.

    Returns dict of per-video metrics.
    """
    frames_json = track_data["frames"]

    # Get image dimensions from mask RLE size (first available detection)
    img_h, img_w = 1080, 1920  # fallback
    for fid_str in sorted(frames_json.keys(), key=lambda x: int(x)):
        dets = frames_json[fid_str]
        if dets and dets[0].get("mask_rle"):
            img_h, img_w = dets[0]["mask_rle"]["size"]
            break

    # Only evaluate at annotated keyframes present in BOTH gt and tracking
    gt_frame_ids = sorted(gt_frames.keys())
    eval_frame_ids = [fid for fid in gt_frame_ids if str(fid) in frames_json]

    if len(eval_frame_ids) == 0:
        return None

    # ── Per-frame matching ────────────────────────────────────────────────────
    # Track identity consistency: pred_track_id -> gt_animal_id mapping
    # Built up across frames to detect ID switches
    track_to_gt = {}  # current assignment: pred_track_id -> gt_animal_id
    gt_to_track = {}  # inverse: gt_animal_id -> pred_track_id

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_sw = 0
    total_motp_sum = 0.0
    total_motp_n = 0

    # IDF1 accumulators: for each (gt_id, pred_id) pair count co-occurrences
    idtp = defaultdict(int)  # frames where gt_id matched pred_id
    idp = defaultdict(int)  # frames pred_id appeared
    idr = defaultdict(int)  # frames gt_id appeared

    for frame_id in eval_frame_ids:
        gt_dets = gt_frames[frame_id]
        pred_dets = frames_json.get(str(frame_id), [])

        # Denormalize GT bboxes to pixels
        gt_boxes = [
            [
                d["bbox_norm"][0] * img_w,
                d["bbox_norm"][1] * img_h,
                d["bbox_norm"][2] * img_w,
                d["bbox_norm"][3] * img_h,
            ]
            for d in gt_dets
        ]
        gt_ids = [d["animal_id"] for d in gt_dets]

        pred_boxes = [d["bbox"] for d in pred_dets]  # already xyxy pixels
        pred_ids = [d["track_id"] for d in pred_dets]

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt += n_gt
        total_pred += n_pred

        # Update IDF1 denominators
        for gid in gt_ids:
            idr[gid] += 1
        for pid in pred_ids:
            idp[pid] += 1

        if n_gt == 0 and n_pred == 0:
            continue

        if n_gt == 0:
            total_fp += n_pred
            continue

        if n_pred == 0:
            total_fn += n_gt
            continue

        # Compute IoU matrix and match
        iou_mat = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(iou_mat, iou_thresh)
        matched_gt = set()
        matched_pred = set()

        for gi, pi in matches:
            matched_gt.add(gi)
            matched_pred.add(pi)

            gt_id = gt_ids[gi]
            pred_id = pred_ids[pi]

            # ID switch detection
            prev_gt_for_pred = track_to_gt.get(pred_id)
            if prev_gt_for_pred is not None and prev_gt_for_pred != gt_id:
                total_id_sw += 1

            # Update identity map
            track_to_gt[pred_id] = gt_id
            gt_to_track[gt_id] = pred_id

            # IDF1 numerator
            idtp[(gt_id, pred_id)] += 1

            # MOTP accumulator
            total_motp_sum += iou_mat[gi, pi]
            total_motp_n += 1

            total_tp += 1

        total_fp += n_pred - len(matched_pred)
        total_fn += n_gt - len(matched_gt)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    mota_denom = total_gt if total_gt > 0 else 1
    mota = 1.0 - (total_fp + total_fn + total_id_sw) / mota_denom

    motp = total_motp_sum / total_motp_n if total_motp_n > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # IDF1 computation
    # IDTP for best assignment per gt_id
    total_idtp = 0
    total_idr_n = sum(idr.values())
    total_idp_n = sum(idp.values())

    # For each gt_id pick the pred_id that maximises idtp
    gt_id_best = defaultdict(int)
    for (gid, pid), cnt in idtp.items():
        if cnt > gt_id_best[gid]:
            gt_id_best[gid] = cnt
    total_idtp = sum(gt_id_best.values())

    idf1_denom = total_idr_n + total_idp_n
    idf1 = (2 * total_idtp / idf1_denom) if idf1_denom > 0 else 0.0

    return {
        "video_id": video_id,
        "eval_frames": len(eval_frame_ids),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "id_switches": total_id_sw,
        "mota": round(mota * 100, 2),
        "motp": round(motp * 100, 2),
        "idf1": round(idf1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Aggregate across videos
# ══════════════════════════════════════════════════════════════════════════════


def aggregate_metrics(results: list) -> dict:
    """
    Aggregate per-video results into dataset-level metrics.
    Uses accumulated TP/FP/FN/IDS for MOTA (not mean of per-video MOTA).
    """
    total_gt = sum(r["total_gt"] for r in results)
    total_pred = sum(r["total_pred"] for r in results)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_ids = sum(r["id_switches"] for r in results)

    mota_denom = total_gt if total_gt > 0 else 1
    mota = (1.0 - (total_fp + total_fn + total_ids) / mota_denom) * 100

    precision = (
        total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
    )
    recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0

    # Mean of per-video MOTP and IDF1 (standard practice)
    motp = np.mean([r["motp"] for r in results])
    idf1 = np.mean([r["idf1"] for r in results])

    return {
        "videos_evaluated": len(results),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_id_switches": total_ids,
        "MOTA": round(mota, 2),
        "MOTP": round(motp, 2),
        "IDF1": round(idf1, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "Avg_IDS_per_video": round(total_ids / len(results), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — CVB Tracking Evaluation")
    parser.add_argument(
        "--split",
        default="all",
        choices=["train", "val", "all"],
        help="Which GT split to evaluate against",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for TP matching (default 0.5)",
    )
    parser.add_argument(
        "--track_dir", default=None, help="Override tracking output directory"
    )
    parser.add_argument("--data_root", default=None, help="Override project root")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N videos (for testing)",
    )
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root = (
        Path(args.data_root) if args.data_root else Path(__file__).resolve().parents[2]
    )
    data_raw = project_root / "data" / "raw"
    track_dir = (
        Path(args.track_dir)
        if args.track_dir
        else project_root / "data" / "processed" / "tracking_v2" / "cvb"
    )
    results_dir = project_root / "results" / "tracking"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GT ────────────────────────────────────────────────────────────────
    csv_map = {
        "train": [data_raw / "cvb" / "cvb_in_ava_format" / "ava_train_set.csv"],
        "val": [data_raw / "cvb" / "cvb_in_ava_format" / "ava_val_set.csv"],
        "all": [
            data_raw / "cvb" / "cvb_in_ava_format" / "ava_train_set.csv",
            data_raw / "cvb" / "cvb_in_ava_format" / "ava_val_set.csv",
        ],
    }
    print(f"Loading GT from: {[str(p) for p in csv_map[args.split]]}")
    gt = load_gt(csv_map[args.split])
    print(f"GT loaded: {len(gt)} videos")

    # ── Find evaluatable videos ────────────────────────────────────────────────
    track_files = sorted(track_dir.glob("*_tracks.json"))
    if args.limit:
        track_files = track_files[: args.limit]

    evaluatable = [
        (tf, tf.stem.replace("_tracks", ""))
        for tf in track_files
        if tf.stem.replace("_tracks", "") in gt
    ]

    print(f"Track files found:   {len(track_files)}")
    print(f"Evaluatable videos:  {len(evaluatable)}")
    print(f"IoU threshold:       {args.iou_thresh}")
    print(f"{'='*60}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    per_video_results = []
    skipped = 0

    for i, (track_path, video_id) in enumerate(evaluatable):
        with open(track_path) as f:
            track_data = json.load(f)

        result = evaluate_video(
            video_id,
            track_data,
            gt[video_id],
            iou_thresh=args.iou_thresh,
        )

        if result is None:
            skipped += 1
            continue

        per_video_results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(evaluatable):
            running = aggregate_metrics(per_video_results)
            print(
                f"  [{i+1:4d}/{len(evaluatable)}]  "
                f"MOTA={running['MOTA']:6.2f}%  "
                f"IDF1={running['IDF1']:6.2f}%  "
                f"IDS={running['total_id_switches']}"
            )

    # ── Final results ──────────────────────────────────────────────────────────
    final = aggregate_metrics(per_video_results)

    print(f"\n{'='*60}")
    print(f"PHASE 4 TRACKING EVALUATION RESULTS")
    print(f"Dataset: CVB  |  Split: {args.split}  |  IoU thresh: {args.iou_thresh}")
    print(f"{'='*60}")
    print(f"  Videos evaluated : {final['videos_evaluated']}  (skipped: {skipped})")
    print(f"  Total GT dets    : {final['total_gt']}")
    print(f"  Total Pred dets  : {final['total_pred']}")
    print(
        f"  TP / FP / FN     : {final['total_tp']} / {final['total_fp']} / {final['total_fn']}"
    )
    print(f"{'─'*60}")
    print(f"  MOTA             : {final['MOTA']:6.2f}%")
    print(f"  MOTP             : {final['MOTP']:6.2f}%")
    print(f"  IDF1             : {final['IDF1']:6.2f}%")
    print(f"  Precision        : {final['Precision']:6.2f}%")
    print(f"  Recall           : {final['Recall']:6.2f}%")
    print(f"  Total ID Switches: {final['total_id_switches']}")
    print(f"  Avg IDS/video    : {final['Avg_IDS_per_video']}")
    print(f"{'='*60}")

    # ── Save results ───────────────────────────────────────────────────────────
    import csv as csv_mod

    # Per-video CSV
    per_video_path = results_dir / f"tracking_per_video_{args.split}.csv"
    fieldnames = [
        "video_id",
        "eval_frames",
        "total_gt",
        "total_pred",
        "tp",
        "fp",
        "fn",
        "id_switches",
        "mota",
        "motp",
        "idf1",
        "precision",
        "recall",
    ]
    with open(per_video_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_video_results)

    # Summary JSON
    summary_path = results_dir / f"tracking_summary_{args.split}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": {
                    "split": args.split,
                    "iou_thresh": args.iou_thresh,
                },
                "results": {
                    k: float(v) if hasattr(v, "item") else v for k, v in final.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\nPer-video results : {per_video_path}")
    print(f"Summary JSON      : {summary_path}")


if __name__ == "__main__":
    main()
