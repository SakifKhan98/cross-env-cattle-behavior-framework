"""
src/segmentation/segment.py
Phase 3 — SAM2 Segmentation

PURPOSE:
    Takes RF-DETR bounding box detections (from Phase 2) and runs SAM2 to produce
    pixel-level segmentation masks for every detected cow in every frame.

WHY MASKS:
    - Bounding boxes lose shape info (lying vs standing look very different in mask form)
    - SAM2 video propagation: prompt once, propagate across many frames cheaply
    - Masks give OC-SORT better IoU cost functions for occlusion handling in Phase 4

HOW IT HANDLES THE TWO DATASETS DIFFERENTLY:

    CBVD-5 (sparse keyframes):
        Only 6 keyframes per clip, seconds apart in real time.
        There is no temporal continuity between them — SAM2 propagation would
        be meaningless. Each keyframe is prompted independently using RF-DETR boxes.

    CVB (dense 450-frame clips):
        450 frames of continuous video per clip.
        We use SAM2 video propagation with K=15 re-prompting:
            Frame 0:     RF-DETR boxes → SAM2 prompt → masks
            Frames 1-14: SAM2 propagates masks using previous mask as hint
            Frame 15:    RF-DETR boxes → SAM2 re-prompt → reset
            Frame 16-29: propagate again ...
        K=15 prevents mask drift that accumulates over 450 frames.

INPUT:
    data/processed/tracking/{dataset}/{video_id}.json
    (Detection JSONs from Phase 2 — one per video)

OUTPUT (masks):
    data/processed/segmentation/{dataset}/{video_id}_masks.json

OUTPUT (thesis logging):
    results/segmentation/{dataset}_segmentation_stats.csv  ← per-video stats
    results/segmentation/{dataset}_summary.json            ← aggregate summary
    results/segmentation/viz/{dataset}/                    ← sample overlay images

USAGE:
    # Sanity check (first 3 videos only):
    python src/segmentation/segment.py --config configs/segmentation/sam2.yaml --sanity

    # Full run on CBVD-5:
    python src/segmentation/segment.py --config configs/segmentation/sam2.yaml --dataset cbvd5

    # Full run on CVB:
    python src/segmentation/segment.py --config configs/segmentation/sam2.yaml --dataset cvb

    # Both datasets:
    python src/segmentation/segment.py --config configs/segmentation/sam2.yaml --dataset both

    # Debug a single video:
    python src/segmentation/segment.py --config configs/segmentation/sam2.yaml --video_id 618
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# sys.path trick so Python can find the segmentation package when running
# from the project root (e.g. python src/segmentation/segment.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmentation.mask_utils import mask_to_rle, mask_to_bbox, mask_area


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: SAM2 Segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation/sam2.yaml",
        help="Path to the SAM2 segmentation config YAML",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cbvd5", "cvb", "both"],
        default="both",
        help="Which dataset to segment. 'both' runs cbvd5 then cvb.",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Sanity mode: only process the first 3 videos, then exit.",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="Process only this specific video ID (useful for debugging).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config and return as a Python dict."""
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[config] Loaded: {config_path}")
    return cfg


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def preflight_checks(cfg: dict, dataset: str):
    """
    Verify all required files and directories exist before allocating GPU memory.
    Fail loudly with a clear fix message if anything is missing.
    """
    errors = []

    checkpoint = cfg["sam2"]["checkpoint"]
    if not os.path.isfile(checkpoint):
        errors.append(
            f"SAM2 checkpoint not found: {checkpoint}\n"
            f"    Fix:\n"
            f"      mkdir -p models/sam2\n"
            f"      wget https://dl.fbaipublicfiles.com/segment_anything_2/"
            f"092824/sam2.1_hiera_large.pt -O {checkpoint}"
        )

    det_dir = Path(cfg["paths"]["detection_dir"]) / dataset
    if not det_dir.is_dir():
        errors.append(f"Detection directory not found: {det_dir}")
    else:
        det_files = list(det_dir.glob("*.json"))
        if len(det_files) == 0:
            errors.append(f"No detection JSON files found in: {det_dir}")
        else:
            print(f"[preflight] Found {len(det_files)} detection JSONs in {det_dir}")

    img_key = "cbvd5_frames" if dataset == "cbvd5" else "cvb_frames"
    img_root = cfg["paths"][img_key]
    if not os.path.isdir(img_root):
        errors.append(f"Image root directory not found: {img_root}")

    if errors:
        print("\n[PREFLIGHT FAILED] Issues found:\n")
        for e in errors:
            print(f"  ✗ {e}\n")
        sys.exit(1)

    print(f"[preflight] All checks passed for dataset='{dataset}'")


# ---------------------------------------------------------------------------
# SAM2 model loader
# ---------------------------------------------------------------------------


def load_sam2_predictor(cfg: dict, device: torch.device):
    """
    Load SAM2 image predictor.

    WHY image predictor (not SAM2's built-in video predictor):
        SAM2's video predictor tries to load all frames into GPU memory at once.
        CVB clips have 450 frames — that would exceed 12GB VRAM on the RTX 3060.
        Instead, we use the image predictor and manually implement propagation
        by feeding the previous frame's mask as a 'mask_input' prompt each frame.
        This gives us full control and stays within the VRAM budget.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("[ERROR] SAM2 is not installed.")
        print("  Fix: pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'")
        sys.exit(1)

    checkpoint = cfg["sam2"]["checkpoint"]
    model_cfg = cfg["sam2"]["model_config"]

    # Locate SAM2's Hydra config directory from the installed package — no local
    # clone or hardcoded path needed. Works on any machine after pip install.
    import sam2 as _sam2_pkg
    configs_dir = os.path.join(os.path.dirname(_sam2_pkg.__file__), "configs")

    # SAM2 uses Hydra for its internal config system. When running from outside
    # the sam2 source directory, Hydra can't find its config files automatically.
    # We must initialize Hydra with the full path to SAM2's configs folder.
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=configs_dir, job_name="sam2", version_base="1.1")

    print(f"[sam2] Model config : {model_cfg}")
    print(f"[sam2] Checkpoint   : {checkpoint}")
    print(f"[sam2] Loading... (this takes ~20 seconds first time)")

    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print(f"[sam2] Loaded successfully on {device}")
    return predictor


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image_rgb(path: str) -> np.ndarray:
    """
    Load an image as RGB numpy array (H, W, 3) uint8.

    WHY convert to RGB:
        OpenCV loads images as BGR by default, but SAM2 expects RGB.
        Feeding BGR to SAM2 produces incorrect masks without any warning.
    """
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_frame_path(cfg: dict, dataset: str, video_id: str, frame_idx: int) -> str:
    """
    Build the full file path for a given frame.

    CBVD-5: {root}/{video_id}_{timestamp:05d}.jpg  e.g. "618_00002.jpg"
    CVB:    {root}/{clip_id}/img_{frame:05d}.jpg   e.g. "clip_001/img_00045.jpg"
    """
    if dataset == "cbvd5":
        root = cfg["paths"]["cbvd5_frames"]
        return os.path.join(root, f"{video_id}_{int(frame_idx):05d}.jpg")
    else:
        root = cfg["paths"]["cvb_frames"]
        return os.path.join(root, video_id, f"img_{int(frame_idx):05d}.jpg")


# ---------------------------------------------------------------------------
# Visualization — save mask overlay images for thesis figures
# ---------------------------------------------------------------------------


def save_viz(image_rgb: np.ndarray, results: list, out_path: str):
    """
    Save a visualization image showing RF-DETR boxes and SAM2 masks overlaid
    on the original frame. Used to generate thesis figures.

    Each cow gets a different color. The mask is shown as a semi-transparent
    colored overlay; the bounding box is drawn on top.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image
        results:   list of mask result dicts from segment_single_image()
        out_path:  file path to save the PNG
    """
    import cv2
    from segmentation.mask_utils import rle_to_mask

    # Work in BGR for OpenCV drawing
    viz = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    # Color palette — one color per detection (cycles if >10 cows)
    colors = [
        (255, 50, 50),  # red
        (50, 255, 50),  # green
        (50, 50, 255),  # blue
        (255, 255, 50),  # yellow
        (255, 50, 255),  # magenta
        (50, 255, 255),  # cyan
        (255, 165, 0),  # orange
        (128, 0, 255),  # purple
        (0, 255, 128),  # mint
        (255, 128, 0),  # amber
    ]

    for i, res in enumerate(results):
        color = colors[i % len(colors)]

        # Draw mask overlay (semi-transparent fill)
        try:
            mask = rle_to_mask(res["mask_rle"]).astype(bool)
            overlay = viz.copy()
            overlay[mask] = color
            viz = cv2.addWeighted(overlay, 0.35, viz, 0.65, 0)
        except Exception:
            pass  # Don't crash visualization if mask decode fails

        # Draw bounding box
        x, y, w, h = [int(v) for v in res["bbox"]]
        cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)

        # Label: score + mask area
        label = f"cow {i+1} | {res['score']:.2f} | {res['mask_area']}px"
        cv2.putText(
            viz,
            label,
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, viz)


# ---------------------------------------------------------------------------
# Core: segment a single image (CBVD-5 keyframes)
# ---------------------------------------------------------------------------


def segment_single_image(
    predictor, image_rgb: np.ndarray, detections: list, score_threshold: float
) -> list:
    """
    Run SAM2 on one image, using RF-DETR boxes as prompts.

    Args:
        predictor:       SAM2ImagePredictor
        image_rgb:       (H, W, 3) uint8 RGB image
        detections:      list of {"bbox": [x,y,w,h], "score": float}
        score_threshold: min score to use as SAM2 prompt (0.3 in config)

    Returns:
        list of dicts: [{"bbox":..., "score":..., "mask_rle":..., "mask_area":...}]
    """
    results = []

    valid_dets = [d for d in detections if d["score"] >= score_threshold]
    if not valid_dets:
        return results

    # set_image computes image embeddings — done once, reused for all boxes
    predictor.set_image(image_rgb)

    for det in valid_dets:
        x, y, w, h = det["bbox"]
        # Convert COCO [x, y, w, h] → SAM2 [x1, y1, x2, y2]
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            box=box_xyxy[None], multimask_output=False  # shape (1, 4)
        )

        # SAM2 returns shape (1, H, W) when multimask_output=False
        best_mask = masks[0].astype(bool)

        results.append(
            {
                "bbox": det["bbox"],
                "score": det["score"],
                "mask_rle": mask_to_rle(best_mask),
                "mask_area": mask_area(best_mask),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Core: segment a video clip with re-prompting (CVB dense clips)
# ---------------------------------------------------------------------------


def segment_video_clip(
    predictor,
    cfg: dict,
    dataset: str,
    video_id: str,
    detection_data: dict,
    score_threshold: float,
    reprompt_interval: int,
) -> dict:
    """
    Run SAM2 across all frames of a CVB clip with K=15 re-prompting.

    PROPAGATION STRATEGY:
        PROMPT frames (i % K == 0):
            Use RF-DETR boxes directly as SAM2 box prompts. Fresh start.
            Save the low-res logits SAM2 returns (shape 1x256x256) for use
            as mask_input on the next frame.

        PROPAGATION frames (between prompt frames):
            Pass the previous frame's low-res logits back as mask_input.
            This is the correct SAM2 API — mask_input must be the low-res
            logits (1x256x256 float32), NOT a full-resolution binary mask.
            Passing a full-res mask causes the tensor size mismatch error:
            "size of tensor a (64) must match tensor b (480)"

    Returns:
        dict: {frame_key (str) → list of result dicts}
    """
    frames_det = detection_data.get("frames", {})
    if not frames_det:
        return {}

    frame_keys_sorted = sorted(frames_det.keys(), key=lambda k: int(k))
    frame_results = {}

    # prev_binary: obj_idx → binary mask (H, W) bool — used for bbox extraction
    # prev_low_res: obj_idx → low-res logits (1, 256, 256) float32 — used as mask_input
    prev_binary = {}
    prev_low_res = {}

    for i, frame_key in enumerate(frame_keys_sorted):
        frame_idx = int(frame_key)
        detections = frames_det[frame_key]

        img_path = get_frame_path(cfg, dataset, video_id, frame_idx)
        if not os.path.isfile(img_path):
            frame_results[frame_key] = []
            continue

        try:
            image_rgb = load_image_rgb(img_path)
        except FileNotFoundError:
            frame_results[frame_key] = []
            continue

        predictor.set_image(image_rgb)

        is_prompt_frame = i % reprompt_interval == 0
        valid_dets = [d for d in detections if d["score"] >= score_threshold]
        frame_out = []

        if is_prompt_frame or not prev_binary:
            # ----------------------------------------------------------------
            # PROMPT FRAME — fresh RF-DETR box prompts
            # ----------------------------------------------------------------
            if not valid_dets:
                prev_binary = {}
                prev_low_res = {}
                frame_results[frame_key] = []
                continue

            new_binary = {}
            new_low_res = {}

            for obj_idx, det in enumerate(valid_dets):
                x, y, w, h = det["bbox"]
                box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

                # Third return value is low-res logits (1, 256, 256) float32
                masks, scores, low_res = predictor.predict(
                    box=box_xyxy[None], multimask_output=False
                )
                best_mask = masks[0].astype(bool)

                # Save both forms for the next frame
                new_binary[obj_idx] = best_mask
                new_low_res[obj_idx] = low_res[0]  # shape (1, 256, 256)

                frame_out.append(
                    {
                        "bbox": det["bbox"],
                        "score": float(det["score"]),
                        "mask_rle": mask_to_rle(best_mask),
                        "mask_area": mask_area(best_mask),
                    }
                )

            prev_binary = new_binary
            prev_low_res = new_low_res

        else:
            # ----------------------------------------------------------------
            # PROPAGATION FRAME — pass previous low-res logits as mask_input
            # ----------------------------------------------------------------
            new_binary = {}
            new_low_res = {}

            for obj_idx in prev_binary:
                # Use previous binary mask to get a location hint (bounding box)
                prop_box = mask_to_bbox(prev_binary[obj_idx])
                if prop_box is None:
                    continue  # object lost — skip

                px, py, pw, ph = prop_box
                box_xyxy = np.array([px, py, px + pw, py + ph], dtype=np.float32)

                # Pass the low-res logits from the previous frame as mask_input.
                # SAM2 expects shape (1, 1, 256, 256) — add the batch dim with [None].
                # This is the CORRECT way to propagate — not full-res binary masks.
                mask_input = prev_low_res[obj_idx][None]  # (1, 1, 256, 256)

                masks, scores, low_res = predictor.predict(
                    box=box_xyxy[None], mask_input=mask_input, multimask_output=False
                )
                best_mask = masks[0].astype(bool)

                new_binary[obj_idx] = best_mask
                new_low_res[obj_idx] = low_res[0]

                new_box = mask_to_bbox(best_mask)
                if new_box is None:
                    continue

                frame_out.append(
                    {
                        "bbox": new_box,
                        "score": float(scores[0]),
                        "mask_rle": mask_to_rle(best_mask),
                        "mask_area": mask_area(best_mask),
                    }
                )

            prev_binary = new_binary
            prev_low_res = new_low_res

        frame_results[frame_key] = frame_out

    return frame_results


# ---------------------------------------------------------------------------
# Process one video — returns stats dict for thesis logging
# ---------------------------------------------------------------------------


def process_video(
    predictor,
    cfg: dict,
    det_json_path: Path,
    output_dir: Path,
    dataset: str,
    viz_dir: Path,
    n_viz_saved: int,
    max_viz: int,
) -> dict:
    """
    Load one detection JSON, run segmentation, write mask JSON.

    Also collects per-video statistics for thesis reporting:
        - n_frames_with_detections
        - n_frames_with_masks
        - total_masks
        - avg_masks_per_frame
        - avg_mask_area_px
        - coverage_rate: masks / detections (how well SAM2 covered the detections)
        - processing_time_s

    Returns a stats dict (empty dict on skip, None on error).
    """
    with open(det_json_path, "r") as f:
        det_data = json.load(f)

    video_id = det_data["video_id"]
    out_path = output_dir / f"{video_id}_masks.json"

    # Skip-if-exists — safe to re-run after interruption
    if out_path.exists():
        print(f"  [skip] {video_id} — already exists")
        return {}

    score_threshold = cfg["sam2"]["score_threshold"]
    reprompt_interval = cfg["sam2"]["reprompt_interval"]
    t_start = time.time()

    try:
        if dataset == "cbvd5":
            frame_results = {}
            for frame_key, detections in det_data.get("frames", {}).items():
                frame_idx = int(frame_key)
                img_path = get_frame_path(cfg, dataset, video_id, frame_idx)
                if not os.path.isfile(img_path):
                    frame_results[frame_key] = []
                    continue
                image_rgb = load_image_rgb(img_path)
                results = segment_single_image(
                    predictor, image_rgb, detections, score_threshold
                )
                frame_results[frame_key] = results
        else:
            frame_results = segment_video_clip(
                predictor,
                cfg,
                dataset,
                video_id,
                det_data,
                score_threshold,
                reprompt_interval,
            )

    except Exception as e:
        print(f"  [ERROR] {video_id} — {e}")
        import traceback

        traceback.print_exc()
        return None

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Write mask JSON output
    # ------------------------------------------------------------------
    out = {"video_id": video_id, "dataset": dataset, "frames": frame_results}
    with open(out_path, "w") as f:
        json.dump(out, f)

    # ------------------------------------------------------------------
    # Compute per-video statistics for thesis logging
    # ------------------------------------------------------------------
    total_detections = sum(len(dets) for dets in det_data.get("frames", {}).values())
    total_masks = sum(len(v) for v in frame_results.values())
    frames_with_dets = sum(
        1 for dets in det_data.get("frames", {}).values() if len(dets) > 0
    )
    frames_with_masks = sum(1 for v in frame_results.values() if len(v) > 0)

    all_areas = [
        res["mask_area"]
        for masks_in_frame in frame_results.values()
        for res in masks_in_frame
    ]
    avg_area = float(np.mean(all_areas)) if all_areas else 0.0

    coverage = total_masks / total_detections if total_detections > 0 else 0.0

    stats = {
        "video_id": video_id,
        "dataset": dataset,
        "n_frames": len(frame_results),
        "n_frames_with_dets": frames_with_dets,
        "n_frames_with_masks": frames_with_masks,
        "total_detections": total_detections,
        "total_masks": total_masks,
        "avg_masks_per_frame": round(total_masks / max(len(frame_results), 1), 2),
        "avg_mask_area_px": round(avg_area, 1),
        "coverage_rate": round(coverage, 4),
        "processing_time_s": round(elapsed, 2),
    }

    print(
        f"  [done] {video_id} — {frames_with_masks} frames, "
        f"{total_masks} masks, area={avg_area:.0f}px, "
        f"coverage={coverage:.1%}, {elapsed:.1f}s"
    )

    # ------------------------------------------------------------------
    # Save visualization for thesis figures (first max_viz videos only)
    # ------------------------------------------------------------------
    if n_viz_saved < max_viz:
        # Pick the first frame that has masks
        for frame_key, results in frame_results.items():
            if len(results) > 0:
                frame_idx = int(frame_key)
                img_path = get_frame_path(cfg, dataset, video_id, frame_idx)
                if os.path.isfile(img_path):
                    try:
                        image_rgb = load_image_rgb(img_path)
                        viz_path = str(viz_dir / f"{video_id}_frame{frame_key}.jpg")
                        save_viz(image_rgb, results, viz_path)
                        print(f"  [viz] Saved: {viz_path}")
                    except Exception as e:
                        print(f"  [viz] Failed for {video_id}: {e}")
                break  # only save one frame per video

    return stats


# ---------------------------------------------------------------------------
# Save per-dataset thesis reports
# ---------------------------------------------------------------------------


def save_thesis_reports(
    all_stats: list,
    dataset: str,
    results_dir: Path,
    elapsed_total: float,
    n_ok: int,
    n_fail: int,
    cfg: dict,
):
    """
    Write two files for the thesis report:
        1. {dataset}_segmentation_stats.csv — one row per video
        2. {dataset}_summary.json           — aggregate statistics

    These provide the data needed for the Phase 3 thesis chapter discussion.
    """
    if not all_stats:
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Per-video CSV
    # ------------------------------------------------------------------
    csv_path = results_dir / f"{dataset}_segmentation_stats.csv"
    fieldnames = [
        "video_id",
        "dataset",
        "n_frames",
        "n_frames_with_dets",
        "n_frames_with_masks",
        "total_detections",
        "total_masks",
        "avg_masks_per_frame",
        "avg_mask_area_px",
        "coverage_rate",
        "processing_time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"[report] Per-video CSV saved: {csv_path}")

    # ------------------------------------------------------------------
    # 2. Aggregate summary JSON
    # ------------------------------------------------------------------
    total_masks_all = sum(s["total_masks"] for s in all_stats)
    total_dets_all = sum(s["total_detections"] for s in all_stats)
    all_areas = [s["avg_mask_area_px"] for s in all_stats if s["avg_mask_area_px"] > 0]
    all_coverage = [s["coverage_rate"] for s in all_stats]
    all_times = [s["processing_time_s"] for s in all_stats]
    zero_mask_videos = sum(1 for s in all_stats if s["total_masks"] == 0)
    zero_det_videos = sum(1 for s in all_stats if s["total_detections"] == 0)

    summary = {
        "dataset": dataset,
        "model": cfg["sam2"]["model_config"],
        "checkpoint": cfg["sam2"]["checkpoint"],
        "score_threshold": cfg["sam2"]["score_threshold"],
        "reprompt_interval_k": cfg["sam2"]["reprompt_interval"],
        # Coverage
        "n_videos_processed": len(all_stats),
        "n_videos_ok": n_ok,
        "n_videos_failed": n_fail,
        "n_videos_zero_detections": zero_det_videos,
        "n_videos_zero_masks": zero_mask_videos,
        # Mask counts
        "total_masks_produced": total_masks_all,
        "total_detections_input": total_dets_all,
        "overall_coverage_rate": round(total_masks_all / max(total_dets_all, 1), 4),
        # Mask quality
        "mean_mask_area_px": round(float(np.mean(all_areas)), 1) if all_areas else 0,
        "median_mask_area_px": (
            round(float(np.median(all_areas)), 1) if all_areas else 0
        ),
        "std_mask_area_px": round(float(np.std(all_areas)), 1) if all_areas else 0,
        "mean_coverage_per_video": round(float(np.mean(all_coverage)), 4),
        # Timing
        "total_runtime_min": round(elapsed_total / 60, 2),
        "mean_time_per_video_s": round(float(np.mean(all_times)), 2),
        "throughput_videos_per_min": round(
            len(all_stats) / max(elapsed_total / 60, 0.01), 1
        ),
    }

    json_path = results_dir / f"{dataset}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[report] Summary JSON saved: {json_path}")

    # ------------------------------------------------------------------
    # Print a human-readable summary to the terminal
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[report] Phase 3 Summary — {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Videos processed   : {len(all_stats)} ({n_fail} failed)")
    print(f"  Zero-detection vids: {zero_det_videos}  (RF-DETR found nothing)")
    print(f"  Zero-mask videos   : {zero_mask_videos}  (SAM2 produced nothing)")
    print(f"  Total masks        : {total_masks_all:,}")
    print(
        f"  Overall coverage   : {summary['overall_coverage_rate']:.1%}  (masks/detections)"
    )
    print(f"  Mean mask area     : {summary['mean_mask_area_px']:,.0f} px")
    print(f"  Median mask area   : {summary['median_mask_area_px']:,.0f} px")
    print(f"  Std mask area      : {summary['std_mask_area_px']:,.0f} px")
    print(f"  Mean coverage/vid  : {summary['mean_coverage_per_video']:.1%}")
    print(f"  Total runtime      : {summary['total_runtime_min']:.1f} min")
    print(
        f"  Throughput         : {summary['throughput_videos_per_min']:.1f} videos/min"
    )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    datasets = ["cbvd5", "cvb"] if args.dataset == "both" else [args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    if device.type == "cuda":
        print(f"[device] GPU  : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] VRAM : {vram:.1f} GB")

    for dataset in datasets:
        preflight_checks(cfg, dataset)

    predictor = load_sam2_predictor(cfg, device)

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"[segment] Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        det_dir = Path(cfg["paths"]["detection_dir"]) / dataset
        out_dir = Path(cfg["paths"]["segmentation_dir"]) / dataset
        results_dir = Path("results/segmentation")
        viz_dir = results_dir / "viz" / dataset

        out_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

        det_files = sorted(det_dir.glob("*.json"))

        if args.video_id:
            det_files = [f for f in det_files if f.stem == args.video_id]
            if not det_files:
                print(
                    f"[warn] No JSON found for video_id='{args.video_id}' in {det_dir}"
                )
                continue

        if args.sanity:
            det_files = det_files[:3]
            print(f"[sanity] Limiting to first 3 videos")

        print(f"[segment] Processing {len(det_files)} video(s)...")

        all_stats = []
        n_ok = 0
        n_fail = 0
        n_viz = 0
        max_viz = cfg.get("logging", {}).get("n_viz_samples", 20)
        t0 = time.time()

        for i, det_file in enumerate(det_files):
            print(f"\n[{i+1}/{len(det_files)}] {det_file.name}")
            stats = process_video(
                predictor, cfg, det_file, out_dir, dataset, viz_dir, n_viz, max_viz
            )
            if stats is None:
                n_fail += 1
            else:
                n_ok += 1
                if stats:  # non-empty means it wasn't skipped
                    all_stats.append(stats)
                    if stats.get("total_masks", 0) > 0:
                        n_viz += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0

        # Save thesis reports
        if not args.sanity:
            save_thesis_reports(
                all_stats, dataset, results_dir, elapsed, n_ok, n_fail, cfg
            )
        else:
            # Still print a mini-summary in sanity mode
            print(f"\n[sanity] {n_ok} OK, {n_fail} failed — " f"{elapsed:.1f}s total")
            if all_stats:
                total_m = sum(s["total_masks"] for s in all_stats)
                print(f"[sanity] Total masks produced: {total_m}")

    print("\n[segment] Phase 3 complete!")


if __name__ == "__main__":
    main()
