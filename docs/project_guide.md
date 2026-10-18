# Cattle Vision Framework — Project Guide for New Students

> **Purpose:** This document is the single source of truth for any new student picking up this project.
> It covers the full codebase end-to-end: what each folder contains, what each major file does,
> which pipeline stage produced which artifact, and how all the pieces connect.
>
> **How it was built:** Written collaboratively during a folder-by-folder walkthrough of the repo
> after all nine pipeline phases were complete. Update it whenever you add a new phase or
> restructure a folder.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder-by-Folder Reference](#2-folder-by-folder-reference)
   - [api/](#api)
   - [configs/](#configs)
   - [data/](#data)
   - [docker/](#docker)
   - [docs/](#docs)
   - [figures/](#figures) *(deleted — see notebooks/thesis_figures.ipynb)*
   - [logs/](#logs)
   - [models/](#models)
   - [notebooks/](#notebooks)
   - [results/](#results) *(pending)*
   - [scripts/](#scripts) *(pending)*
   - [src/](#src) *(pending)*
   - [tests/](#tests) *(pending)*
   - [ui/](#ui) *(pending)*
3. [End-to-End Pipeline Flow](#3-end-to-end-pipeline-flow)
   - [Training pipeline (offline)](#training-pipeline-offline)
   - [Inference pipeline (live video)](#inference-pipeline-live-video)
4. [Key Data Contracts](#4-key-data-contracts) *(pending)*
5. [Results Summary](#5-results-summary) *(pending)*

---

## 1. Project Overview

**Goal:** Recognize 7 dairy-cattle behaviors from surveillance video using a fully automated pipeline.

**Institution:** Texas State University — MS thesis by Sakif Khan (2026)

**Pipeline summary (v2 — primary thesis results):**

```
Raw video
  → RF-DETR-Seg  (detect cows + produce instance masks, single-stage model)
  → OC-SORT      (track cows across frames using mask IoU)
  → Tubelet cut  (extract 16-frame clips per track, 224×224 px)
  → VideoMAE     (classify each clip into one of 7 behavior classes)
  → Analytics    (activity budgets, behavior timelines, deviation detection)
  → Annotated video output
```

**7 behavior classes:**

| ID | Behavior | Datasets |
|----|----------|----------|
| 0 | Standing | CBVD-5, CVB |
| 1 | Lying | CBVD-5, CVB |
| 2 | Foraging/Grazing | CBVD-5, CVB |
| 3 | Drinking | CBVD-5, CVB |
| 4 | Ruminating | CBVD-5, CVB |
| 5 | Grooming | CVB only |
| 6 | Other | CVB only |

**Two parallel training paths exist:**
- **v1 path** — RF-DETR detection → SAM2 segmentation → OC-SORT (mask IoU) → VideoMAE
- **v2 path (primary)** — RF-DETR-Seg (single-stage detection + segmentation) → OC-SORT (mask IoU) → VideoMAE

All reported thesis results use the v2 path unless noted otherwise.

---

## 2. Folder-by-Folder Reference

---

### `api/`

**What it is:** The FastAPI backend for the Phase 9 inference web application.
Three files (plus `__pycache__`):

```
api/
├── __init__.py       # empty — makes api/ a Python package
├── job_store.py      # in-memory job registry
└── main.py           # FastAPI app — HTTP endpoints
```

#### `api/job_store.py`

An in-memory job registry backed by threading primitives.

- **`Job` dataclass** — holds `job_id`, `status` (`pending` → `running` → `complete`/`failed`),
  `last_event` (latest progress snapshot from the pipeline), `result_paths`, `error`,
  `video_filename`, `created_at`.
- **`JobStore` class** — wraps a dict of jobs behind a `threading.Lock` for thread safety.
  Each job also gets a `threading.Condition` so the SSE stream can block cheaply waiting for
  the next update rather than busy-polling.
- **`store`** — module-level singleton shared by all API requests.

> **Important constraint:** jobs are in-memory only and disappear on server restart.
> This was intentional for a thesis demo; a production system would use a database.

#### `api/main.py`

The FastAPI application. Six HTTP endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/jobs` | POST | Accept a video upload, save to temp dir, launch inference pipeline in a background thread, return `job_id` immediately |
| `/jobs/{id}/stream` | GET | Server-Sent Events — browser connects here and receives live progress events stage by stage |
| `/jobs/{id}/status` | GET | Poll fallback — same job state as SSE but one-shot, for clients that can't do SSE |
| `/jobs/{id}/results` | GET | Return a manifest of output file URLs once job is complete |
| `/jobs/{id}/analytics` | GET | Parse result CSVs, return structured JSON (activity budget, outliers, timelines) for React dashboard |
| `/jobs/seed` | POST | Register an existing result directory as a completed job — used to demo results without re-running |

**SPA serving:** The bottom of `main.py` mounts `ui/dist/` (the built React frontend) under
`/assets` and has a catch-all route that serves `index.html` for any unmatched GET path —
the standard single-page app pattern.

**How it connects to the rest of the repo:**
- Calls `src.inference.pipeline.run_pipeline()` in the background thread.
- Reads config from `configs/inference/default.yaml`.
- Writes per-job outputs to `results/inference/{job_id}/`.

---

## 3. End-to-End Pipeline Flow

### Training pipeline (offline)

*(Covered in detail under `scripts/` and `src/` — pending)*

### Inference pipeline (live video)

This is the Phase 9 real-time inference path. A single video enters through the web UI
and is processed by 7 sequential stages inside `src/inference/pipeline.py:run_pipeline()`.

#### Entry point: `POST /jobs` (`api/main.py:56`)

The browser uploads a video. The API:
1. Saves the video to a temp directory.
2. Loads `configs/inference/default.yaml` and overrides it with any user-supplied parameters
   (checkpoint paths, confidence threshold).
3. Spawns a **background thread** running `_run_pipeline_thread()`.
4. Returns `job_id` immediately — the browser never blocks waiting for the pipeline.

The browser then opens the SSE stream at `/jobs/{id}/stream` to receive live progress updates.

#### Inside `src/inference/pipeline.py:run_pipeline()`

After each stage, `_emit()` fires a progress-event dict:
```python
{
  "stage": int,          # 1-based
  "stage_name": str,
  "total_stages": 7,
  "frame": int,
  "total_frames": int,
  "status": "running" | "done" | "error"
}
```
This travels: `_emit()` → `progress callback` → `job_store.update_event()` → SSE stream → browser progress bar.

**Stage 1 — Ingest** (`src/inference/video_ingestor.py`)
Opens the video with OpenCV, reads `frame_count`, `fps`, `width`, `height`. No frames decoded
yet — just metadata. Nearly instant.

**Stage 2 — Detect + Segment** (`src/segmentation/rfdetr_seg_infer.py`)
Loads the RF-DETR-Seg checkpoint. Runs `predict_frame()` on every frame, returning bounding
boxes + binary instance masks for each detected cow. Results accumulated into a dict and
written to `output_root/detections.json`.
*Slowest stage — ~22 s for a 10-second clip on RTX 3060.*

**Stage 3 — Track** (`src/tracking/track.py:run_tracking()`)
Passes per-frame detections into OC-SORT using **mask IoU** for association (not box IoU —
this is the v2 improvement). Links detections across frames into persistent tracks.
Writes `tracks.json`.

**Stage 4 — Extract Tubelets** (`src/data/export_tubelets.py:export_tubelets_from_tracks()`)
For each track, cuts 16-frame clips (stride 4, 224×224 px) from the original video.
Same tubelet format used during training — critical for distribution match.
Writes frames to `output_root/tubelets/`.
Returns a list of `tubelet_rows` (path, start/end frame, track_id).
*Second-slowest — ~289 s for a 10-second clip because it re-reads the video for every tubelet.*

**Stage 5 — Classify** (`src/behavior/classify.py:classify_tubelets()`)
Loads the VideoMAE checkpoint. Runs inference on every tubelet, assigns a `pred_label_id`
(0–6) to each clip. Writes `predictions.csv`.

**Stage 6 — Analyze** (`src/analytics/timeline.py`, `src/analytics/budget.py`)
Two sub-steps run in sequence:
- `run_timeline_analysis()` — converts per-tubelet predictions into per-track Gantt-style
  timelines (behavior from second X to second Y). Writes per-track CSVs to `output_root/timelines/`.
- `run_budget_analysis()` — aggregates timelines into:
  - `activity_budget.csv` — % time per behavior per track
  - `behavior_deviation.csv` — which tracks deviate from the group median

These two CSVs are what `/jobs/{id}/analytics` parses and sends to the React dashboard.

**Stage 7 — Render** (`src/tracking/render_behavior_video.py:render_inference_video()`)
Reads the original video frame by frame, overlays bounding boxes and behavior labels
(color-coded by class), writes `annotated.mp4` using OpenCV's `VideoWriter`.

#### Output artifacts under `results/inference/{job_id}/`

| File | Created by stage |
|---|---|
| `detections.json` | 2 — Detect |
| `tracks.json` | 3 — Track |
| `tubelets/` | 4 — Extract |
| `predictions.csv` | 5 — Classify |
| `timelines/*.csv` | 6 — Analyze |
| `activity_budget.csv` | 6 — Analyze |
| `behavior_deviation.csv` | 6 — Analyze |
| `annotated.mp4` | 7 — Render |
| `timing.csv` | End of pipeline |

If `cleanup: true` is passed in the config, `detections.json`, `tracks.json`, and `tubelets/`
are deleted after completion — they are intermediate artifacts; only the final CSVs and
annotated video are kept.

---

---

### `configs/`

All pipeline hyperparameters live here — no hardcoded values in Python source. There are 5
subfolders and 19 YAML files in total. Three categories: **actually ran**, **never ran**, and
**empty/orphaned**.

#### `configs/detection/` — 3 files

**How consumed:** `scripts/05_train_detector.sh` sets `CONFIG=configs/detection/rfdetr_combined.yaml`
and passes it to `src/detection/train.py --config`. The Python script maps each YAML key to
RF-DETR API arguments.

| File | Ran? | Notes |
|---|---|---|
| `rfdetr_combined.yaml` | **Yes** | Produced `runs/detection/rfdetr_combined_v1/` — the detector used for all thesis results |

Only one detection config exists. Per-dataset configs (`rfdetr_cbvd5`, `rfdetr_cvb`) were
written as exploratory placeholders, never ran, and were deleted during cleanup. If a future
student wants to train per-dataset, use `rfdetr_combined.yaml` as the template and change
`dataset.name`, `dataset_dir`, and `output_dir`.

**Key RTX 3060 (12 GB) constraints baked into `rfdetr_combined.yaml`:**
- `batch_size: 2` + `grad_accum_steps: 8` → effective batch of 16
- `resolution: 576` — smallest value divisible by 64 above 560; saves VRAM vs 640
- `gradient_checkpointing: true` — saves ~30% VRAM at ~20% speed cost; required for 12 GB
- `augmentation.preset: AUG_CONSERVATIVE` — avoids aggressive augmentations on single-class detection

#### `configs/segmentation/` — 2 files

**How consumed:** Shell scripts pass `--config configs/segmentation/X.yaml`; the Python module
reads checkpoint paths, score thresholds, and frame-naming patterns.

| File | Ran? | Used by | Pipeline path |
|---|---|---|---|
| `sam2.yaml` | **Yes** | `07_run_segmentation.sh` → `src/segmentation/segment.py` | v1 path |
| `rfdetr_seg.yaml` | **Yes** | `07b_run_rfdetr_seg.sh` → `src/segmentation/rfdetr_seg_infer.py` | v2 path (primary) |

`sam2.yaml` specifies `reprompt_interval: 15` — SAM2 re-prompts every 15 frames rather than
propagating masks indefinitely (masks degrade over long occlusions).

`rfdetr_seg.yaml` points to `runs/seg_medium_lr5e5/checkpoint_best_ema.pth` — the checkpoint
trained on HiPE1. Both configs define frame-naming patterns for CBVD-5 and CVB separately
because the two datasets use different file naming conventions.

#### `configs/tracking/` — no files

OC-SORT hyperparameters live in two places:
- Batch pipeline (scripts 08/08b): hardcoded CLI args inside the shell scripts.
- Inference pipeline: embedded in `configs/inference/default.yaml` under the `tracking:` key.

An empty `ocsort.yaml` placeholder existed and was deleted during cleanup.

#### `configs/behavior/` — 12 files

**How consumed:** Scripts 10/10b iterate over all matching YAML files in `configs/behavior/`
and call `src/behavior/train.py --config <file>` for each one. Scripts 11/11b do the same for
evaluation. Each config fully describes one training experiment.

The 12 files form a 2×5+2 grid — 10 training configs (5 experiment types × v1/v2), plus 2
special-purpose configs:

| Config | Ran? | Tubelet source | Train set | Val set | Eval labels |
|---|---|---|---|---|---|
| `videomae_cbvd5.yaml` | **Yes** (v1) | `tubelets/` | CBVD-5 | CBVD-5 | all 7 |
| `videomae_cbvd5_v2.yaml` | **Yes** (v2) | `tubelets_rfdetr/` | CBVD-5 | CBVD-5 | all 7 |
| `videomae_cvb.yaml` | **Yes** (v1) | `tubelets/` | CVB | CVB | all 7 |
| `videomae_cvb_v2.yaml` | **Yes** (v2) | `tubelets_rfdetr/` | CVB | CVB | all 7 |
| `videomae_cbvd5_to_cvb.yaml` | **Yes** (v1) | `tubelets/` | CBVD-5 | CVB | 5-class |
| `videomae_cbvd5_to_cvb_v2.yaml` | **Yes** (v2) | `tubelets_rfdetr/` | CBVD-5 | CVB | 5-class |
| `videomae_cvb_to_cbvd5.yaml` | **Yes** (v1) | `tubelets/` | CVB | CBVD-5 | 5-class |
| `videomae_cvb_to_cbvd5_v2.yaml` | **Yes** (v2) | `tubelets_rfdetr/` | CVB | CBVD-5 | 5-class |
| `videomae_combined.yaml` | **Yes** (v1) | `tubelets/` | both | both | all 7 |
| `videomae_combined_v2.yaml` | **Yes** (v2) | `tubelets_rfdetr/` | both | both | all 7 |
| `videomae_mmcows.yaml` | Script exists, eval skipped | `tubelets/mmcows/` | — (eval only) | mmCows | 4-class |

**The only three differences between a v1 and v2 config:**
1. `labels_csv` — `tubelets/labels.csv` vs `tubelets_rfdetr/labels.csv`
2. `output_dir` — `..._v1` vs `..._v2` suffix
3. `experiment_name` — same suffix distinction

All hyperparameters (lr, batch size, epochs, model backbone, early stopping) are identical.
This was deliberate: v1 vs v2 is a controlled comparison where the only variable is the
tubelet source (SAM2-tracked vs RF-DETR-Seg-tracked).

**Cross-domain configs** (`cbvd5_to_cvb`, `cvb_to_cbvd5`) use `label_subset: [0, 1, 2, 3, 4]`
for validation — restricts evaluation to the 5 behaviors shared by both datasets, since
Grooming (5) and Other (6) are CVB-only.

`videomae_mmcows.yaml` evaluates only 4 classes and is the most heavily commented config; it
documents the temporal mismatch caveat (mmCows is ~0.067 fps; VideoMAE trained on 30 fps clips).

A dev-only smoke-test config (`videomae_sanity.yaml`, 2 epochs) was deleted during cleanup.

#### `configs/inference/` — 1 file

`default.yaml` — used by `api/main.py`, `src/inference/run_inference.py`, and scripts 24/25/26/31.
The only config that covers the full inference pipeline in one file: model checkpoints,
detection threshold, OC-SORT hyperparameters, tubelet parameters, and output settings.

The API overrides individual keys at runtime (e.g. swapping the checkpoint path).

> **Critical constraint:** `clip_len: 16`, `frame_stride: 4`, `input_size: 224` must match
> the values used when training VideoMAE. If they differ, inference will silently produce
> incorrect predictions with no error.

---

---

### `data/`

The `data/` directory is the **primary staging area for all pipeline inputs and outputs**.
Almost everything here is gitignored — the repo only commits the small file that cannot be
reconstructed from code. All large datasets and processed artifacts live locally (or on HiPE1)
and must be downloaded/generated before running the pipeline.

**What is committed:**
- `data/label_map.json` — committed but currently empty (0 bytes). The actual label mapping
  lives in `src/data/label_utils.py` as `LABEL_NAMES`, `_CVB_BEHAVIOR_MAP`, `_CBVD5_ACTION_MAP`,
  and `_FREEMAN_LABEL_MAP`. A future student can populate this JSON if they want a machine-readable
  label registry, but nothing in the pipeline reads it at runtime.

**What is gitignored (everything else):**

```
data/
├── label_map.json              # committed (but empty — see note above)
├── raw/                        # gitignored — source datasets, never modified by scripts
│   ├── cbvd5/                  # primary training: CBVD-5 (behavior labels, 25 fps videos)
│   ├── cvb/                    # primary training: CVB (behavior labels, 30 fps videos)
│   ├── cattle-eye-view/        # OOD Phase 8: CattleEyeView (top-down outdoor)
│   ├── cows2021/               # OOD Phase 8: Cows2021 (indoor barn)
│   ├── freeman-cmb-2024/       # OOD Phase 8: Freeman Center (angled ranch)
│   ├── mmCows/                 # OOD Phase 8: mmCows (NeurIPS 2024, Purdue/UW–Madison)
│   ├── opencow2020/            # OOD Phase 8: OpenCows2020 (aerial top-down)
│   └── kaggle-cow-segmentation-dataset/   # unused; ignore
├── processed/                  # gitignored — pipeline intermediates; grows ~100 GB
│   ├── detection/              # scripts 02–04 + OOD prep scripts: YOLO-format image + label dirs
│   │   ├── cbvd5/              #   → train/valid/test with symlinks into raw/cbvd5/
│   │   ├── cvb/                #   → train/valid/test with symlinks into raw/cvb/
│   │   ├── combined/           #   → merged CBVD-5 + CVB (used for detector training, script 05)
│   │   ├── combined_mini/      #   → small subset used for quick smoke tests
│   │   ├── cattleeyeview/      #   → OOD: CattleEyeView prepared by script 22-series
│   │   ├── cows2021/           #   → OOD
│   │   ├── freeman/            #   → OOD: Freeman Center
│   │   ├── mmcows/             #   → OOD: mmCows prepared by script 27
│   │   └── opencows2020/       #   → OOD
│   ├── segmentation/           # script 07 (v1 SAM2 path): per-frame mask arrays + overlays
│   │   ├── cbvd5/
│   │   └── cvb/
│   ├── segmentation_rfdetr/    # script 07b (v2 RF-DETR-Seg path): per-frame bbox + mask JSON
│   │   ├── cbvd5/
│   │   └── cvb/
│   ├── tracking/               # script 06: per-frame raw detection JSONs (no tracks yet)
│   │   ├── cbvd5/              #   format: {frame: [{bbox, score, mask_rle?}, ...]}
│   │   └── cvb/
│   ├── tracking_v2/            # script 08 (v1 SAM2 path): linked tracks with mask_rle
│   │   ├── cbvd5/              #   format: {video_id, frames: {frame: [{track_id, bbox, mask_rle}]}}
│   │   └── cvb/
│   ├── tracking_v2_rfdetr/     # script 08b (v2 RF-DETR-Seg path): linked tracks — PRIMARY THESIS
│   │   ├── cbvd5/              #   same format; association_mode = "mask_iou"
│   │   └── cvb/
│   ├── tracking_ablation/      # ablation runs varying OC-SORT hyperparameters
│   │   ├── cvb_mh1/
│   │   └── cvb_mh2/
│   ├── tubelets/               # script 09 (v1 SAM2 path): 16-frame clips per track
│   │   ├── cbvd5/              #   subdirs named by track_id; each frame is a 224×224 JPEG
│   │   ├── cvb/
│   │   ├── mmcows/             #   OOD mmCows tubelets (generated, eval skipped — 1 fps issue)
│   │   └── labels.csv          #   the index: dataset,video_id,tubelet_dir,start,end,label_id
│   └── tubelets_rfdetr/        # script 09b (v2 RF-DETR-Seg path): 16-frame clips — PRIMARY
│       ├── cbvd5/
│       ├── cvb/
│       └── labels.csv
├── rfdetr_seg/                 # gitignored — YOLO-seg format training data for RF-DETR-Seg
│   └── cattle/                 #   images + polygon segmentation labels (SAM2 pseudo-labels)
└── rfdetr_seg_eval/            # gitignored — RF-DETR-Seg evaluation data (Kaggle dataset)
    ├── images/
    ├── labels/
    └── overlays/               #   visual overlay verification of pseudo-labels
```

> **Known inconsistency — `rfdetr_seg/` and `rfdetr_seg_eval/` placement:**
> Both are derived artifacts and belong under `data/processed/` by the repo's own convention.
> `rfdetr_seg/cattle/` was generated by `src/tools/sam2_to_coco_seg.py` from
> `data/processed/segmentation/` (SAM2 outputs converted to YOLO-seg format).
> `rfdetr_seg_eval/` was prepared from `data/raw/kaggle-cow-segmentation-dataset/`.
> Neither is raw source data, yet both sit at the top level of `data/` alongside `raw/` and
> `processed/`. They were not moved because the RF-DETR-Seg model is already trained and
> these scripts will not run again before thesis defense. If a future student re-trains
> RF-DETR-Seg, move both into `data/processed/` and update path references in:
> `scripts/07_train_rfdetr_seg.sh`, `scripts/hipe/train_rfdetr_seg_medium.py`,
> `scripts/hipe/train_rfdetr_seg_parameterized.py`, `src/tools/sam2_to_coco_seg.py`,
> `src/tools/eval_rfdetr_seg.py`, `src/tools/eval_rfdetr_seg_kaggle.py`.

#### Key concepts

**raw/ is read-only.** Pipeline scripts never write into `raw/`. All outputs go into
`processed/`. This means you can always re-run any pipeline stage from scratch without
touching the source data.

**processed/ is a one-directional flow.** Each stage reads from the previous stage's output dir:
```
detection/ → segmentation[_rfdetr]/ → tracking/ → tracking_v2[_rfdetr]/ → tubelets[_rfdetr]/
```
Scripts are numbered to reflect this order. If you corrupt a processed dir, just re-run
the corresponding script — nothing in raw/ is at risk.

**v1 vs v2 naming convention.** Directories without a suffix (e.g. `segmentation/`,
`tracking_v2/`, `tubelets/`) are the v1 SAM2 path. Directories with `_rfdetr` suffix are the
v2 RF-DETR-Seg path. The v2 path produced the primary thesis numbers.

**The combined/ detection dataset has a validation set caveat.** `processed/detection/combined/valid/`
contains only CBVD-5 images (1,612 images) because the CVB validation frames were not merged
during the dataset preparation step. The model was checkpointed against this CBVD-5-scoped
validation set. This is why the per-dataset test AP numbers (CBVD-5 45.9%, CVB 5.7%) are the
authoritative figures, not the 70.4% reported during training.

**Symlinks in detection/.** Images in `processed/detection/` are symlinks into `raw/`.
If you move the repo without moving the raw data, all symlinks will break and inference scripts
will silently skip all images (0/N result). Fix with `scripts/fix_detection_symlinks.py`.

**label_map.json is empty.** Don't be confused — the real label logic is in
`src/data/label_utils.py`. The JSON file is a placeholder from early project design.

#### What consumes `labels.csv`

`data/processed/tubelets/labels.csv` (v1) and `tubelets_rfdetr/labels.csv` (v2) are the
primary index files consumed by VideoMAE training and evaluation. Each row is one tubelet:

```
dataset, video_id, tubelet_dir, start_frame, end_frame, label_id
```

`tubelet_dir` is a relative path from the repo root. `track_id` is the last component of
that path. Scripts 10/10b and 11/11b both read this CSV to build their PyTorch datasets.

---

---

### `docker/`

One Dockerfile per pipeline stage group, plus a `docker-compose.yml` that ties them together.
All images share the same base: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`.
This was chosen because it is compatible with both the RTX 3060 (local dev, CUDA 12.1)
and the V100 (HiPE1, CUDA 11.x runtime but 12.x driver).

```
docker/
├── docker-compose.yml       # defines 6 services, one per stage group
├── Dockerfile.data          # CPU: scripts 01–04, 09, 12
├── Dockerfile.detection     # GPU: scripts 05–06 (RF-DETR detect + train)
├── Dockerfile.segmentation  # GPU: script 07 (SAM2 inference)
├── Dockerfile.rfdetr_seg    # GPU: script 07_train_rfdetr_seg.sh (HiPE1 training)
├── Dockerfile.tracking      # CPU: script 08 (OC-SORT tracking)
└── Dockerfile.behavior      # GPU: scripts 10–11 (VideoMAE train + eval)
```

#### Service map (`docker-compose.yml`)

| Service | Dockerfile | GPU? | Pipeline scripts | Notes |
|---|---|---|---|---|
| `data` | Dockerfile.data | No | 01–04, 09, 12 | Dataset prep, tubelet export, analytics |
| `detection` | Dockerfile.detection | Yes | 05–06 | RF-DETR training + detection inference |
| `segmentation` | Dockerfile.segmentation | Yes | 07 | SAM2 mask generation (v1 path) |
| `rfdetr_seg` | Dockerfile.rfdetr_seg | Yes | 07_train | RF-DETR-Seg distillation; runs on HiPE1 |
| `tracking` | Dockerfile.tracking | No | 08 | OC-SORT; requires third_party/OC_SORT/ |
| `behavior` | Dockerfile.behavior | Yes | 10–11 | VideoMAE train + eval; needs shm_size=16g |

**Why no `analytics` service?** Analytics (script 12) is CPU-only and shares all deps with
`data` prep, so it reuses the `data` service.

#### What was actually used for the thesis

All 6 Dockerfiles exist and are wired into `docker-compose.yml`, but most were never the
actual execution method for the thesis pipeline. The real execution split was:

| Dockerfile | Actually ran? | How the thesis actually ran it |
|---|---|---|
| `Dockerfile.rfdetr_seg` | **Yes** | RF-DETR-Seg training on HiPE1 ran via `docker compose run rfdetr_seg` — Docker was mandatory here because HiPE1 has no conda env |
| `Dockerfile.behavior` | **Partially** | `10b_train_behavior_rfdetr.sh` SSHes into HiPE1 and runs `docker run` for the GPU training; falls back to local conda for local runs |
| `Dockerfile.data` | **No** | Scripts 01–04, 09, 12 activate conda and call Python directly |
| `Dockerfile.detection` | **No** | `05_train_detector.sh` and `06_run_detection.sh` activate conda and call Python directly |
| `Dockerfile.segmentation` | **No** | `07_run_segmentation.sh` activates conda and calls Python directly |
| \`Dockerfile.tracking\` | **No** | tracking runs via \`07_run_segmentation.sh\` (v1) or \`08b_run_tracking_rfdetr.sh\` (v2) using conda directly |

**The rule of thumb:** Docker was the runtime only on HiPE1 (which has no conda — everything
there runs in containers). On the local RTX 3060, every script activates the `cattletransformer`
conda environment and runs Python directly. Docker was designed as an alternative path and
is documented in `docs/docker.md`, but conda was used for all actual local thesis runs.

The Dockerfiles are still worth keeping — they serve as a reproducible environment spec
even if you never `docker run` them. A future student without a local conda setup could
use them to replicate the environment exactly.

#### Volume mounts

All services mount the repo as `/workspace` inside the container. The pattern:
- `../data`, `../results`, `../runs` — mounted **read-write** (the pipeline writes here)
- `../configs`, `../src`, `../weights`, `../scripts/hipe` — mounted **read-only** (code only)
- `../third_party` — read-only; only mounted in the `tracking` service (needs `OC_SORT/`)

This design means you can iterate on Python code in `src/` and re-run a container without
rebuilding the image. The image only needs rebuilding when system packages or pip deps change.

#### Critical constraints

- **`shm_size: "16g"`** on GPU services — VideoMAE's PyTorch DataLoader spawns multiple
  workers that share memory via `/dev/shm`. The default Docker shm (64 MB) causes the
  DataLoader to crash. This is the most common Docker failure on this project.
- **`runtime: nvidia`** — must have NVIDIA Container Toolkit installed. Missing this causes
  a generic "unknown runtime" error from Docker.
- **`Dockerfile.tracking` needs OC-SORT pre-cloned** — the Dockerfile cannot clone it
  (no internet in build context by design). Clone first: `git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT`

#### How to use

```bash
# Run a single stage (example: behavior training):
docker compose run behavior python src/behavior/train.py --config configs/behavior/videomae_combined_v2.yaml

# Run a full pipeline stage via its shell script:
docker compose run data bash scripts/09b_generate_tubelets_rfdetr.sh

# Build a specific image:
docker compose build behavior
```

---

---

### `figures/` *(deleted)*

This directory no longer exists. Figure generation was consolidated into
`notebooks/thesis_figures.ipynb`. See the `notebooks/` section for details.

---

---

### `logs/`

Committed training and evaluation logs from HiPE1 — the V100 server where all VideoMAE
training runs and the RF-DETR-Seg distillation ran. Nothing here is generated locally;
every file is a captured stdout/stderr stream from a Docker container on HiPE1.

```
logs/
├── README.md                        # index of every log file with best-F1 and notes
├── train_cbvd5_v1.log               # v1 VideoMAE Config 1 training: CBVD-5 in-domain
├── train_cvb_v1.log                 # v1 VideoMAE Config 2 training: CVB in-domain
├── train_combined_v1.log            # v1 VideoMAE Config 5 training: CBVD-5 + CVB combined
├── train_cbvd5_to_cvb_v1.log        # v1 VideoMAE Config 3 training: CBVD-5 → CVB cross-domain
├── train_cvb_to_cbvd5_v1.log        # v1 VideoMAE Config 4 training: CVB → CBVD-5 cross-domain
├── train_cbvd5_v2.log               # v2 VideoMAE Config 1 training (RF-DETR-Seg tubelets)
├── train_cvb_v2.log                 # v2 VideoMAE Config 2 training
├── train_combined_v2.log            # v2 VideoMAE Config 5 training
├── train_cbvd5_to_cvb_v2.log        # v2 VideoMAE Config 3 training
├── train_cvb_to_cbvd5_v2.log        # v2 VideoMAE Config 4 training
├── eval_cbvd5_v1.log                # script 11 eval output for Config 1 (v1)
├── eval_cvb_v1.log                  # script 11 eval output for Config 2 (v1)
├── eval_combined_v1.log             # script 11 eval output for Config 5 (v1)
├── eval_cbvd5_to_cvb_v1.log         # script 11 eval output for Config 3 (v1)
├── eval_cvb_to_cbvd5_v1.log         # script 11 eval output for Config 4 (v1)
├── tubelet_reexport_cvb.log         # CVB tubelet re-export (2026-04-29) after relaxing frame threshold
└── train_rfdetr_seg_hipe1.log       # RF-DETR-Seg Phase 3b training log (1.3 MB JSON Lines)
```

#### Training logs (`cbvd5.log`, `cvb.log`, etc.)

Each file captures stdout from `src/behavior/train.py` running inside a Docker container on
HiPE1. The format is one line per epoch:

```
Epoch   4 | lr=5.00e-05 | train_loss=0.1934 train_f1=0.6613 | val_loss=1.0126 val_macro_f1=0.3149
         per-class val F1: Standing=0.906  Lying=0.303  Foraging=0.896  Drinking=0.000  Ruminating=0.100  Grooming=0.000  Other=0.000
  -> new best val_macro_f1=0.3149
```

The `cudnn` `UserWarning: Plan failed` lines that appear at the top of every log are harmless —
PyTorch falls back to an alternative convolution plan and results are unaffected.

v1 logs (`train_*_v1.log`) ran April 28 – May 1, 2026.
v2 logs (`train_*_v2.log`, RF-DETR-Seg tubelets) ran May 20–21, 2026.

The best macro-F1 from each log is the definitive source for the Config 1–5 numbers in the
thesis. Cross-reference: `logs/README.md` has the summary table; `results/behavior/f1_per_class.csv`
has the full per-class breakdown from the eval scripts.

#### Eval logs (`eval_*.log`)

Outputs from script 11 (`11_evaluate.sh`), which loads the best checkpoint and writes
per-class F1, precision, and recall. These logs cover only the v1 configs — v2 evaluation
was run but output was captured directly into `results/behavior/` CSVs rather than log files.

#### `tubelet_reexport_cvb.log`

Not a training log — this records a tubelet re-export run (script 09) after the CVB frame
presence threshold was relaxed from 16/16 to 12/16 (accept tubelets with at least 12 of 16
frames present). The log shows 114,217 new CVB tubelets added, bringing the total to 125,586
rows in `labels.csv`. This change happened before the v1 training runs.

#### `LOG_FILE`

A Docker error from a failed local attempt to run a container on the RTX 3060:
```
nvidia-container-cli: device error: DEVICE_ID: unknown device: unknown.
```
This is an artifact of debugging the Docker setup on the local machine. All actual training
ran on HiPE1, not locally via Docker. This file can be deleted if it causes confusion —
it has no research value.

#### `train_rfdetr_seg_hipe1.log` (1.3 MB)

The training log from the RF-DETR-Seg distillation experiment (Phase 3b) on HiPE1. Unlike
the VideoMAE logs, this is in JSON Lines format — one dense JSON object per training step,
emitted by the RF-DETR training framework. It corresponds to the "Config B" hyperparameter
sweep: `seg_medium_lr5e5` (medium backbone, learning rate 5e-5), which produced the
checkpoint used for all v2 thesis results (`runs/seg_medium_lr5e5/checkpoint_best_ema.pth`).

At 1.3 MB it is not human-readable directly; use `jq` or Python to extract epoch-level
metrics if you need to plot the RF-DETR-Seg training curve.

---

---

---

### `models/`

Pretrained weights from public sources — the starting points for this project's fine-tuning.
Everything here is gitignored (large binary files) and must be downloaded before running
the pipeline. Download commands are in `docs/setup.md`.

```
models/
├── rf-detr/
│   ├── rf-detr-medium.pth       # RF-DETR-Medium COCO-pretrained backbone (387 MB)
│   └── rf-detr-seg-medium.pt    # RF-DETR-Seg-Medium COCO-pretrained backbone (137 MB)
└── sam2/
    └── sam2.1_hiera_large.pt    # SAM2 Hiera-Large checkpoint (857 MB)
```

**The key distinction between `models/` and `runs/`:**
- `models/` — weights downloaded from public sources (RF-DETR paper, SAM2 paper). These are
  the pretrained starting points that you fine-tune.
- `runs/` — fine-tuned checkpoints produced by *this project's* training scripts. These are
  the models that actually run on cattle data.

A new student needs both: `models/` to start training, `runs/` to run inference or evaluation.

#### `models/rf-detr/rf-detr-medium.pth` (387 MB)

COCO-pretrained RF-DETR-Medium backbone. Used as the initialization for fine-tuning the
cattle detector in script 05 (`src/detection/train.py`). The RF-DETR library reads this
via `configs/detection/rfdetr_combined.yaml`.

After training, the fine-tuned cattle detector lives at:
`runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`

Download: `huggingface-cli download sakifkhan/cattle-vision-framework rf-detr-medium.pth --local-dir models/rf-detr/`

#### `models/rf-detr/rf-detr-seg-medium.pt` (137 MB)

COCO-pretrained RF-DETR-Seg-Medium backbone. Used as the initialization for the RF-DETR-Seg
distillation training on HiPE1 (`scripts/hipe/train_rfdetr_seg_medium.py`). The training
script passes `CHECKPOINT = None` to signal "start from COCO pretrained" — the RF-DETR
library resolves that to this file.

After training, the fine-tuned RF-DETR-Seg model lives at:
`runs/seg_medium_lr5e5/checkpoint_best_ema.pth`

This is the most important checkpoint in the repo — it is the single-stage detect+segment
model used for all v2 thesis results.

#### `models/sam2/sam2.1_hiera_large.pt` (857 MB)

SAM2 Hiera-Large checkpoint from Meta. Used only in the v1 pipeline path (script 07,
`src/segmentation/segment.py`) to generate per-frame instance masks that OC-SORT then
uses for mask IoU association.

Referenced in `configs/segmentation/sam2.yaml:checkpoint`.

Not used in the v2 path at all — v2 replaced SAM2 with the single-stage RF-DETR-Seg model.
If you are only running the v2 pipeline, you do not need this file.

Download: `wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O models/sam2/sam2.1_hiera_large.pt`

#### `third_party/OC_SORT/` — why it is NOT here

OC-SORT is a Python source package, not a weight file. `src/tracking/track.py` imports
directly from its internal module tree (`from trackers.ocsort_tracker.ocsort import OCSort`),
which requires the full source to be on `sys.path`. It cannot be pip-installed with the
same import path. `third_party/` is the standard location for vendored external source
that can't go through a package manager. Clone it during setup:

```bash
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
```

---

---

---

### `notebooks/`

All exploratory analysis, figure generation, and visual plate assembly. Nothing here is
imported by pipeline code — notebooks are standalone tools for understanding and presenting
results. All notebooks must be run from the repo root with the `cattletransformer` conda
environment active.

```
notebooks/
├── figure_charts.ipynb      # All matplotlib charts — reads CSVs, outputs graphs/
├── figure_plates.ipynb      # All image plates — renders frames, assembles multi-panel plates
└── dataset_analysis/        # One notebook per dataset, exploratory only
    ├── analysis_cbvd5.ipynb
    ├── analysis_cvb.ipynb
    ├── analysis_cattleeyeview.ipynb
    ├── analysis_cows2021.ipynb
    ├── analysis_freeman.ipynb
    ├── analysis_mmcows.ipynb
    ├── analysis_opencows2020.ipynb
    └── dataset_comparison.ipynb
```

#### `figure_charts.ipynb`

Pure matplotlib/seaborn chart generation. Reads committed CSVs from `results/` and
produces publication-quality figures saved to `results/figures/graphs/`. Fast to run —
no model inference, no raw video access.

Charts produced (one section per chart group):

| Section | Source data | Output |
|---|---|---|
| Behavior F1 Comparison | hardcoded v2 thesis numbers | `graphs/behavior_f1_comparison.{pdf,png}` |
| Cross-Domain F1 Comparison | hardcoded v2 thesis numbers | `graphs/cross_domain_f1_comparison.{pdf,png}` |
| Activity Budget + Timeline | `results/analytics/timelines/` | `graphs/activity_budget_timeline.{pdf,png}` |
| RF-DETR-Seg Training | `logs/train_rfdetr_seg_hipe1.log` | `graphs/rfdetr_seg_training/*.png` |
| Training Curves | `results/behavior/training_logs/` | `graphs/training_curves_*.png` |
| Per-Class F1 | `results/behavior/f1_per_class.csv` | `graphs/per_class_f1.png` |
| Cross-Dataset mAP | `results/detection/*.json`, `results/generalization/ood_summary.csv` | `graphs/cross_dataset_map.png` |
| Perturbation Heatmap | `results/generalization/perturbation_delta.csv` | `graphs/perturbation_heatmap.png` |
| Tracking Distribution | `results/tracking/tracking_per_video_all.csv` | `graphs/tracking_distribution.png` |
| min_hits Ablation | `results/tracking/minhits_ablation.csv` | `graphs/minhits_ablation.png` |
| Activity Budget (aggregate) | `results/analytics/activity_budget.csv` | `graphs/activity_budget.png` |
| Behavior Transition | `results/analytics/transition_matrix.csv` | `graphs/transition_heatmap.png` |
| Behavioral Deviation | `results/analytics/behavior_deviation.csv` | `graphs/behavioral_deviation.png` |
| Pipeline Timing | `results/inference/timing.csv` | `graphs/pipeline_timing.png` |
| Segmentation Coverage | `results/segmentation/*_stats.csv` | `graphs/seg_coverage.png` |

The RF-DETR-Seg training section compares Config A (lr=1e-4) vs Config B (lr=5e-5). Only
Config B's log is committed (`logs/train_rfdetr_seg_hipe1.log`). Set `LOG_A = None` (the
default) to skip Config A curves — the section runs fine on Config B alone.

#### `figure_plates.ipynb`

Image plate generation — the visual figures in the thesis showing pipeline progression,
example frames, and perturbation effects. Runs in two sequential phases:

**Phase 1 — Render** (slow, requires GPU + model checkpoints):
Loads `runs/seg_medium_lr5e5/checkpoint_best_ema.pth`, selects representative frames
from each dataset, runs RF-DETR-Seg inference, and saves per-frame PNGs at five
progressively richer overlay levels:

| Render pass | What it adds | Output path |
|---|---|---|
| `detection` | RF-DETR bounding boxes only | `frames/{ds}/detection/` |
| `det_seg` | + instance mask overlay | `frames/{ds}/det_seg/` |
| `det_seg_track` | + OC-SORT track IDs | `frames/{ds}/det_seg_track/` |
| `full_pipeline` | + behavior label from VideoMAE | `frames/{ds}/full_pipeline/` |
| `perturbations` | clean + low + high for each of 5 types | `frames/{ds}/perturbations/{type}_{sev}/` |

All frame paths are recorded in `results/figures/frames/manifest.json` at the end of the
render phase. **Run Phase 1 only once** — it is slow and idempotent (existing files are
not overwritten). Re-run only if you change the frame selection or the checkpoint.

**Phase 2 — Assemble** (fast, CPU only):
Reads `manifest.json` and lays out six figure plates:

| Plate | Layout | Output |
|---|---|---|
| Pipeline overview | 6 datasets × 4 render passes grid | `plates/pipeline_overview_plate.png` |
| Pipeline progression strips | per-stage strip for CBVD-5 and CVB | `plates/pipeline_progression_{ds}.png` |
| Behavior class examples | 7 classes × 3 example tubelets | `plates/behavior_class_examples.png` |
| Perturbation comparison | one plate per perturbation type | `plates/perturbation_plate_{type}.png` |
| Freeman severity strip | clean + 10 perturbation conditions | `plates/perturbation_severity_strip_freeman.png` |
| Detection gallery | one strip per dataset | `plates/detection_gallery_{ds}.png` |

Also contains the **Perturbation Illustrations** and **Tracking Identity Strip** static
figures from the thesis — these read pre-rendered PNGs from `results/figures/frames/`
and raw tracking JSONs from `data/processed/tracking_v2/`.

All plate outputs go to `results/figures/plates/` at 300 DPI.

#### `dataset_analysis/`

Eight standalone exploratory notebooks — one per dataset (CBVD-5, CVB, CattleEyeView,
Cows2021, Freeman Center, mmCows, OpenCows2020) plus a cross-dataset comparison.

These were used during Phase 0 and Phase 8 to understand annotation distributions, frame
counts, class imbalances, and domain gaps before building the pipeline. They are not
required to reproduce any thesis result — they are reference material and starting points
for a student who wants to understand the datasets deeply.

Each notebook reads directly from `data/raw/{dataset}/` and produces console output and
inline plots. Nothing is saved to `results/`.

---

### `results/`

Committed output artifacts from every pipeline stage. Nothing here is gitignored at the
top level — large subdirectories (`figures/frames/`, `figures/plates/`, `tracking/videos/`,
`tracking/behavior_videos/`, `tracking/visualizations/`, `segmentation/viz/`) are gitignored
because they contain generated images and videos that can be reproduced from scripts.

```
results/
├── dataset_analysis/          # Phase 0/8 dataset exploration outputs
├── detection/                 # Per-dataset detector evaluation JSONs + demo video
├── behavior/                  # VideoMAE classifier outputs (predictions, confusion matrices, training logs)
│   ├── predictions_sam2_tubelets/    # Predictions from tubelets derived via SAM2 tracking path (v1 + v2 runs)
│   ├── predictions_rfdetr_tubelets/  # Predictions from tubelets derived via RF-DETR-Seg tracking path (v2, thesis primary)
│   ├── confusion_matrices/           # One PNG per run
│   └── training_logs/                # Per-epoch loss/F1 CSVs for all 10 VideoMAE runs
├── behavior_analytics/        # Behavioral analytics from the SAM2-tubelet pipeline (v1)
│   └── timelines/             # Per-video per-track behavior timeline CSVs (gitignored)
├── behavior_analytics_rfdetr/ # Behavioral analytics from the RF-DETR-Seg-tubelet pipeline (v2, primary thesis)
│   └── timelines/             # Per-video per-track behavior timeline CSVs (gitignored)
├── segmentation/              # RF-DETR-Seg training metadata and evaluation results
├── tracking/                  # OC-SORT evaluation JSONs, ablation CSVs, and sample videos
├── generalization/            # OOD detection summary + perturbation robustness results
├── inference/                 # Per-dataset batch inference outputs (detection JSONs, timing CSVs)
├── figures/                   # Publication figures (see figure_charts.ipynb / figure_plates.ipynb)
│   ├── graphs/                # Matplotlib chart PNGs and PDFs
│   ├── plates/                # Multi-panel figure plates at 300 DPI (gitignored)
│   └── frames/                # Per-frame render passes (gitignored)
└── visualizations/            # Miscellaneous exploratory visualizations (one tubelet GIF)
```

#### `results/dataset_analysis/`

Dataset exploration outputs generated by `notebooks/dataset_analysis/` notebooks during
Phase 0 (CBVD-5, CVB) and Phase 8 (CattleEyeView, Cows2021, Freeman, mmCows, OpenCows2020).

Each dataset contributes three files:
- `{dataset}_summary.json` — frame counts, annotation counts, class distribution, domain metadata
- `{dataset}_sample_frames.png` — grid of representative frames
- `{dataset}_class_distribution.png` — bar chart of annotation counts per behavior class

Three cross-dataset comparison figures (`comparison_behavior_distribution.png`,
`comparison_dataset_scale.png`, `comparison_evaluation_scope.png`) are produced by
`notebooks/dataset_analysis/dataset_comparison.ipynb`.

None of these files are required to reproduce thesis results — they are exploratory reference material.

#### `results/detection/`

Generated by `src/tools/eval_detection_ood.py` (OOD evals), `src/tools/aggregate_ood_summary.py`
(rollup), and `src/detection/infer_video.py` (demo video). OOD evals are driven by
`scripts/14_eval_freeman_detection.sh` and `scripts/19_eval_mmcows_detection.sh`; the remaining
OOD datasets (CattleEyeView, Cows2021, OpenCows2020) were evaluated via direct `eval_detection_ood.py`
invocations during Phase 8.

| File | Contents |
|---|---|
| `cbvd5_test_ap.json` | RF-DETR mAP on CBVD-5 test split (45.9% mAP@50) |
| `cvb_test_ap.json` | RF-DETR mAP on CVB test split (5.7% mAP@50) |
| `cattleeyeview_eval.json` | OOD eval — CattleEyeView (47.0% mAP@50) |
| `cows2021_eval.json` | OOD eval — Cows2021 (27.3% mAP@50) |
| `opencows2020_eval.json` | OOD eval — OpenCows2020 (33.3% mAP@50) |
| `freeman_detection_eval.json` | OOD eval — Freeman Center (73.0% mAP@50) |
| `mmcows_eval.json` | OOD eval — mmCows (32.4% mAP@50) |
| `combined_ood.json` | Rolled-up summary across all OOD datasets |
| `618_detections.json` | Raw CBVD-5 clip 618 detections (demo artifact) |
| `618_detected.mp4` | Annotated demo video for clip 618 |

#### `results/behavior/`

**`f1_per_class.csv`** — consolidated macro-F1 and per-class F1 for all 10 VideoMAE runs (5
configs × v1/v2). This is the primary thesis behavior results table. Rows are appended
incrementally by `src/behavior/evaluate.py` — each eval run adds its row.

**`v1_v2_comparison.csv`** — side-by-side v1 vs v2 macro-F1 for each config. Used in thesis §5.
Generated manually by aggregating rows from `f1_per_class.csv`.

**`predictions_sam2_tubelets/`** — 11 prediction CSVs from models trained on SAM2-derived
tubelets. Generated by `scripts/11_evaluate.sh` → `src/behavior/evaluate.py`. Contains both
v1 and v2 VideoMAE runs for all 5 configs, plus a sanity-check run.
Naming: `videomae_{config}_{v1|v2}_val.csv`.

**`predictions_rfdetr_tubelets/`** — 5 prediction CSVs from models trained on RF-DETR-Seg-derived
tubelets (v2 path only, thesis primary numbers). Generated by `scripts/11b_evaluate_rfdetr.sh`.
Same naming convention but only v2.

**`confusion_matrices/`** — one PNG per prediction CSV, generated by `src/behavior/evaluate.py`.

**`training_logs/`** — per-epoch `train_loss`, `val_loss`, `val_f1` CSVs for all runs.
v1 logs written directly by `src/behavior/train.py` during `scripts/10_train_behavior.sh`.
v2 logs parsed from HiPE1 training output by `src/tools/parse_v2_logs.py` (HiPE1 ran the
v2 training; the raw logs are committed under `logs/train_*_v2.log`).
Used by `figure_charts.ipynb` (Training Curves section).

#### `results/behavior_analytics/` and `results/behavior_analytics_rfdetr/`

Behavioral analytics computed from pipeline outputs. The two folders are structurally
identical — one uses the SAM2-tubelet path (v1, `behavior_analytics/`), the other uses
the RF-DETR-Seg-tubelet path (v2, `behavior_analytics_rfdetr/`). The RF-DETR variant is
the thesis primary.

Generated by `scripts/12_generate_analytics.sh` and `scripts/12b_generate_analytics_rfdetr.sh`.

| File | Contents |
|---|---|
| `activity_budget.csv` | Mean % time per behavior class per dataset |
| `transition_matrix.csv` | Behavior-to-behavior transition probabilities |
| `behavior_deviation.csv` | Per-behavior deviation from median per video |
| `timelines/` | Per-video per-track behavior timeline CSVs — **gitignored** (large) |

Note: `behavior_analytics_rfdetr/timelines/` only contains CBVD-5 data (CVB timeline
generation was not re-run on the RF-DETR path).

#### `results/segmentation/`

The per-dataset stats and summaries are written by `src/segmentation/segment.py` during
`scripts/07_run_segmentation.sh`. The `kaggle_eval*` directories were produced by
`src/tools/eval_rfdetr_seg_kaggle.py` against the CattleEyeView test set. The `seg_medium_lr*_metadata/` directories contain training artifacts copied from HiPE1
after the RF-DETR-Seg training runs (Phase 3b).

| Path | Contents |
|---|---|
| `cbvd5_segmentation_stats.csv` / `cvb_segmentation_stats.csv` | Per-frame mask coverage and IoU stats from SAM2 segmentation |
| `cbvd5_summary.json` / `cvb_summary.json` | Summary of segmentation run (frame count, coverage rate) |
| `cattleeyeview_maskiou.json` | Mask IoU on CattleEyeView (86.5%) — best OOD result |
| `kaggle_eval/` | Stale RF-DETR-Seg eval using hardcoded absolute checkpoint path — superseded by `kaggle_eval_seg_lr5e5/` |
| `kaggle_eval_seg_lr5e5/` | RF-DETR-Seg Config B (lr=5e-5) eval on 269 CattleEyeView images: bbox mAP@50=51.5%, mask mAP@50=74.9% |
| `seg_medium_lr1e4_metadata/` | Training metadata for Config A (lr=1e-4): log, results JSON, run config |
| `seg_medium_lr5e5_metadata/` | Training metadata for Config B (lr=5e-5, thesis checkpoint): same structure |
| `viz/` | Per-frame segmentation overlay PNGs for CBVD-5 and CVB — **gitignored** |

#### `results/tracking/`

Eval JSONs and CSVs are written by standalone tool scripts run after tracking (script 07 v1 path or script 08b v2 path):
`src/tools/write_cvb_idf1.py` produces the per-dataset IDF1/MOTA JSONs and
`tracking_per_video_all.csv`; `src/tools/write_minhits_ablation.py` produces the ablation
files. Visualization videos and PNGs are generated by `src/tracking/render_tracks.py`.

| File / Dir | Contents |
|---|---|
| `cbvd5_idf1.json` / `cvb_idf1.json` | Per-video IDF1 and MOTA on each dataset |
| `tracking_summary_all.json` | Rolled-up IDF1=67.31%, MOTA=36.61% across both datasets |
| `tracking_per_video_all.csv` | Per-video tracking metrics for all videos (used in figure_charts.ipynb) |
| `minhits_ablation.csv` | OC-SORT `min_hits` ablation: IDF1/MOTA/ID switches at min_hits ∈ {1,2,3,5} |
| `cvb_mh{1,2,3,5}_summary.json` / `_per_video.csv` | Detailed results for each ablation point |
| `videos/` | 2 tracking-only visualization MP4s (bounding boxes + track IDs, no behavior labels) — **gitignored** |
| `behavior_videos/` | 3 tracking + behavior label overlay MP4s — **gitignored** |
| `visualizations/` | Per-frame tracking overlay PNGs for CBVD-5 and CVB — **gitignored** |

#### `results/generalization/`

`ood_summary.csv` is produced by `src/tools/aggregate_ood_summary.py` by rolling up all
per-dataset detection eval JSONs from `results/detection/`. The 60 `perturbation_runs/` JSONs
are produced by `src/tools/eval_detection_perturbation.py` (one invocation per
dataset × type × severity) and then summarised into `perturbation_delta.csv` by
`src/tools/aggregate_perturbation.py`.

| File / Dir | Contents |
|---|---|
| `ood_summary.csv` | Detection mAP@50 for all 5 OOD datasets + in-domain baselines |
| `perturbation_delta.csv` | mAP drop (Δ) for 5 perturbation types × 2 severities × 6 datasets (60 rows) |
| `perturbation_runs/` | Raw per-run JSON eval files: `{dataset}_{type}_{severity}.json` (60 files) |

The perturbation runs use the base RF-DETR detector (not the RF-DETR-Seg checkpoint). See
§10 of `CLAUDE.md` for the re-run requirement before final submission.

#### `results/inference/`

Per-dataset outputs from batch inference runs. Each named subdirectory contains per-video
result folders with detection JSONs, tracking JSONs, and timing CSVs.

| Subdir | Generated by | Notes |
|---|---|---|
| `cbvd5/` | `scripts/17_infer_cbvd5.sh` → `src/inference/pipeline.py` | Batch inference on all CBVD-5 videos |
| `cvb/` | `scripts/16_infer_cvb.sh` → `src/inference/pipeline.py` | Batch inference on all CVB videos |
| `freeman/` | `scripts/15_infer_video.sh` × 14 videos | 14-video Phase 9 end-to-end validation on ranch footage |
| `mmcows/` | `scripts/18_prepare_mmcows_detection.sh` + direct `pipeline.py` invocation | Detection-only subset run |

#### `results/figures/`

Publication figures generated by the two figure notebooks:

| Dir | Contents | Generated by |
|---|---|---|
| `graphs/` | Matplotlib chart PNGs and PDFs (15 chart groups) | `figure_charts.ipynb` |
| `plates/` | Multi-panel figure plates at 300 DPI (6 plate types, ~38 files) — **gitignored** | `figure_plates.ipynb` Phase 2 |
| `frames/` | Per-frame render pass PNGs + `manifest.json` — **gitignored** | `figure_plates.ipynb` Phase 1 |

#### `results/visualizations/`

Contains one exploratory artifact: `tubelet_kf7_inst9cbccd.gif` — an animated GIF of a
single tubelet clip used during Phase 5 to verify tubelet generation correctness. Not
referenced by any pipeline script; kept as reference only.

---

### `runs/`

Local training run outputs — checkpoints, logs, and metadata produced by each training
script. Weight files (`.pt`, `.pth`) are gitignored globally and must be downloaded from
HuggingFace (see `docs/setup.md`). Non-weight artifacts follow a per-subdirectory policy
described below.

```
runs/
├── detection/
│   └── rfdetr_combined_v1/      # RF-DETR detector trained on combined CBVD-5 + CVB
│       ├── checkpoint_best_total.pth   # ← gitignored; download from HuggingFace
│       ├── log.txt              # ← COMMITTED (176 KB training log)
│       ├── results.json         # ← COMMITTED (final mAP numbers)
│       └── metrics_plot.png     # ← COMMITTED (loss/mAP curve)
├── behavior/
│   └── videomae_{config}_{v1|v2}/   # One dir per VideoMAE training run (11 total)
│       ├── checkpoint_best.pt   # ← gitignored; download from HuggingFace
│       └── log.csv              # ← gitignored; duplicated in results/behavior/training_logs/
└── seg_medium_lr5e5/            # RF-DETR-Seg Config B (lr=5e-5) — thesis checkpoint
    ├── checkpoint_best_ema.pth  # ← gitignored; download from HuggingFace
    ├── log.txt                  # ← gitignored; duplicated in logs/train_rfdetr_seg_hipe1.log
    ├── results.json             # ← gitignored; duplicated in results/segmentation/seg_medium_lr5e5_metadata/
    └── run_config.json          # ← gitignored; duplicated in results/segmentation/seg_medium_lr5e5_metadata/
```

#### Why no `runs/sam2/`

SAM2 is a pretrained model — it is never trained or fine-tuned in this pipeline. Script 07
runs SAM2 inference using the checkpoint downloaded to `models/sam2/`. There is no local
training run, so no `runs/sam2/` directory exists or is needed.

#### Why no `runs/tracking/`

OC-SORT does not require training from scratch. It uses default pretrained weights bundled
with the OC_SORT repo. Scripts 08/08b run tracking inference only; no checkpoint is
produced. The directory was removed as an empty placeholder.

#### Gitignore strategy

The blanket `runs/` ignore was replaced with targeted rules so that small, useful artifacts
can be committed without pulling in multi-GB weight files:

| Rule | What it covers | Reason |
|---|---|---|
| `*.pth` / `*.pt` (global) | All weight files everywhere | Too large for git; on HuggingFace |
| `*.tfevents.*` (global) | TensorBoard binary event files | Binary, large, not useful in git |
| `runs/behavior/` | All behavior run dirs | `log.csv` files duplicated in `results/behavior/training_logs/` |
| `runs/seg_medium_lr5e5/log.txt` | Seg training log | Duplicated in `logs/train_rfdetr_seg_hipe1.log` |
| `runs/seg_medium_lr5e5/results.json` + `run_config.json` | Seg run metadata | Duplicated in `results/segmentation/seg_medium_lr5e5_metadata/` |

`runs/detection/rfdetr_combined_v1/log.txt`, `results.json`, and `metrics_plot.png` are the
only files in `runs/` that are committed — they have no duplicate elsewhere in the repo.

---

### `scripts/`

Shell wrappers that drive the pipeline end-to-end. Every script runs from the **project root** with `bash scripts/<name>.sh`. Python is always invoked through the `cattletransformer` conda environment (the newer scripts activate it internally; older ones expect it already active).

#### Numbering scheme

Scripts 01–12 are the core training pipeline. Scripts 13–19 cover OOD evaluation and inference. Two utility scripts (`run_pipeline.sh`, `start_app.sh`) live outside the numbered sequence.

The core pipeline also has a **v1 / v2 split**: unnumbered scripts run the v1 SAM2 path; `b`-suffix scripts (`07b`, `08b`, `09b`, `10b`, `11b`, `12b`) run the v2 RF-DETR-Seg path that produced the primary thesis results.

#### Shared setup (01–06, both paths)

| Script | Stage | Reads | Writes |
|---|---|---|---|
| `01_inspect_data.sh` | Verify downloads | `data/raw/{cbvd5,cvb}/` | console summary |
| `02_prepare_cbvd5.sh` | CBVD-5 → COCO | `data/raw/cbvd5/` | `data/processed/detection/cbvd5/` |
| `03_prepare_cvb.sh` | CVB → COCO | `data/raw/cvb/` | `data/processed/detection/cvb/` |
| `04_merge_datasets.sh` | Merge + mini subset | `data/processed/detection/{cbvd5,cvb}/` | `combined/`, `combined_mini/` |
| `05_train_detector.sh` | Train RF-DETR | `configs/detection/rfdetr_combined.yaml` | `runs/detection/rfdetr_combined_v1/` |
| `06_run_detection.sh` | Detection inference | detector checkpoint + raw videos | `data/processed/tracking/{cbvd5,cvb}/*_detections.json` |

`05_train_detector.sh` accepts `--sanity` (quick smoke test) and `--resume` (continue from checkpoint). `06_run_detection.sh` checks that the checkpoint exists before running and exits early with a clear error if missing.

#### v1 path — SAM2 segmentation (07 → 12)

| Script | Stage | Reads | Writes |
|---|---|---|---|
| `07_run_segmentation.sh` | SAM2 segmentation + OC-SORT tracking (integrated) | detection JSONs + `models/sam2/sam2.1_hiera_large.pt` | `data/processed/tracking_v2/{cbvd5,cvb}/*_tracks.json` |
| `09_generate_tubelets.sh` | Tubelet extraction | tracking_v2 JSONs + raw frames | `data/processed/tubelets/` + `labels.csv` (125k clips) |
| `10_train_behavior.sh` | VideoMAE training (5 configs) | tubelets + `configs/behavior/` | `runs/behavior/videomae_*_v1/checkpoint_best.pt` |
| `11_evaluate.sh` | Behavior evaluation | tubelets + v1 checkpoints | `results/behavior/predictions_sam2_tubelets/`, `f1_per_class.csv`, confusion matrices |
| `12_generate_analytics.sh` | Activity analytics | predictions + tracking_v2 | `results/behavior_analytics/{activity_budget,transition_matrix,behavior_deviation}.csv` |

`07_run_segmentation.sh` checks for SAM2 installation (`sam2` pip package), `pycocotools`, and the checkpoint before running. Accepts `--sanity` (3 videos), `--dataset cbvd5|cvb`, `--video_id <id>`. OC-SORT must be cloned to `third_party/OC_SORT` before running. `10_train_behavior.sh` accepts an optional single config path as `$1`; if omitted it trains all 5 v1 configs sequentially. `11_evaluate.sh` likewise accepts `CONFIG CHECKPOINT` positional args for single-config mode.

#### v2 path — RF-DETR-Seg (07b → 12b) — primary thesis results

| Script | Stage | Reads | Writes |
|---|---|---|---|
| `07b_run_rfdetr_seg.sh` | RF-DETR-Seg inference | `runs/seg_medium_lr5e5/checkpoint_best_ema.pth` + detection JSONs | `data/processed/segmentation_rfdetr/{cbvd5,cvb}/*_masks.json` |
| `08b_run_tracking_rfdetr.sh` | OC-SORT tracking (mask-IoU) | `segmentation_rfdetr/` JSONs | `data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/*_tracks.json` |
| `09b_generate_tubelets_rfdetr.sh` | Tubelet extraction | `tracking_v2_rfdetr/` + raw frames | `data/processed/tubelets_rfdetr/` + `labels.csv` |
| `10b_train_behavior_rfdetr.sh` | VideoMAE training (5 v2 configs) | `tubelets_rfdetr/` + `configs/behavior/` | `runs/behavior/videomae_*_v2/checkpoint_best.pt` |
| `11b_evaluate_rfdetr.sh` | Behavior evaluation | `tubelets_rfdetr/` + v2 checkpoints | `results/behavior/predictions_rfdetr_tubelets/`, confusion matrices, `f1_per_class.csv` (v2 rows appended) |
| `12b_generate_analytics_rfdetr.sh` | Activity analytics | `predictions_rfdetr_tubelets/` + `tracking_v2_rfdetr/` | `results/behavior_analytics_rfdetr/{activity_budget,transition_matrix,behavior_deviation}.csv` |

`07b_run_rfdetr_seg.sh` checks for `rfdetr` and `pycocotools` packages and the EMA checkpoint. Accepts the same `--sanity`, `--dataset`, `--video_id` flags as 07. `10b_train_behavior_rfdetr.sh` adds a `--hipe1` flag that rsyncs tubelets and src to the remote server and runs Docker training there. `11b_evaluate_rfdetr.sh` copies prediction CSVs from `predictions_sam2_tubelets/` into `predictions_rfdetr_tubelets/` automatically after evaluation (the evaluate script always writes to the sam2 dir; 11b copies them to the rfdetr dir for analytics separation).

#### RF-DETR-Seg training (07_train_rfdetr_seg.sh)

This script is provided for **local reproduction** only. The actual thesis training ran on HiPE1 via `scripts/hipe/train_rfdetr_seg_medium.py`. Steps: (1) converts SAM2 mask JSONs to a COCO segmentation dataset using `src/tools/sam2_to_coco_seg.py`, (2) fine-tunes RF-DETR-Seg-Medium. Accepts `--skip_conversion`, `--cbvd5_only`, `--epochs=N`, `--lr=V`. The resulting checkpoint `runs/seg_medium_lr5e5/checkpoint_best_ema.pth` is what all v2 downstream scripts use.

#### OOD evaluation (22–31)

Scripts 22–28 cover the Phase 8 OOD detection evaluation across external datasets.

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `13_prepare_freeman.sh` | Freeman Center YOLO → COCO | `data/raw/freeman-cmb-2024/` | `data/processed/detection/freeman/{train,valid,test}/` |
| `14_eval_freeman_detection.sh` | OOD detection on Freeman Center | freeman test split + detector checkpoint | `results/detection/freeman_detection_eval.json` |
| `18_prepare_mmcows_detection.sh` | mmCows cam_4 YOLO → COCO | `data/raw/mmCows/` (date=0725, cam_4) | `data/processed/detection/mmcows/` |
| `19_eval_mmcows_detection.sh` | OOD detection on mmCows | mmcows COCO + RF-DETR-Seg checkpoint | `results/detection/mmcows_eval.json`; appends to `results/generalization/ood_summary.csv` |

The mmCows behavior eval (tubelet generation + VideoMAE inference) was not run for the thesis — 1 fps capture rate makes tubelet generation impractically slow (GitHub issues #41–#42, closed as wontfix). The detection result from script 19 is the authoritative mmCows entry in the OOD table.

#### Inference scripts (24–26, 31)

These scripts batch-run the Phase 9 inference pipeline (`src/inference/run_inference.py`) across a dataset's videos or frame directories.

| Script | Target | Input | Output |
|---|---|---|---|
| `15_infer_video.sh` | Single video or frames dir | `--video <path>` or `--frames_dir <path>` | `results/inference/<job_id>/` |
| `16_infer_cvb.sh` | All CVB clips | `data/raw/cvb/raw_frames/` | `results/inference/cvb/<clip>/` |
| `17_infer_cbvd5.sh` | All CBVD-5 videos | `data/raw/cbvd5/videos/videos/*.mp4` | `results/inference/cbvd5/<stem>/` |

Both batch scripts (16, 17) support `--resume` (skip clips with existing `activity_budget.csv`), `--limit N` (test run with N clips), `--cleanup` (delete intermediates after each clip), and `--conf_thresh`.

#### Utility scripts

**`run_pipeline.sh`** — orchestrates the entire v1 pipeline (stages 01–12) with two convenience flags:
- `--from N` — skip stages before N (resume after a failure)
- `--stage N` — run only stage N in isolation

**`start_app.sh`** — starts the FastAPI + React web app. Checks that `ui/dist/` is built, activates `cattletransformer`, then launches `uvicorn api.main:app`. Accepts `--port PORT` (default 8000). Attempts to open a browser tab automatically on start.

#### `scripts/hipe/`

HiPE1-specific Python training scripts run inside Docker on the V100 nodes. These are not shell wrappers — they are standalone Python programs invoked directly.

| File | Purpose |
|---|---|
| `train_rfdetr_seg_medium.py` | **Actual thesis training run** — RF-DETR-Seg-Medium, lr=5e-5, Config B EMA. Produced `runs/seg_medium_lr5e5/checkpoint_best_ema.pth`. |
| `train_rfdetr_seg_large.py` | RF-DETR-Seg-Large variant (experimental, not used for final thesis) |
| `train_rfdetr_seg_parameterized.py` | Parameterized variant that reads hyperparams from environment variables |
| `README.md` | HiPE1 Docker launch instructions for these scripts |

The HiPE1 training workflow: rsync `src/`, `configs/`, and `data/rfdetr_seg/cattle/` to HiPE1, then `docker run --gpus all cattle-behavior python scripts/hipe/train_rfdetr_seg_medium.py`. See `docs/hipe_ops.md` for the full workflow.

---

### src/

All importable library code lives here. Scripts in `scripts/` are thin wrappers that call into `src/`. Nothing in `src/` has side effects at import time; every file exposes functions or classes and uses an `if __name__ == "__main__"` guard (or delegates to `main()`) when it can also be run directly.

```
src/
├── analytics/      activity-budget and timeline analysis (Phase 7)
├── behavior/       VideoMAE training, evaluation, and inference (Phases 6, 9)
├── data/           dataset converters, tubelet export, label utilities (Phases 0–5)
├── detection/      RF-DETR training and detection inference (Phases 1–2)
├── inference/      full end-to-end inference pipeline for new videos (Phase 9)
├── segmentation/   SAM2 and RF-DETR-Seg inference (Phases 3, 3b)
├── tools/          one-off evaluation and visualization scripts (Phases 8, 9b)
├── tracking/       OC-SORT tracking, tracking evaluation, video rendering (Phase 4)
└── utils/          (empty — reserved)
```

---

#### `src/analytics/`

Post-classification analysis: converts per-tubelet predictions into per-track timelines, then aggregates into activity budgets and deviation scores.

| File | Purpose |
|---|---|
| `timeline.py` | Assembles prediction CSVs into per-track frame-level timelines with a median-filter smoother |
| `budget.py` | Computes activity budgets, behavior transition matrices, and deviation from expected baselines |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `build_timeline` | `(group: DataFrame, fps: float) → DataFrame` | Converts a group of tubelet predictions for one track into a dense timeline |
| `run_timeline_analysis` | `(pred_csv, out_dir, fps, ...)` | End-to-end: reads `predictions/*.csv`, builds timelines, writes `results/analytics/timelines/` |
| `load_all_timelines` | `(timelines_dir: Path) → DataFrame` | Loads all per-video timeline CSVs into a single combined frame |
| `compute_activity_budget` | `(timelines: DataFrame) → DataFrame` | Fraction of time per behavior class per animal |
| `compute_transition_matrix` | `(timelines: DataFrame) → DataFrame` | Behavior-to-behavior transition counts |
| `compute_behavioral_deviation` | `(budget: DataFrame) → DataFrame` | Z-score deviation of each animal's budget from the herd mean |
| `run_budget_analysis` | `(timelines_dir, out_dir, ...)` | End-to-end: loads timelines → computes budget/transitions/deviation → writes CSVs |

Both modules can also be run as scripts: `python -m src.analytics.timeline --help`.

**Called by:** scripts `12_generate_analytics.sh` and `12b_generate_analytics.sh`.

---

#### `src/behavior/`

VideoMAE fine-tuning and inference for the 7-class behavior classifier.

| File | Purpose |
|---|---|
| `dataset.py` | `TubeletDataset` — PyTorch `Dataset` that reads 16-frame clip directories and applies VideoMAE pixel normalization |
| `train.py` | Full training loop: loads YAML config, builds `VideoMAEForVideoClassification`, runs train/val epochs, checkpoints best macro-F1 |
| `evaluate.py` | Loads a checkpoint, runs inference on the val/test split, writes predictions CSV and confusion matrix PNG |
| `classify.py` | Stateless inference function used by the Phase 9 pipeline — does not need a `labels.csv` split file |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `TubeletDataset` | `(labels_csv, split, cfg, augment)` | Dataset yielding `(pixel_values, label)` tensors; applies class-weighted sampler when `augment=True` |
| `classify_tubelets` | `(tubelet_rows, checkpoint, output_dir, job_id, ...)` | Runs VideoMAE inference on a list of tubelet dicts (from `export_tubelets_from_tracks`) and writes `predictions.csv`; used by `src/inference/pipeline.py` |

**YAML config keys consumed by `train.py`** (all in `configs/behavior/*.yaml`):
`model_name`, `num_classes`, `num_frames`, `frame_stride`, `img_size`, `batch_size`, `grad_accum_steps`, `epochs`, `lr`, `weight_decay`, `warmup_steps`, `use_amp`, `gradient_checkpointing`, `train_csv`, `val_csv`, `tubelets_dir`, `output_dir`.

**Called by:** `10_train_behavior.sh` / `10b_train_behavior.sh` (train), `11_evaluate.sh` / `11b_evaluate.sh` (evaluate), `src/inference/pipeline.py` (classify).

---

#### `src/data/`

Dataset preparation, label mapping, tubelet export, and COCO-format utilities.

| File | Purpose |
|---|---|
| `label_utils.py` | Canonical label map + dataset-specific converters; IoU helpers for detection matching |
| `export_tubelets.py` | Crops 16-frame clips from tracking JSON outputs and writes `data/processed/tubelets/` + `labels.csv` |
| `merge_coco.py` | Merges per-dataset COCO annotation JSONs into a single combined split |
| `validate_tubelets.py` | Sanity-checks `labels.csv`: verifies every tubelet directory exists and frame count is correct |
| `make_mini.py` | Samples a small reproducible subset of the detection dataset for rapid iteration |
| `convert_cbvd5.py` | Converts CBVD-5 raw annotations → COCO detection format |
| `convert_cvb.py` | Converts CVB annotations → COCO format; filters to frames with matching image files |
| `convert_cattleeyeview.py` | Converts CattleEyeView dataset → COCO |
| `convert_cows2021.py` | Converts Cows2021 → COCO |
| `convert_opencows2020.py` | Converts OpenCows2020 → COCO |
| `convert_freeman.py` | Converts Freeman Center annotations → COCO |
| `convert_mmcows_detection.py` | Converts mmCows bounding boxes → COCO (detection only; behavior path unused) |
| `convert_mmcows_tubelets.py` | Tubelet export for mmCows (implemented but never run — 1 fps makes it impractical) |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `cvb_behavior_to_label` | `(behavior_str) → int\|None` | Maps a CVB behavior string (e.g. `"Standing"`) to a canonical label ID |
| `cbvd5_actions_to_label` | `(action_ids: list[int]) → int` | Maps a list of CBVD-5 action IDs to a canonical label ID |
| `freeman_label_to_canonical` | `(freeman_id: int) → int` | Maps Freeman Center annotation IDs to canonical IDs |
| `match_predicted_to_gt` | `(preds, gt, iou_thresh)` | Greedy IoU matching of detection boxes to ground truth |
| `export_tubelets_from_tracks` | `(track_json, raw_video_dir, out_dir, ...)` | Reads a `*_tracks.json` file, crops 16-frame windows per track, writes frames + returns tubelet row dicts |
| `validate` | `(labels_csv)` | Returns `True` if all tubelets in the CSV are readable on disk |

**Called by:** `02_prepare_cbvd5.sh`, `03_prepare_cvb.sh`, `04_merge_datasets.sh`, `09_generate_tubelets.sh`, `09b_generate_tubelets.sh`, and OOD dataset prep scripts (13–19).

---

#### `src/detection/`

RF-DETR object detection: training and per-frame/per-video inference.

| File | Purpose |
|---|---|
| `train.py` | Fine-tunes RF-DETR on the combined detection dataset; reads a YAML config; supports resuming from checkpoint |
| `infer_dataset.py` | Runs the trained detector over all CBVD-5 and CVB frames and writes per-video detection JSONs to `data/processed/tracking/` |
| `infer_video.py` | Single-video inference with optional visualization overlay; also used by the inference pipeline |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `load_model` *(both files)* | `(checkpoint_path)` | Loads RF-DETR from a `.pth` checkpoint and returns the model in eval mode |
| `normalize_detections` | `(raw_output)` | Converts raw RF-DETR output to `[{bbox, score, label}]` dicts with absolute pixel coords |
| `infer_video` | `(model, video_path, out_dir, conf_thresh, ...)` | Runs detection frame-by-frame on a video, writes a detection JSON and optional annotated video |

**Called by:** `05_train_detector.sh` (train), `06_run_detection.sh` (infer_dataset), inference pipeline (infer_video).

---

#### `src/segmentation/`

Instance segmentation: SAM2 (v1 path) and RF-DETR-Seg (v2 path).

| File | Purpose |
|---|---|
| `segment.py` | SAM2 segmentation loop — takes detection boxes as prompts and produces per-frame mask outputs written to `data/processed/tracking_v2/` (v1 path) |
| `rfdetr_seg_infer.py` | RF-DETR-Seg single-stage inference — predicts boxes + masks in one pass, writes the same tracking_v2 JSON format (v2 path) |
| `mask_utils.py` | RLE encode/decode helpers shared by both segmentation modules and `src/tracking/track.py` |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `mask_to_rle` | `(binary_mask: ndarray) → dict` | Encodes a binary mask as COCO RLE (`{"counts": ..., "size": [H, W]}`) |
| `rle_to_mask` | `(rle: dict) → ndarray` | Decodes COCO RLE back to a boolean array |
| `mask_to_bbox` | `(binary_mask: ndarray) → list` | Returns tight `[x1, y1, x2, y2]` bounding box of a mask |
| `mask_area` | `(binary_mask: ndarray) → int` | Pixel count of a binary mask |
| `process_video` *(segment.py)* | `(cfg, video_id, dataset, predictor, ...)` | Runs SAM2 on all frames of one video, saves tracking_v2 JSON |

**Called by:** `07_run_segmentation.sh` (SAM2 / v1 path), `07b_run_rfdetr_seg.sh` (RF-DETR-Seg / v2 path).

---

#### `src/tracking/`

OC-SORT multi-object tracking, evaluation, and video rendering.

| File | Purpose |
|---|---|
| `track.py` | Main tracking driver — wraps OC-SORT with mask-IoU association; reads segmentation JSONs and writes `*_tracks.json` |
| `eval_tracking.py` | Computes IDF1, MOTA, MOTP, and other MOT metrics from track JSONs vs. CBVD-5 GT |
| `render_video.py` | Renders a tracking overlay video (colored boxes + track IDs) from a `*_tracks.json` |
| `render_behavior_video.py` | Renders a tracking + behavior label overlay video using predictions CSV; used for thesis figures |
| `visualize_tracks.py` | Exports a contact-sheet grid of track thumbnails for qualitative inspection |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `mask_iou_matrix` | `(masks_a, masks_b) → ndarray` | Pairwise IoU between two lists of binary masks; used as OC-SORT cost matrix |
| `track_video` | `(seg_json, out_json, min_hits, max_age, ...)` | Runs OC-SORT on one video's segmentation JSON and writes `*_tracks.json` |
| `run_tracking` | `(seg_dir, out_dir, cfg, ...)` | Batch wrapper: iterates all videos in a segmentation directory |
| `evaluate_video` | `(video_id, track_data, gt_frames, iou_thresh)` | Per-video MOT metrics (TP/FP/FN/ID switches) |
| `aggregate_metrics` | `(results: list) → dict` | Combines per-video metrics into IDF1, MOTA, MOTP |

> **OC-SORT dependency:** `track.py` inserts `third_party/OC_SORT` into `sys.path` at import time. Clone it before running script `08b_run_tracking_rfdetr.sh` (see §External Dependencies above).

**Called by:** `08b_run_tracking_rfdetr.sh` (track), tracking evaluation scripts.

---

#### `src/inference/`

End-to-end inference pipeline for new, unlabelled videos (Phase 9). Designed to run without ground-truth annotations and to be driven by the React/FastAPI web app.

| File | Purpose |
|---|---|
| `pipeline.py` | Orchestrates the 7-stage pipeline (ingest → detect → segment → track → tubelet → classify → analytics) and streams progress events via a callback |
| `run_inference.py` | CLI entry point (`python -m src.inference.run_inference --video ...`); reads a YAML config and calls `run_pipeline` |
| `video_ingestor.py` | `VideoIngestor` class — extracts frames from a video file, validates frame count, and populates a temp working directory |

**Key public API**

| Symbol | Signature | What it does |
|---|---|---|
| `run_pipeline` | `(config: dict, progress: ProgressCallback)` | Runs all 7 stages; emits progress dicts `{stage, stage_name, total_stages, frame, total_frames, status}` via callback |
| `VideoIngestor` | `VideoIngestor(video_path, work_dir, fps)` | Extracts frames using OpenCV; exposes `.frame_count`, `.fps`, `.frames_dir` |

`ProgressCallback = Callable[[Dict[str, Any]], None]` — the FastAPI backend passes a Server-Sent Events emitter here so the UI gets live updates.

**Called by:** `scripts/15_infer_video.sh` (CLI), `ui/backend/` (web app).

---

#### `src/tools/`

One-off evaluation, aggregation, and visualization scripts. Each file is a standalone runnable script (`python src/tools/<file>.py --help`), not a library module. They are not imported by any other `src/` module.

| File | Called by | Produces |
|---|---|---|
| `eval_detection_ood.py` | scripts `14` (Freeman), `19` (mmCows); also run ad-hoc for OpenCows2020, Cows2021, CattleEyeView (see `docs/datasets.md`) | `results/detection/{dataset}_eval.json` — COCO mAP@50 |
| `eval_maskiou_ood.py` | Ad-hoc (see `docs/datasets.md`) | `results/segmentation/cattleeyeview_maskiou.json` — Mask IoU = 86.5% |
| `eval_detection_perturbation.py` | Ad-hoc per dataset/condition | `results/generalization/perturbation_runs/{dataset}_{perturbation}_{severity}.json` |
| `aggregate_ood_summary.py` | Ad-hoc after all OOD evals complete | `results/generalization/ood_summary.csv` |
| `aggregate_perturbation.py` | Ad-hoc after all perturbation conditions complete | `results/generalization/perturbation_delta.csv` |
| `eval_rfdetr_seg.py` | Referenced in `07_train_rfdetr_seg.sh` echo; run manually on HiPE1 | Prints Mask IoU / AP to stdout |
| `eval_rfdetr_seg_kaggle.py` | Ad-hoc — cross-domain CattleEyeView/Kaggle segmentation eval | `results/segmentation/kaggle_eval_seg_lr5e5/predictions.json` + `gt_coco.json` |
| `sam2_to_coco_seg.py` | `07_train_rfdetr_seg.sh` | `data/rfdetr_seg/cattle/` — COCO-format segmentation dataset for RF-DETR-Seg training |
| `parse_v2_logs.py` | Ad-hoc on HiPE1 text logs | Extracts loss/mAP curves for figure generation |
| `plot_training_curves.py` | Ad-hoc | `results/rfdetr_seg/comparison/` — training curve PNGs + `comparison_table.csv` |
| `plot_training_results.py` | Ad-hoc (Config A vs B HiPE1 runs) | `plots/` — hyperparameter comparison bar charts |
| `write_cvb_idf1.py` | Ad-hoc | `results/tracking/cvb_idf1.json`, `results/tracking/cbvd5_idf1.json` |
| `write_minhits_ablation.py` | Ad-hoc | `results/tracking/minhits_ablation.csv` |

> **Note on ad-hoc tools:** OpenCows2020, Cows2021, and CattleEyeView OOD evals have no dedicated shell script wrapper. They were run directly from the command line following `docs/datasets.md`. The perturbation eval (Phase 9b) was also run ad-hoc — see `docs/design/reports/phase9_perturbation_report.md` for the exact commands used.

---

### tests/

Six pytest files. Run the full suite with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer && python -m pytest tests/ -v
```

No test requires GPU, real video files, or external network access. The inference pipeline test requires OC-SORT to be cloned (`third_party/OC_SORT`) and skips automatically if it is not.

| File | What it tests | Type |
|---|---|---|
| `test_label_utils.py` | `src/data/label_utils.py` — CVB behavior strings → label IDs, CBVD-5 action lists → label IDs (including multi-action priority rules), bounding box IoU, predicted-to-GT matching | Unit |
| `test_video_ingestor.py` | `src/inference/video_ingestor.py` — frame extraction from a synthetic video, metadata (`frame_count`, `fps`), error handling on missing files | Unit |
| `test_api.py` | `api/main.py` + `api/job_store.py` — `JobStore` state transitions (create, start, complete, fail), `/submit`, `/status`, `/results`, and SSE `/stream` HTTP endpoints via FastAPI `TestClient` with mocked pipeline | Unit + integration |
| `test_inference_pipeline.py` | `src/inference/pipeline.py` — full 7-stage pipeline on a synthetic video; RF-DETR-Seg and VideoMAE mocked; OC-SORT, tubelet export, analytics, and video render run real; verifies output files exist. Skips if `third_party/OC_SORT` not cloned. | Integration |
| `test_convert_mmcows_detection.py` | `src/data/convert_mmcows_detection.py` — YOLO → COCO bounding box coordinate conversion, annotation schema of the COCO output JSON | Unit |
| `test_convert_mmcows_tubelets.py` | `src/data/convert_mmcows_tubelets.py` — behavior CSV loading, label file parsing, label-map completeness, exclusion criteria (short clips, unlabelled frames), output row schema | Unit |

> **Note on mmCows tests:** `test_convert_mmcows_detection.py` and `test_convert_mmcows_tubelets.py` cover the scripts that were implemented but never run for the thesis (behavior eval was skipped at 1 fps). The tests still pass and remain in the suite as a correctness guard if the path is ever re-opened.

---

### third_party/

Holds external repositories that are cloned at setup time and are not pip-installable as standard packages. The directory is gitignored; nothing here is committed.

```
third_party/
└── OC_SORT/    ← git clone https://github.com/noahcao/OC_SORT.git
```

`src/tracking/track.py` inserts `third_party/OC_SORT` into `sys.path` at import time so that OC-SORT's internal modules resolve correctly. Clone it before running script `08b_run_tracking_rfdetr.sh` or `test_inference_pipeline.py`. See `docs/setup.md` for the exact command.

---

### ui/

React single-page application for the Phase 9 inference web app. Built with Vite. The FastAPI backend that serves it lives at `api/` (repo root), not inside `ui/`.

```
ui/
├── src/
│   ├── App.jsx           top-level router
│   ├── main.jsx          Vite entry point
│   ├── index.css
│   └── pages/
│       ├── UploadPage.jsx    video upload form → calls POST /submit
│       ├── ProgressPage.jsx  live progress bar → consumes GET /stream (SSE)
│       ├── ResultsPage.jsx   displays predictions, analytics, rendered video
│       └── HistoryPage.jsx   lists past inference jobs
├── public/               static assets
├── dist/                 production build output (served by FastAPI via StaticFiles)
├── index.html
├── vite.config.js
├── eslint.config.js
├── package.json
└── package-lock.json
```

**Backend (`api/`):**

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app — `/submit`, `/status/{job_id}`, `/stream/{job_id}` (SSE), `/results/{job_id}`, mounts `ui/dist/` as static files at `/` |
| `api/job_store.py` | In-memory `JobStore` — thread-safe job state machine (queued → running → complete / failed) |

**Development workflow:**

```bash
# Terminal 1 — React dev server (hot reload on :5173)
cd ui && npm install && npm run dev

# Terminal 2 — FastAPI backend (on :8000)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer
uvicorn api.main:app --reload

# Or launch both together via the convenience script:
bash scripts/start_app.sh
```

`start_app.sh` builds the React bundle (`npm run build` → `ui/dist/`) then starts uvicorn, which serves both the API and the built frontend from port 8000. See `docs/design/phase9_inference_tool_prd.md` for the original design spec.

---

### docs/

End-user documentation, operational runbooks, and all internal planning/design documents.
Subdirectories `adr/`, `agents/`, `dataset-papers/`, and `final-thesis-writing/` are out of scope for
this guide (ADRs are covered in `CONTEXT.md`; agents/ and dataset-papers/ are reference material;
final-thesis-writing/ is the LaTeX thesis source).

---

#### Root .md files

These are the primary reference docs a new student should read before running anything.

| File | Purpose | Freshness notes |
|---|---|---|
| `setup.md` | End-to-end install guide — conda env, OC-SORT clone, SAM2 install, all HuggingFace weight downloads, quickstart to skip GPU inference | **Primary starting point.** Weights download paths updated from stale `weights/` to `models/` (see below). |
| `datasets.md` | Full description of CBVD-5 and CVB datasets — annotation formats, label mappings, quirks, expected directory layout; Phase 8 OOD datasets (OpenCows2020, Cows2021, CattleEyeView, Freeman Center, mmCows) with download instructions and convert commands | Accurate and complete. |
| `pipeline.md` | Script-by-script walkthrough of all 12+ pipeline stages — prerequisites, expected runtimes, output paths, verification commands | Covers v1 SAM2 path and v2 RF-DETR-Seg path. |
| `docker.md` | Docker image catalog, build commands, per-stage `docker compose run` invocations, GPU config, shared memory, troubleshooting | Accurate. |
| `results.md` | Consolidated metrics across all phases — detection mAP, tracking IDF1/MOTA, behavior macro-F1 per config (v1 vs v2), OOD generalization, perturbation robustness | Accurate. Phase 3b checkpoint path corrected to `runs/seg_medium_lr5e5/checkpoint_best_ema.pth`. |
| `hipe_ops.md` | HiPE1/HiPE2 server setup — SSH config, conda activation, Docker on HiPE, RF-DETR-Seg 4-config hyperparameter grid and launch commands, VideoMAE training and retrieval commands | Historical runbook — all training is complete. Accurate for re-running if needed. |
| `hipe1_v2_training_guide.md` | Step-by-step guide for the v2 VideoMAE training run — rsync sync commands, 5-config Docker run templates, monitoring, evaluation (script 11b), analytics (script 12b) | Historical runbook — all v2 training complete. Useful if training must be re-run. |

**`weights/` → `models/` fix (done in this session):** The scripts use `models/rf-detr/` and
`models/sam2/` for pretrained backbone weights. All docs previously referenced a stale `weights/`
directory. Fixed in `setup.md`, `pipeline.md`, `docker.md`, `results.md`, `hipe_ops.md`, and
`docker/docker-compose.yml`.

---

#### design/ — Planning documents

`design/` holds PRDs and runbooks written before or during each phase. They are archived in-repo
because they record decisions, parameter choices, and scope tradeoffs that are not captured elsewhere.

**Root-level PRDs and runbooks:**

| File | Phase | Status | What it decided |
|---|---|---|---|
| `phase5_7_plan.md` | 5–7 | Historical | Tubelet generation → VideoMAE → analytics sequence; parameter choices (stride, 16-frame clips) |
| `phase7_cleanup_prd.md` | 7 | Historical (complete) | Post-phase-7 repo cleanup checklist — Steps A–I all done |
| `phase8_additional_datasets_prd.md` | 8 | Historical (complete) | OOD evaluation scope: 5 datasets, eval-only (no training), detection + segmentation metrics |
| `phase9_inference_tool_prd.md` | 9 | Historical (complete) | FastAPI + React inference web app design — endpoints, SSE streaming, job store |
| `dataset_analysis_notebooks_prd.md` | 0 | Historical (complete) | Dataset analysis notebook requirements (Issue #9) |
| `phase3b_pipeline_rerun_prd.md` | 3b | Historical (complete) | RF-DETR-Seg pipeline re-run scope using `seg_medium_lr5e5` checkpoint |
| `rfdetr_seg_pipeline_runbook.md` | 3b–12b | Historical (complete) | Step-by-step execution log for v2 parallel pipeline (scripts 07b–12b) |
| `thesis_data_completion_prd.md` | all | Historical (complete) | List of missing results files needed for thesis chapters — all now populated |
| `mmcows_ood_eval_prd.md` | 8 | Closed — detection done, behavior skipped | mmCows detection mAP = 32.4% ✅; behavior eval (Issues #41–#42) deferred (1 fps too slow) |
| `perturbation_eval_prd.md` | 9b | Historical (complete) | 60-condition perturbation robustness eval — 5 types × 2 severities × 6 datasets |
| `repo_cleanup_audit_prd.md` | — | **Active** (2026-06-17) | Current cleanup + defense prep checklist — ongoing |

**`design/reports/` — Phase completion reports:**

Each report documents the implementation decisions, parameters, and results for its phase.
They are the authoritative record for thesis sections and should be read before modifying any phase's code.

| File | Covers | Key contents |
|---|---|---|
| `phase0_report.md` | Phase 0 | Dataset analysis, 7-class label taxonomy design rationale |
| `phase1_detection_report.md` | Phase 1 | RF-DETR training setup, mAP results, CBVD-5 val-only early stopping note |
| `phase3_segmentation_report.md` | Phase 3a | SAM2 inference — mask counts, reprompt interval (15 frames), runtime |
| `phase3_rfdetr_seg_experiment_report.md` | Phase 3b | RF-DETR-Seg distillation experiment — 4-config grid, Config B winner, Kaggle eval |
| `phase4_tracking_report.md` | Phase 4 | OC-SORT — IDF1=67.31%, MOTA=36.61%, IoU threshold, per-video CSV |
| `phase5_tuebelet_report.md` | Phase 5 | Tubelet generation — 125,586 clips, stride choices, majority-vote label assignment |
| `phase6_behavior_classify_report.md` | Phase 6 | VideoMAE — all 5 config results, v1 macro-F1 table, per-class breakdown |
| `phase7_analytics_report.md` | Phase 7 | Activity budget, transition matrix, welfare flags — analytics module design |
| `phase8_ood_evaluation_report.md` | Phase 8 | OOD detection across 4 datasets, Freeman Center full pipeline run |
| `phase9_perturbation_report.md` | Phase 9b | 60-condition perturbation eval — brightness catastrophic, fog moderate, noise/blur/rain robust |
| `thesis_data_completion_report.md` | all | Consolidated metrics across all phases for thesis chapter writing |
| `thesis_audit_plan.md` | all | Pre-defense audit checklist — figures, tables, citation checks |
| `thesis_errors_to_fix.md` | all | Tracked errors/inconsistencies found during thesis writing (live doc) |
| `defense_prep_qa.md` | defense | Q&A practice for thesis defense committee questions |
| `inference_tracking_degradation.md` | 9 | Analysis of tracking quality degradation in Freeman Center inference run |

`design/reports/figures/` — committed figure assets referenced by the phase reports (PNG/SVG).
