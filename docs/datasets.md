# Datasets

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

---

## 1. CBVD-5 (Cattle Behavior Video Dataset — 5 classes)

### Overview

| Property          | Value                                                                        |
| ----------------- | ---------------------------------------------------------------------------- |
| Source            | Surveillance cameras in indoor dairy barn                                    |
| Videos            | ~687 short clips                                                             |
| Resolution        | Varies (typically 1920×1080)                                                 |
| Annotation type   | Bounding box + action label (AVA format, CSV)                                |
| Behaviors labeled | 5 (Standing, Lying, Foraging, Drinking, Ruminating)                          |
| Splits            | Train / Valid / Test (test = val — same video IDs, no separate test release) |
| Size              | ~12 GB                                                                       |

### Download

Obtain from the official CBVD-5 paper and repository. After downloading, place files at:

```
data/raw/cbvd5/
├── annotations/
│   └── *.csv           ← AVA-format bounding box + action labels
└── videos/
    └── *.mp4           ← raw video clips
```

**Verify:**

```bash
ls data/raw/cbvd5/videos/ | wc -l    # ~687 files
ls data/raw/cbvd5/annotations/ | wc -l
```

### Annotation Format

Each CSV row: `video_id, timestamp_sec, x1, y1, x2, y2, action_id, track_id`

Action IDs map to behaviors:

| CBVD-5 Action ID | Behavior       | Label ID       |
| ---------------- | -------------- | -------------- |
| 1                | stand          | 0 (Standing)   |
| 2                | lying down     | 1 (Lying)      |
| 3                | foraging       | 2 (Foraging)   |
| 4                | drinking water | 3 (Drinking)   |
| 5                | rumination     | 4 (Ruminating) |

**Multi-label priority:** CBVD-5 annotations are multi-label (one bbox can have multiple action IDs
on the same frame). Resolution: highest-priority label wins.

Priority order (descending): Drinking (3) > Foraging (2) > Ruminating (4) > Lying (1) > Standing (0)

Rationale: rarer behaviors carry more information and must not be suppressed by the dominant Standing class.

### Quirks

- **Test = validation:** CBVD-5 has no separate test set. The test split of `data/processed/detection/cbvd5/test/` contains the same video IDs as the validation split. Always report results on `--split val`.
- **Annotation density:** Not all frames of a video are annotated; annotations appear at keyframe intervals.

---

## 2. CVB (Cattle Video Behavior)

### Overview

| Property          | Value                                                    |
| ----------------- | -------------------------------------------------------- |
| Source            | Outdoor surveillance cameras at cattle farm              |
| Videos            | ~502 clips                                               |
| Resolution        | Varies                                                   |
| Annotation type   | Per-frame bounding box + behavior string                 |
| Behaviors labeled | 7 (all 7 canonical classes including Grooming and Other) |
| Splits            | Train / Valid (no test split)                            |
| Size              | ~15 GB                                                   |

### Download

Obtain from the official CVB paper and repository. After downloading:

```
data/raw/cvb/
├── annotations/
│   └── *.json          ← per-frame behavior annotations
└── raw_frames/
    └── {video_id}/
        └── frame_{idx:06d}.jpg   ← extracted frames
```

**Verify:**

```bash
ls data/raw/cvb/raw_frames/ | wc -l   # ~502 video directories
```

### Annotation Format

JSON with per-frame entries: `{frame_idx, track_id, bbox: [x,y,w,h], behavior: string}`

Behavior string mapping:

| CVB String            | Label ID       | Notes                          |
| --------------------- | -------------- | ------------------------------ |
| `resting-standing`    | 0 (Standing)   |                                |
| `resting-lying`       | 1 (Lying)      |                                |
| `grazing`             | 2 (Foraging)   | Equivalent to Foraging         |
| `drinking`            | 3 (Drinking)   |                                |
| `ruminating-standing` | 4 (Ruminating) | Activity only; posture ignored |
| `ruminating-lying`    | 4 (Ruminating) |                                |
| `grooming`            | 5 (Grooming)   | CVB-only class                 |
| `other`               | 6 (Other)      | CVB-only catch-all             |
| `hidden`              | **SKIP**       | Not a behavior                 |
| `walking`             | **SKIP**       | No CBVD-5 equivalent           |
| `running`             | **SKIP**       | < 1% of instances              |
| `none`                | **SKIP**       | Unlabeled                      |

### Quirks

- **Frame presence:** Not all annotated frames have corresponding image files in `raw_frames/`. `src/data/convert_cvb.py` filters to frames where the image file actually exists.
- **CVB frame presence threshold for tubelets:** A tubelet (16 frames) must have at least 12 of 16 frames present in a track to be included. This was relaxed from 16/16 to recover more training data.
- **Label assignment:** For tubelet clips (16 frames), behavior is assigned by majority vote over all frames with valid annotations.

---

## 3. Canonical 7-Class Taxonomy

Both datasets are unified under the same 7 label IDs, defined in `data/label_map.json` and `src/data/label_utils.py`.

| Label ID | Behavior         | CBVD-5 | CVB |
| -------- | ---------------- | ------ | --- |
| 0        | Standing         | ✓      | ✓   |
| 1        | Lying            | ✓      | ✓   |
| 2        | Foraging/Grazing | ✓      | ✓   |
| 3        | Drinking         | ✓      | ✓   |
| 4        | Ruminating       | ✓      | ✓   |
| 5        | Grooming         | —      | ✓   |
| 6        | Other            | —      | ✓   |

**Cross-dataset evaluation** uses only IDs 0–4. CBVD-5 contains no Grooming or Other annotations.

---

## 4. Tubelet Format

After running script 09, tubelets are stored as:

```
data/processed/tubelets/
├── labels.csv
├── cbvd5/
│   └── {video_id}/
│       └── {track_id}/
│           ├── frame_000000.jpg
│           ├── frame_000001.jpg
│           └── ...   (16 frames per tubelet)
└── cvb/
    └── {video_id}/
        └── {track_id}/
            └── frame_*.jpg
```

### labels.csv Schema

| Column        | Type | Description                                                                          |
| ------------- | ---- | ------------------------------------------------------------------------------------ |
| `dataset`     | str  | `cbvd5` or `cvb`                                                                     |
| `video_id`    | str  | numeric folder name (e.g., `341`)                                                    |
| `tubelet_dir` | str  | path to tubelet directory (e.g., `data/processed/tubelets/cbvd5/341/kf6_instc85ac7`) |
| `start_frame` | int  | first frame index in the original video                                              |
| `end_frame`   | int  | last frame index                                                                     |
| `label_id`    | int  | canonical behavior ID (0–6)                                                          |

`track_id` is encoded as the last component of `tubelet_dir` (e.g., `kf6_instc85ac7`).

### Tubelet Parameters (Frozen)

| Parameter          | Value                                          |
| ------------------ | ---------------------------------------------- |
| Frames per tubelet | 16                                             |
| Stride (CBVD-5)    | 4 frames                                       |
| Stride (CVB)       | 8 frames (50% overlap)                         |
| Crop size          | 224×224 px                                     |
| Crop padding       | 20 px around bbox, clamped to frame boundaries |
| Total tubelets     | 125,586                                        |

---

## 5. Phase 8 Additional Datasets (OOD Evaluation)

These datasets are used exclusively for out-of-distribution (OOD) evaluation of the
trained RF-DETR detector and RF-DETR-Seg model. They are **not** used for training.

---

### 5.1 OpenCows2020

| Property    | Value                                                     |
| ----------- | --------------------------------------------------------- |
| Source      | University of Bristol — outdoor farm, top-down cameras    |
| Images      | 7,039 (single test split — full dataset used as OOD eval) |
| Annotations | Bounding boxes (Supervisely JSON format)                  |
| Eval result | `results/detection/opencows2020_eval.json` — mAP@50=33.3% |
| Paper       | Andrew et al., 2021 (OpenCows2020)                        |

#### Download

```
# From Zenodo or the official OpenCows2020 repo; place at:
data/raw/opencows2020/
├── ann/      ← Supervisely JSON annotation files (*.jpg.json)
└── img/      ← JPEG images
```

#### Convert

```bash
python3 src/data/convert_opencows2020.py
# Output: data/processed/detection/opencows2020/_annotations.coco.json
```

---

### 5.2 Cows2021

| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Source        | University of Bristol — indoor barn, top-down surveillance      |
| Images (test) | 2,131 (test split used for OOD eval)                            |
| Annotations   | Bounding boxes (Supervisely JSON format, detection only)        |
| Eval result   | `results/detection/cows2021_eval.json` — mAP@50=27.3%           |
| Note          | No track IDs in annotations — tracking eval (IDF1) not possible |

#### Download

```
# From https://github.com/AnimalEyeQ/Cows2021 or DatasetNinja; place at:
data/raw/cows2021/
├── detection_and_localisation-train/
│   ├── ann/   ← *.jpg.json Supervisely annotations
│   └── img/   ← JPEG images
├── detection_and_localisation-val/
└── detection_and_localisation-test/
```

#### Convert

```bash
python3 src/data/convert_cows2021.py          # test split only (default)
python3 src/data/convert_cows2021.py --all_splits   # all three splits
# Output: data/processed/detection/cows2021/test/_annotations.coco.json
```

---

### 5.3 CattleEyeView

| Property      | Value                                                           |
| ------------- | --------------------------------------------------------------- |
| Source        | Singapore University of Technology and Design + AnimalEyeQ      |
| Videos        | 14 (top-down outdoor, 1920×1080)                                |
| Images (test) | 2,490 across 5 test videos                                      |
| Annotations   | YOLO-format bounding boxes + polygon instance masks             |
| Eval result   | `results/detection/cattleeyeview_eval.json` — mAP@50=47.0%      |
|               | `results/segmentation/cattleeyeview_maskiou.json` — Mask IoU    |
| Paper         | Ong et al., 2023 (VCIP) — doi:10.1109/VCIP59821.2023.10402676   |
| Note          | No track IDs in annotations — tracking eval (IDF1) not possible |

#### Download

```
# From https://github.com/AnimalEyeQ/CattleEyeView (follow repo instructions); place at:
data/raw/cattle-eye-view/
├── dataset/
│   ├── detect/
│   │   ├── images/{train,val,test}/{video_id}/*.jpg
│   │   └── labels/{train,val,test}/{video_id}/*.txt   ← YOLO bbox
│   └── segment/
│       └── labels/{train,val,test}/{video_id}/*.txt   ← YOLO polygon mask
└── README.md
```

#### Convert

```bash
python3 src/data/convert_cattleeyeview.py     # test split only (default)
# Output: data/processed/detection/cattleeyeview/test/_annotations.coco.json
#         (includes both bbox and polygon segmentation annotations)
```

#### Evaluate

```bash
# Detection mAP
python3 src/tools/eval_detection_ood.py \
    --checkpoint runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth \
    --dataset_dir data/processed/detection/cattleeyeview/test \
    --output results/detection/cattleeyeview_eval.json \
    --dataset_name cattleeyeview

# Mask IoU (RF-DETR-Seg)
python3 src/tools/eval_maskiou_ood.py \
    --checkpoint runs/seg_medium_lr5e5/checkpoint_best_ema.pth \
    --dataset_dir data/processed/detection/cattleeyeview/test \
    --output results/segmentation/cattleeyeview_maskiou.json
```

---

### 5.4 Freeman Center (CMB 2024)

| Property      | Value                                                                |
| ------------- | -------------------------------------------------------------------- |
| Source        | Freeman Ranch, Texas State University — angled outdoor ranch cameras |
| Videos        | 55 `.avi` clips                                                      |
| Images (test) | ~3,000 (test split of YOLO-format annotated stills)                  |
| Annotations   | YOLO-format bounding boxes (Roboflow export)                         |
| Eval result   | `results/detection/freeman_detection_eval.json` — mAP@50=73.0%      |
| Inference     | Full pipeline run on all 55 videos (Phase 9) — same pattern as script 26 |
| Note          | Closest domain match to CBVD-5 (angled surveillance); highest OOD AP |

#### Download

```
# Roboflow export; place at:
data/raw/freeman-cmb-2024/
├── CMB_config.txt
├── CMB_config.yml
├── CMB_dataset/CMB_dataset/
│   ├── train/
│   │   ├── images/*.jpg
│   │   └── labels/*.txt   ← YOLO bbox
│   ├── valid/
│   └── test/
└── freeman-raw-videos/
    └── *.avi              ← 55 raw video clips for inference
```

#### Convert & Evaluate

```bash
bash scripts/13_prepare_freeman.sh         # YOLO → COCO
bash scripts/14_eval_freeman_detection.sh  # OOD mAP eval
```

---

### 5.5 mmCows (NeurIPS 2024)

| Property      | Value                                                                          |
| ------------- | ------------------------------------------------------------------------------ |
| Source        | Purdue University / UW–Madison — indoor barn, top-down multi-camera rig        |
| Cameras       | 4 (cam_1–cam_4), date folder `0725`                                            |
| Images        | 5,040 per camera at ~1 fps (one frame every ~15 s)                             |
| Annotations   | YOLO-format bounding boxes (`labels/combined/0725/cam_4/`)                     |
| Eval result   | `results/detection/mmcows_eval.json` — mAP@50=32.4% (RF-DETR-Seg checkpoint)  |
| Inference     | Not run for thesis — 1 fps makes behavior outputs approximate; detection result is the authoritative OOD entry |
| Paper         | NeurIPS 2024 dataset track (Purdue/UW–Madison)                                 |
| Note          | 1 fps capture rate makes tubelet generation slow; behavior labels are approximate |

#### Download

```
# Place at:
data/raw/mmCows/
├── images/
│   └── 0725/
│       ├── cam_1/   ← 5040 .jpg frames
│       ├── cam_2/
│       ├── cam_3/
│       └── cam_4/
├── labels/
│   └── combined/
│       └── 0725/
│           └── cam_4/   ← 5040 YOLO .txt annotation files
├── behavior_labels/
├── proj_mat/
└── visual_location/
```

#### Convert & Evaluate

```bash
bash scripts/18_prepare_mmcows_detection.sh   # YOLO → COCO (cam_4 only)
bash scripts/19_eval_mmcows_detection.sh      # OOD mAP eval → results/detection/mmcows_eval.json
```

---

## 6. Processed Data Contracts

### Detection JSON (intermediate — not committed)

`data/processed/tracking/{dataset}/{video_id}_detections.json`

```json
{"frames": [{"frame_idx": 0, "detections": [{"bbox": [x, y, w, h], "score": 0.92}]}]}
```

### Tracking JSON (tracking_v2 — not committed, available on HuggingFace)

`data/processed/tracking_v2/{dataset}/{video_id}_tracks.json`

```json
{
  "frames": [
    {
      "frame_idx": 0,
      "tracks": [
        {
          "track_id": 1,
          "bbox": [x, y, w, h],
          "mask_rle": {"counts": "...", "size": [h, w]}
        }
      ]
    }
  ]
}
```

`mask_rle` is present only when script 07a (SAM2 segmentation) was run.
When script 08 (box-only tracking) was run instead, `mask_rle` is absent.

### Predictions CSV (committed — needed for analytics)

`results/behavior/predictions/{config}_val.csv`

| Column                | Description                         |
| --------------------- | ----------------------------------- |
| `dataset`             | `cbvd5` or `cvb`                    |
| `video_id`            | numeric video folder name           |
| `tubelet_dir`         | path (track_id is last component)   |
| `start_frame`         | tubelet start frame in source video |
| `end_frame`           | tubelet end frame                   |
| `label_id`            | ground truth behavior label (0–6)   |
| `pred_label_id`       | predicted behavior label            |
| `logit_0` … `logit_6` | raw model logits per class          |
