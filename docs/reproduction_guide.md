# Cattle Vision Framework — Full Reproduction Guide

**MS Thesis, Sakif Khan, Texas State University, 2026**

This guide walks you through reproducing the full project from scratch — from raw video
datasets to final behavior classification results. It is written for the person continuing
this project and assumes minimal prior knowledge of deep learning or computer vision.

If you are a reviewer or committee member, start with [§1 — Project Summary](#1-project-summary)
for a concise overview, then skip to [§9 — Final Results](#9-final-results) for the numbers.

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Key Concepts in Plain Language](#2-key-concepts-in-plain-language)
3. [What You Will Need](#3-what-you-will-need)
4. [Time and Hardware Expectations](#4-time-and-hardware-expectations)
5. [One-Time Environment Setup](#5-one-time-environment-setup)
6. [Pipeline Walkthrough — Phase by Phase](#6-pipeline-walkthrough--phase-by-phase)
   - [Phase 1: Prepare the Datasets](#phase-1-prepare-the-datasets-scripts-0104)
   - [Phase 2: Train the Cattle Detector](#phase-2-train-the-cattle-detector-script-05)
   - [Phase 3: Run Detection on All Videos](#phase-3-run-detection-on-all-videos-script-06)
   - [Phase 4: Segment and Track Each Cow](#phase-4-segment-and-track-each-cow)
   - [Phase 5: Generate Behavior Clips](#phase-5-generate-behavior-clips-scripts-09b)
   - [Phase 6: Train the Behavior Classifier](#phase-6-train-the-behavior-classifier-scripts-10b)
   - [Phase 7: Evaluate and Generate Analytics](#phase-7-evaluate-and-generate-analytics-scripts-11b-12b)
7. [Working on HiPE1 — Full Workflow](#7-working-on-hipe1--full-workflow)
8. [Where to Find Every Output File](#8-where-to-find-every-output-file)
9. [Final Results](#9-final-results)
10. [Skipping Long Steps — Using Pretrained Weights](#10-skipping-long-steps--using-pretrained-weights)

---

## 1. Project Summary

### What the system does

This project automatically recognizes what dairy cows are doing — standing, lying, eating,
drinking, ruminating, grooming — from surveillance camera footage, without any human
watching the video. Given a set of videos, the system outputs an **activity budget**: how
much time each cow spends on each behavior throughout the day.

This matters for livestock welfare monitoring. Abnormal behavior patterns (e.g., a cow that
stops eating) can indicate illness days before visible symptoms appear. Automating this
analysis at scale is the core contribution.

### The four-stage pipeline

```
Raw video
    │
    ▼
[1] DETECTION — RF-DETR model finds every cow in every frame and draws
    a bounding box around each one.
    │
    ▼
[2] SEGMENTATION + TRACKING — RF-DETR-Seg refines each bounding box into
    a precise pixel mask. OC-SORT links the same cow across frames over time,
    assigning a consistent ID to each animal throughout the video.
    │
    ▼
[3] BEHAVIOR CLASSIFICATION — Short video clips (tubelets) are cut out around
    each tracked cow and fed into VideoMAE, which classifies the behavior in
    each clip into one of 7 categories.
    │
    ▼
[4] ANALYTICS — Per-cow activity timelines and herd-level activity budgets
    are computed from the per-clip predictions.
```

### The two datasets

| Dataset | # Videos | Resolution | View       | Behaviors covered |
| ------- | -------- | ---------- | ---------- | ----------------- |
| CBVD-5  | 687      | varies     | Side-view  | 5 (Standing, Lying, Foraging, Drinking, Ruminating) |
| CVB     | 502      | varies     | Top-down   | 7 (above + Grooming, Other) |

### Why these models

| Stage       | Model      | Why chosen                                                     |
| ----------- | ---------- | -------------------------------------------------------------- |
| Detection   | RF-DETR    | State-of-the-art real-time detection; ResNet-50 backbone fits in 12 GB GPU |
| Segmentation| RF-DETR-Seg| Single-stage detector+segmentor distilled from SAM2; 10× faster than running SAM2 per-frame |
| Tracking    | OC-SORT    | Camera-motion robust; works with mask IoU (better than bbox IoU for occluded cows) |
| Classification | VideoMAE | Video transformer pretrained on large video datasets; handles 16-frame temporal window |

### The v1 vs v2 distinction

Two pipeline variants exist because the segmentation approach was improved mid-project:

- **v1 path** — uses SAM2 (a large foundation model) for per-frame segmentation. Accurate
  but very slow (~6 hours of GPU time just for segmentation). Used for comparison.
- **v2 path** — uses RF-DETR-Seg, a smaller model distilled from SAM2. 10× faster with
  comparable mask quality. **This is the primary thesis path** and produced the better
  results on Configs 1–4.

All scripts have a `b`-suffix twin for the v2 path (e.g., `09_generate_tubelets.sh` vs
`09b_generate_tubelets.sh`). **Follow the v2 path unless you specifically need v1.**

---

## 2. Key Concepts in Plain Language

You do not need to understand the math. You need to understand what each step produces.

**Model / neural network** — A program that has been trained to recognize patterns. Before
training it knows nothing; after training on thousands of examples, it can generalize to new
examples it has never seen.

**Training** — The process of adjusting a model's internal parameters by showing it many
labeled examples and correcting its mistakes. Produces a checkpoint file (`.pth` or `.pt`).
Training is slow (hours to days) and requires a GPU.

**Inference** — Running a trained model on new data to get predictions. Much faster than
training. Can sometimes run on CPU, but a GPU is still much faster.

**Checkpoint / weights** — A file that stores a trained model's learned parameters. If
someone has already trained the model for you and uploaded the checkpoint, you can skip
training and go straight to inference.

**Bounding box** — A rectangle drawn around an object in an image, defined by its four
corner coordinates.

**Instance mask** — A per-pixel map showing exactly which pixels belong to a specific cow,
as opposed to just a bounding box. More precise, especially when cows overlap.

**Tubelet** — A short video clip (here: 16 frames) cut from the original video, cropped
tightly around one tracked cow. Each tubelet gets a behavior label (its ground-truth class)
and a behavior prediction (what the model thinks the cow is doing).

**IoU (Intersection over Union)** — A number between 0 and 1 that measures how well two
boxes or masks overlap. Used in tracking to decide whether two detections in consecutive
frames are the same cow.

**Macro-F1** — The primary evaluation metric used in this project. It is the average F1
score across all behavior classes. F1 score combines precision (how often the model is right
when it makes a prediction) and recall (how often the model catches every real occurrence).
A macro-F1 of 1.0 is perfect; 0.0 is completely wrong. Values above 0.7 are considered good
for behavior recognition.

**Conda environment** — An isolated Python installation with a specific set of packages.
Using `conda activate cattletransformer` ensures all code uses the correct package versions.

---

## 3. What You Will Need

### Hardware

| Component | Minimum          | Used in this thesis |
| --------- | ---------------- | ------------------- |
| GPU (local) | NVIDIA, 12 GB VRAM | RTX 3060 12 GB (dubuntu) |
| GPU (training) | NVIDIA, 16 GB VRAM | 2× Tesla V100 (HiPE1) |
| RAM | 32 GB | — |
| Disk | 300 GB free | ~200 GB data + ~100 GB processed |

A 12 GB local GPU is enough for everything except VideoMAE training and RF-DETR-Seg training
(those require the HiPE1 cluster). All other steps run on the local machine.

### Software

- Linux (Ubuntu 22.04 tested)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Mamba
- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker ≥ 24
- Git, `wget`, `rsync`

### Accounts

- **Texas State HiPE cluster** — SSH access via your advisor (Prof. Jiang's group, `zxs12` account)
- **HuggingFace** — Free account at huggingface.co. Needed to download pretrained weights
  from `sakifkhan98/cattle-vision-framework`

### Datasets (download separately)

The raw video datasets are not in the repository. See `docs/datasets.md` for exact download
links and instructions.

| Dataset | Size   | Where to put it          |
| ------- | ------ | ------------------------ |
| CBVD-5  | ~12 GB | `data/raw/cbvd5/`        |
| CVB     | ~15 GB | `data/raw/cvb/`          |

---

## 4. Time and Hardware Expectations

Before you start, here is what each phase actually takes. "Local" means your own machine
with a GPU. "HiPE1" means you must SSH to the cluster.

| Phase | What happens | Where | Approx time |
| ----- | ------------ | ----- | ----------- |
| Environment setup | Install packages, clone deps, download weights | Local | 1–2 hours |
| Data prep (scripts 01–04) | Convert annotations, create splits | Local, no GPU | 30–60 min |
| Train detector (05) | Fine-tune RF-DETR | Local GPU | 3–6 hours |
| Detection inference (06) | Run detector on all videos | Local GPU | 3–5 hours |
| RF-DETR-Seg inference (07b + 08b) | Segment + track all videos | Local GPU | 4–8 hours |
| Tubelet generation (09b) | Slice 137k clips from tracked cows | Local, no GPU | 3–6 hours |
| Sync data to HiPE1 | rsync ~40 GB tubelets | Network | 1–3 hours |
| VideoMAE training (10b) | Train 5 behavior configs | HiPE1, 2× V100 | 5–15 hours total |
| Retrieve results from HiPE1 | rsync checkpoints back | Network | 30 min |
| Evaluation (11b) | Run inference on all val clips | Local GPU | 1–2 hours |
| Analytics (12b) | Compute timelines and budgets | Local, no GPU | < 5 min |

**Total wall-clock time for full v2 reproduction:** roughly 2–3 days including data
download, most of which is waiting for GPU jobs to finish.

**If you download pretrained weights** (see §10), you can reproduce the final results
(steps 11b and 12b) in under 2 hours without any cluster access.

---

## 5. One-Time Environment Setup

Do these steps once when you first set up the project on a new machine.

### Step 1 — Clone the repository

```bash
git clone https://github.com/sakifkhan98/cattle-vision-framework.git
cd cattle-vision-framework
```

Everything in this guide runs from this directory (the repo root) unless stated otherwise.

### Step 2 — Create the Python environment

This project uses a specific set of Python packages. The `cattletransformer` conda
environment has everything pre-specified:

```bash
conda env create -f environment.yml
conda activate cattletransformer
```

**Every time you open a new terminal**, activate the environment before running any code:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer
```

Verify it worked:

```bash
python -c "from src.data.label_utils import BEHAVIOR_NAMES; print(BEHAVIOR_NAMES)"
# Expected: ['Standing', 'Lying', 'Foraging/Grazing', 'Drinking', 'Ruminating', 'Grooming', 'Other']
```

### Step 3 — Install OC-SORT (the tracking library)

OC-SORT is a tracking algorithm that is not in PyPI, so it must be cloned manually:

```bash
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
```

This directory is gitignored. The tracking script automatically finds it at runtime.

### Step 4 — Install SAM2 (only needed for the v1 path)

SAM2 is Meta's segmentation model. You only need this if you want to reproduce the v1
results; the v2 path uses the faster RF-DETR-Seg model instead.

```bash
pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'
mkdir -p models/sam2/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -O models/sam2/sam2.1_hiera_large.pt
```

### Step 5 — Download the raw datasets

Follow the instructions in `docs/datasets.md` to download CBVD-5 and CVB into:

```
data/raw/cbvd5/    ← CBVD-5 videos and annotations
data/raw/cvb/      ← CVB videos and annotations
```

### Step 6 — Download the RF-DETR backbone weight

This is the COCO-pretrained starting point for the cattle detector. Fine-tuning starts
from this file (it is not trained on cattle — you fine-tune it on your cattle data in Phase 2):

```bash
pip install huggingface-hub
mkdir -p models/rf-detr/
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-medium.pth \
  --local-dir models/rf-detr/
```

### Step 7 — Verify your GPU is working with Docker

The HiPE1 training steps use Docker containers. Even locally, some pipeline stages can
run via Docker. Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Should print a GPU table. If it errors, see docs/docker.md §1.
```

---

## 6. Pipeline Walkthrough — Phase by Phase

### Phase 1: Prepare the Datasets (Scripts 01–04)

**What happens here:** The raw CBVD-5 and CVB datasets store their annotations in
different formats. These scripts convert both into the standard COCO format that the
detector training code expects, and merge them into a single combined training dataset.

**No GPU needed. Runs on local machine. Takes ~30–60 minutes total.**

#### Script 01 — Sanity check (< 1 minute)

Before processing anything, check that both datasets downloaded correctly:

```bash
bash scripts/01_inspect_data.sh
```

This prints video counts and annotation counts. No files are written — it is a read-only check.

#### Script 02 — Convert CBVD-5 annotations (~5–10 minutes)

```bash
bash scripts/02_prepare_cbvd5.sh
```

Converts AVA-format CSV annotations to COCO format with train/valid/test splits.

**What you get:** `data/processed/detection/cbvd5/{train,valid,test}/_annotations.coco.json`
plus symbolic links to the actual image files.

**Note:** CBVD-5 has no separate test set — the test split uses the same videos as the
validation split. This is a known limitation of the dataset.

#### Script 03 — Convert CVB annotations (~10–20 minutes)

```bash
bash scripts/03_prepare_cvb.sh
```

**What you get:** `data/processed/detection/cvb/{train,valid}/_annotations.coco.json`

#### Script 04 — Merge into a combined dataset (< 2 minutes)

```bash
bash scripts/04_merge_datasets.sh
```

Merges both datasets into a single directory for joint detector training.

**What you get:** `data/processed/detection/combined/{train,valid}/_annotations.coco.json`

**Verify all three scripts completed:**

```bash
python -c "
import json
for ds in ['cbvd5', 'cvb']:
    for split in ['train', 'valid']:
        try:
            d = json.load(open(f'data/processed/detection/{ds}/{split}/_annotations.coco.json'))
            print(f'{ds}/{split}: {len(d[\"images\"])} images, {len(d[\"annotations\"])} annotations')
        except FileNotFoundError:
            print(f'MISSING: {ds}/{split}')
"
```

---

### Phase 2: Train the Cattle Detector (Script 05)

**What happens here:** We fine-tune RF-DETR — a real-time object detection model — to
find cows in video frames. Starting from weights pre-trained on the COCO dataset (which
includes cows as one of 80 categories), we further train it on our specific cattle footage
so it gets much better at finding cows in farm camera views.

**Requires a GPU (≥12 GB). Takes 3–6 hours on an RTX 3060.**

```bash
bash scripts/05_train_detector.sh
```

**What you get:** `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`

This checkpoint file contains the trained detector. It is the only file you need from
this phase — everything else the training script writes (intermediate checkpoints, logs)
can be ignored.

**What to watch for:** The training log prints mAP@50 after each epoch. You want this
number to increase over the first ~20 epochs and then plateau. If it never improves,
something is wrong with the data paths.

**Results achieved in this thesis:**
- mAP@50 on CBVD-5 test: **45.9%**
- mAP@50 on CVB test: **5.7%** (the model was validated on CBVD-5 only — CVB is treated as out-of-domain)

**Skip this step** by downloading the pretrained detector (see §10). This is recommended
if you just want to reproduce the behavior classification results.

---

### Phase 3: Run Detection on All Videos (Script 06)

**What happens here:** We run the trained detector on every frame of every video in both
datasets. The output is a JSON file per video listing the bounding boxes of every cow
detected in every frame.

**Requires a GPU. Takes 3–5 hours total.**

```bash
bash scripts/06_run_detection.sh
```

**What you get:** `data/processed/tracking/{cbvd5,cvb}/{video_id}_detections.json`
(one JSON file per video, ~1,189 files total)

**Verify:**

```bash
ls data/processed/tracking/cbvd5/ | wc -l   # ~687
ls data/processed/tracking/cvb/   | wc -l   # ~502
```

**Skip this step** by downloading pre-computed tracking outputs (see §10).

---

### Phase 4: Segment and Track Each Cow

This phase has two sub-steps: **segmentation** (finding the precise pixel outline of each
cow) and **tracking** (linking the same cow across frames over time).

We use the **v2 path** (RF-DETR-Seg). If you want to run the v1 path (SAM2) for
comparison, see `docs/pipeline.md §Script 07a`.

#### Sub-step 4a — RF-DETR-Seg inference (Script 07b)

**What happens here:** The RF-DETR-Seg model takes the detection bounding boxes from
Phase 3 and adds precise segmentation masks. It produces per-frame JSON files that
include both the bounding box and the pixel-level mask for every detected cow.

RF-DETR-Seg was trained on HiPE1 using SAM2 predictions as teacher labels (this is
called knowledge distillation — we taught a smaller, faster model to mimic a larger
one). You do not need to re-train it; just download the checkpoint.

**Download the RF-DETR-Seg checkpoint** (if not already done):

```bash
mkdir -p runs/seg_medium_lr5e5/
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-seg-medium.pt \
  --local-dir runs/seg_medium_lr5e5/
mv runs/seg_medium_lr5e5/rf-detr-seg-medium.pt \
   runs/seg_medium_lr5e5/checkpoint_best_ema.pth
```

**Run inference:**

```bash
bash scripts/07b_run_rfdetr_seg.sh
```

Requires GPU, takes 4–8 hours.

**What you get:** `data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json`
(temporary format — masks added but no track IDs yet)

#### Sub-step 4b — OC-SORT tracking (Script 08b)

**What happens here:** OC-SORT is a tracking algorithm that connects detections across
frames. It answers the question: "Is the cow in box A in frame 100 the same cow that
was in box B in frame 99?" It assigns each cow a persistent track ID that stays the
same throughout the video.

This project uses mask IoU for association — instead of comparing the overlap of
bounding boxes, it compares the overlap of the pixel masks. This is more accurate when
cows are close together or partially occluded.

```bash
bash scripts/08b_run_tracking.sh
```

No GPU needed. Takes ~1–2 hours.

**What you get:** `data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json`
(updated in-place — now includes track IDs)

Each JSON file looks like:
```json
{
  "video_id": "...",
  "frames": {
    "1": [{"track_id": 11, "bbox": [x1,y1,x2,y2], "mask_rle": {...}}],
    "2": [{"track_id": 11, "bbox": [...], "mask_rle": {...}}, {"track_id": 12, ...}]
  }
}
```

**Skip both 4a and 4b** by downloading pre-computed tracking outputs (see §10).

---

### Phase 5: Generate Behavior Clips (Scripts 09b)

**What happens here:** Now that we know where each cow is in each frame (tracking_v2),
we cut out short video clips — called **tubelets** — centered on each tracked cow.
Each tubelet is 16 frames long (4 seconds at 4 fps), 224×224 pixels, cropped and
padded around the cow's bounding box.

Every tubelet is also assigned a **behavior label** (0–6) by matching the clip's frames
against the ground-truth annotations in the dataset.

These tubelets are what the behavior classifier (Phase 6) is trained on.

```bash
bash scripts/09b_generate_tubelets.sh
```

No GPU needed. Takes 3–6 hours (purely disk I/O — reading frames, resizing, writing JPEG files).

**What you get:**

```
data/processed/tubelets_rfdetr/
    cbvd5/
        video_001/
            track_11/
                frame_0000.jpg  frame_0001.jpg  ... frame_0015.jpg  ← one tubelet
            track_11/     ← next 16-frame window
                ...
    cvb/
        ...
    labels.csv   ← one row per tubelet: dataset, video_id, path, start_frame, end_frame, label_id
```

**Verify:**

```bash
wc -l data/processed/tubelets_rfdetr/labels.csv
# Expected: 137596 (137595 tubelets + 1 header row)
```

---

### Phase 6: Train the Behavior Classifier (Scripts 10b)

**What happens here:** We fine-tune VideoMAE — a video understanding model pre-trained
on Kinetics-400 (a large human action recognition dataset) — to classify cow behaviors.
We train 5 separate models for 5 different experimental configurations:

| Config | Trains on | Evaluates on | Purpose |
| ------ | --------- | ------------ | ------- |
| combined | CBVD-5 + CVB | CBVD-5 + CVB | Best overall accuracy |
| cvb | CVB only | CVB | In-domain CVB performance |
| cbvd5 | CBVD-5 only | CBVD-5 | In-domain CBVD-5 performance |
| cbvd5_to_cvb | CBVD-5 | CVB | Cross-domain: can CBVD-5 knowledge transfer to CVB? |
| cvb_to_cbvd5 | CVB | CBVD-5 | Cross-domain: can CVB knowledge transfer to CBVD-5? |

**This phase runs on HiPE1 (the university cluster).** The VideoMAE model requires
16 GB of GPU memory per training run, which exceeds the local RTX 3060's 12 GB.

See §7 below for the complete HiPE1 workflow. A summary of what happens:
1. You sync the tubelets and config files to HiPE1 over SSH
2. You launch 5 Docker training jobs (2 at a time, one per GPU)
3. Each job runs for 10–20 epochs (~1–3 hours per config)
4. You sync the checkpoint files back to your local machine

**What you get (after retrieving from HiPE1):**

```
runs/behavior/
    videomae_combined_v2/checkpoint_best.pt
    videomae_cvb_v2/checkpoint_best.pt
    videomae_cbvd5_v2/checkpoint_best.pt
    videomae_cbvd5_to_cvb_v2/checkpoint_best.pt
    videomae_cvb_to_cbvd5_v2/checkpoint_best.pt
```

**Skip this step** by downloading pretrained v2 checkpoints (see §10). This is the
most time-consuming step and is safe to skip for reproduction purposes.

---

### Phase 7: Evaluate and Generate Analytics (Scripts 11b, 12b)

**What happens here:** We run the trained VideoMAE models on the validation set of each
config and measure how well they perform. Then we use the predictions to compute
activity budgets and timelines.

**Both steps run locally. No cluster needed. Takes ~2 hours total.**

#### Evaluation (Script 11b)

```bash
bash scripts/11b_evaluate_rfdetr.sh
```

This runs all 5 v2 checkpoints on their respective validation sets and writes:

- **Prediction CSVs** — one row per tubelet, with the true label and predicted label
- **Confusion matrix PNGs** — visual breakdown of which behaviors get confused with which
- **f1_per_class.csv** — per-class and macro-F1 for all configs

**Verify:**

```bash
grep v2 results/behavior/f1_per_class.csv
```

#### Analytics (Script 12b)

```bash
bash scripts/12b_generate_analytics_rfdetr.sh
```

Uses the `combined_v2` predictions together with the tracking data to build per-cow
activity timelines and herd-level activity budgets.

**What you get:**

```
results/analytics/
    activity_budget_v2.csv     ← % time each behavior per cow
    transition_matrix_v2.csv   ← how often cows switch between behaviors
    welfare_flags_v2.csv       ← cows showing unusual patterns
```

---

## 7. Working on HiPE1 — Full Workflow

HiPE1 is a Texas State GPU cluster with 2× Tesla V100 GPUs (16 GB each). You use it
for Phase 6 (VideoMAE training). Phase 3b (RF-DETR-Seg training) was also done on HiPE1,
but that checkpoint already exists — you do not need to retrain it.

### 7.1 SSH access

You need credentials for the `zxs12` account. Set up SSH shortcuts once:

Add to `~/.ssh/config` on your local machine:

```
Host leap2
    HostName leap2.txstate.edu
    User zxs12
    ServerAliveInterval 60

Host hipe1
    HostName hipe1.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    ServerAliveInterval 60
```

Test: `ssh hipe1 "nvidia-smi"` — should show two V100 GPUs.

### 7.2 What to send to HiPE1

After Phase 5 (tubelet generation), send the following to HiPE1. Run all commands from
your local machine:

```bash
# Create the directory layout on HiPE1
ssh hipe1 "mkdir -p ~/cattle_behavior/{data/processed/tubelets_rfdetr,configs/behavior,src,runs/behavior,logs}"

# Send source code and configs (fast)
rsync -avz --progress src/ configs/ hipe1:~/cattle_behavior/

# Send the tubelets (~33–40 GB — this is the slow step, takes 1–3 hours over VPN)
rsync -avz --progress \
    data/processed/tubelets_rfdetr/ \
    hipe1:~/cattle_behavior/data/processed/tubelets_rfdetr/

# Verify the tubelet count arrived correctly
ssh hipe1 "wc -l ~/cattle_behavior/data/processed/tubelets_rfdetr/labels.csv"
# Expected: 137596
```

rsync is resumable — if your VPN drops, re-run the exact same command and it picks up
where it left off. Never use `scp` for large transfers; rsync is strictly better.

### 7.3 Load the Docker image on HiPE1

The behavior training runs inside a Docker container to avoid environment conflicts on
the shared cluster. Transfer and load it once:

```bash
# Build the image locally and save it
docker build -f docker/Dockerfile.behavior -t cattle-videomae:v1 .
docker save cattle-videomae:v1 | gzip > cattle-videomae-v1.tar.gz

# Transfer to HiPE1 (~4–5 GB)
rsync -avz --progress cattle-videomae-v1.tar.gz hipe1:~/cattle_behavior/

# Load on HiPE1
ssh hipe1 "cd ~/cattle_behavior && docker load < cattle-videomae-v1.tar.gz"

# Verify GPU is accessible inside Docker
ssh hipe1 "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
```

### 7.4 Launch training (5 configs in 3 batches)

HiPE1 has 2 GPUs. You run 2 configs at a time, using tmux so the jobs keep running
if your SSH session drops.

```bash
ssh hipe1
cd ~/cattle_behavior
tmux new-session -s behavior_v2
```

tmux is a terminal multiplexer. Think of it as a session that keeps running on the
server even after you disconnect. Key shortcuts:
- `Ctrl+B c` — open a new window
- `Ctrl+B 0` / `Ctrl+B 1` — switch between windows
- `Ctrl+B d` — detach (leave the session running and go back to your laptop)
- `tmux attach -t behavior_v2` — reattach later

**Batch 1 — start two jobs simultaneously:**

Window 0 (GPU 0):
```bash
docker run --rm --gpus '"device=0"' --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    src/behavior/train.py --config configs/behavior/videomae_combined_v2.yaml \
    2>&1 | tee logs/combined_v2.log
```

Press `Ctrl+B c`, then in Window 1 (GPU 1):
```bash
docker run --rm --gpus '"device=1"' --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    src/behavior/train.py --config configs/behavior/videomae_cvb_v2.yaml \
    2>&1 | tee logs/cvb_v2.log
```

Press `Ctrl+B d` to detach. The jobs continue running.

**Batches 2 and 3:** When a window finishes (returns to the shell prompt), reattach
and launch the next config in that window on the same GPU. The full order is:

| Batch | GPU 0 | GPU 1 |
| ----- | ----- | ----- |
| 1 | combined_v2 | cvb_v2 |
| 2 | cbvd5_v2 | cbvd5_to_cvb_v2 |
| 3 | cvb_to_cbvd5_v2 | (idle) |

For the remaining 3 configs, use the same docker run command template but swap in
the config file name and log file name. See `docs/hipe_ops.md §7.5–7.6` for the
exact commands.

### 7.5 Monitoring jobs from your laptop

You do not need to stay logged in to check progress:

```bash
# Check the latest training epoch
ssh hipe1 "tail -3 ~/cattle_behavior/logs/combined_v2.log"
ssh hipe1 "tail -3 ~/cattle_behavior/logs/cvb_v2.log"

# Check which containers are still running
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Check GPU utilization
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
```

A typical log line looks like:
```
Epoch 8 | lr=5.00e-05 | train_loss=0.42 train_f1=0.81 | val_loss=0.91 val_macro_f1=0.77
  -> new best val_macro_f1=0.7770
```

Training stops automatically when the validation F1 hasn't improved for 8 consecutive
epochs (early stopping). Each config takes 10–20 epochs, roughly 1–3 hours per config.

### 7.6 Retrieve results from HiPE1

Once all 5 configs finish, pull the checkpoints and logs back to your local machine:

```bash
mkdir -p runs/behavior/ logs/

# Checkpoints
rsync -avz hipe1:~/cattle_behavior/runs/behavior/ runs/behavior/

# Training logs
rsync -avz hipe1:~/cattle_behavior/logs/ logs/

# Verify all 5 checkpoints arrived
for cfg in combined cvb cbvd5 cbvd5_to_cvb cvb_to_cbvd5; do
    ckpt="runs/behavior/videomae_${cfg}_v2/checkpoint_best.pt"
    [ -f "$ckpt" ] && echo "OK   $ckpt" || echo "MISSING  $ckpt"
done
```

You now have everything you need to run evaluation locally.

---

## 8. Where to Find Every Output File

This table maps each pipeline phase to the files it produces and where they are committed
in the repository (or why they are gitignored).

| Phase | Key output | Path | In git? |
| ----- | ---------- | ---- | ------- |
| Data prep | COCO annotation JSONs | `data/processed/detection/` | No (regenerate) |
| Detector training | Best checkpoint | `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth` | No (too large; HuggingFace) |
| Detection inference | Per-video detection JSONs | `data/processed/tracking/` | No (regenerate) |
| Segmentation + tracking | Per-video track JSONs | `data/processed/tracking_v2/` | No (HuggingFace dataset) |
| Tubelet generation | JPEG clips + labels.csv | `data/processed/tubelets_rfdetr/` | No (too large) |
| VideoMAE training | Best checkpoints (×5) | `runs/behavior/videomae_*_v2/checkpoint_best.pt` | No (HuggingFace) |
| Evaluation | Prediction CSVs | `results/behavior/predictions/` | **Yes** |
| Evaluation | Confusion matrix PNGs | `results/behavior/confusion_matrices/` | **Yes** |
| Evaluation | F1 per class | `results/behavior/f1_per_class.csv` | **Yes** |
| Analytics | Activity budget CSV | `results/analytics/activity_budget_v2.csv` | **Yes** |
| Training logs | HiPE1 training logs | `logs/train_*_v2.log` | **Yes** |

Files in **bold Yes** are committed and do not need to be regenerated. You can examine
the full evaluation results immediately after cloning the repo.

---

## 9. Final Results

### Detection

| Dataset | mAP@50 | Notes |
| ------- | ------ | ----- |
| CBVD-5 test | 45.9% | Primary in-domain benchmark |
| CVB test | 5.7% | Out-of-domain (model selected on CBVD-5 val only) |

### Tracking (on CBVD-5)

| Metric | Score |
| ------ | ----- |
| IDF1 | 67.31% |
| MOTA | 36.61% |

IDF1 measures how consistently a tracker assigns the same ID to the same cow over time.
MOTA combines missed detections, false alarms, and ID switches into one number.

### Behavior Classification (macro-F1)

The v2 RF-DETR-Seg results are the primary thesis numbers. Higher is better; 1.0 is perfect.

| Config | Train → Val | v1 (SAM2) | v2 (RF-DETR-Seg) | Primary |
| ------ | ----------- | --------- | ---------------- | ------- |
| 1: CBVD-5 in-domain | CBVD-5 → CBVD-5 | 0.3149 | **0.4511** | ★ v2 |
| 2: CVB in-domain | CVB → CVB | 0.7607 | **0.7770** | ★ v2 |
| 3: Cross-domain | CBVD-5 → CVB | 0.1690 | **0.1722** | ★ v2 |
| 4: Cross-domain | CVB → CBVD-5 | 0.1789 | **0.2253** | ★ v2 |
| 5: Combined | both → both | **0.7537** | 0.7507 | ★ v1 |

Key observations:
- Config 2 (CVB in-domain) achieves the best absolute performance (0.7770 macro-F1), showing the system works well when trained and tested on the same camera view
- Config 1 improvement from v1 to v2 (+0.136 macro-F1) is the largest gain — better segmentation masks help most when the CBVD-5 footage is crowded
- Cross-domain configs (3 and 4) are much lower, showing that camera viewpoint (side-view vs top-down) is still a major generalization challenge
- Config 5 (combined) is the only case where v1 marginally beats v2; this is within noise

Detailed per-class breakdown is in `results/behavior/f1_per_class.csv`.

---

## 10. Skipping Long Steps — Using Pretrained Weights

If you want to reproduce just the final evaluation results without re-running the entire
pipeline (which takes 2–3 days), download the pretrained outputs from HuggingFace.

### Minimum to reproduce final behavior results (Steps 11b + 12b only)

You need: the tubelets, the v2 checkpoints, and the tracking_v2 data.

```bash
pip install huggingface-hub

# 1. Download pre-computed tracking outputs (skips scripts 06, 07b, 08b)
mkdir -p data/processed/tracking_v2/
huggingface-cli download sakifkhan98/cattle-vision-data \
  tracking_v2_cbvd5.tar.gz tracking_v2_cvb.tar.gz \
  --repo-type dataset --local-dir /tmp/tracking_v2/
tar -xf /tmp/tracking_v2/tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2/
tar -xf /tmp/tracking_v2/tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2/

# 2. Run tubelet generation (still needed — this is CPU only, takes a few hours)
bash scripts/09b_generate_tubelets.sh

# 3. Download all 5 v2 VideoMAE checkpoints (skips HiPE1 training)
for cfg in combined cvb cbvd5 cbvd5_to_cvb cvb_to_cbvd5; do
    mkdir -p runs/behavior/videomae_${cfg}_v2/
    huggingface-cli download sakifkhan98/cattle-vision-framework videomae_${cfg}_v2.pt \
        --local-dir runs/behavior/videomae_${cfg}_v2/
    mv runs/behavior/videomae_${cfg}_v2/videomae_${cfg}_v2.pt \
       runs/behavior/videomae_${cfg}_v2/checkpoint_best.pt
done

# 4. Run evaluation
bash scripts/11b_evaluate_rfdetr.sh

# 5. Run analytics
bash scripts/12b_generate_analytics_rfdetr.sh
```

### Reproducing from detection onwards (skipping only re-training the detector)

```bash
# Download the cattle detector checkpoint
mkdir -p runs/detection/rfdetr_combined_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework rfdetr_combined_v1_best.pth \
  --local-dir runs/detection/rfdetr_combined_v1/
mv runs/detection/rfdetr_combined_v1/rfdetr_combined_v1_best.pth \
   runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth

# Download the RF-DETR-Seg checkpoint
mkdir -p runs/seg_medium_lr5e5/
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-seg-medium.pt \
  --local-dir runs/seg_medium_lr5e5/
mv runs/seg_medium_lr5e5/rf-detr-seg-medium.pt \
   runs/seg_medium_lr5e5/checkpoint_best_ema.pth

# Then run from Phase 3 onwards:
bash scripts/06_run_detection.sh       # ~4 hours GPU
bash scripts/07b_run_rfdetr_seg.sh     # ~4 hours GPU
bash scripts/08b_run_tracking.sh       # ~1 hour CPU
bash scripts/09b_generate_tubelets.sh  # ~4 hours CPU
# Then HiPE1 for training, or download v2 checkpoints and skip to evaluation
```

---

## Quick Checklist for Full Reproduction

```
Setup
  [ ] conda env created and activates correctly
  [ ] OC-SORT cloned to third_party/OC_SORT/
  [ ] data/raw/cbvd5/ and data/raw/cvb/ downloaded
  [ ] models/rf-detr/rf-detr-medium.pth downloaded

Phase 1 — Data prep
  [ ] scripts/01 through 04 ran successfully
  [ ] data/processed/detection/combined/ exists

Phase 2 — Detector
  [ ] runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth exists

Phase 3 — Detection inference
  [ ] data/processed/tracking/cbvd5/ has ~687 files
  [ ] data/processed/tracking/cvb/   has ~502 files

Phase 4 — Segmentation + tracking (v2 path)
  [ ] runs/seg_medium_lr5e5/checkpoint_best_ema.pth exists
  [ ] data/processed/tracking_v2/{cbvd5,cvb}/ populated

Phase 5 — Tubelet generation
  [ ] data/processed/tubelets_rfdetr/labels.csv has 137596 lines

Phase 6 — VideoMAE training (HiPE1)
  [ ] Tubelets synced to HiPE1 (wc -l shows 137596)
  [ ] Docker image cattle-videomae:v1 loaded on HiPE1
  [ ] All 5 configs trained (check logs for "Early stopping" or "Done")
  [ ] Checkpoints retrieved: runs/behavior/videomae_*_v2/checkpoint_best.pt (×5)

Phase 7 — Evaluation and analytics
  [ ] scripts/11b_evaluate_rfdetr.sh ran cleanly
  [ ] results/behavior/f1_per_class.csv has v2 rows
  [ ] scripts/12b_generate_analytics_rfdetr.sh ran cleanly
  [ ] results/analytics/activity_budget_v2.csv exists
```

---

## Key Files to Know

| File | What it is |
| ---- | ---------- |
| `data/label_map.json` | The 7 behavior class definitions (committed) |
| `src/behavior/train.py` | VideoMAE training entry point |
| `src/behavior/evaluate.py` | Evaluation entry point |
| `src/tracking/track.py` | OC-SORT tracking wrapper |
| `src/segmentation/segment.py` | SAM2 segmentation (v1 path) |
| `configs/behavior/videomae_*_v2.yaml` | Training hyperparameters for each config |
| `results/behavior/f1_per_class.csv` | All evaluation results |
| `results/behavior/predictions/` | Per-tubelet predictions (committed) |
| `docs/hipe_ops.md` | Complete HiPE1 operations reference |
| `docs/pipeline.md` | Per-script reference with I/O details |
| `docs/setup.md` | All weight download commands |
