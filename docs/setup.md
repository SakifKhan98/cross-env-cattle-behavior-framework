# Setup Guide

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

---

## 1. Prerequisites

- Linux (Ubuntu 22.04 tested)
- NVIDIA GPU with ≥12 GB VRAM (RTX 3060 for local smoke tests; V100 for full training)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://github.com/conda-forge/miniforge)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Docker GPU stages)
- Git, `wget`, `tar`

---

## 2. Clone the Repository

```bash
git clone https://github.com/sakifkhan98/cattle-vision-framework.git
cd cattle-vision-framework
```

---

## 3. Conda Environment

Create and activate the environment (Python 3.11, PyTorch 2.8 + CUDA 12.1):

```bash
conda env create -f environment.yml
conda activate cattletransformer
```

Or install pip-only (no conda):

```bash
pip install -r requirements.txt
```

### Verify

```bash
python -c "from src.data.label_utils import BEHAVIOR_NAMES; print(BEHAVIOR_NAMES)"
# Expected: ['Standing', 'Lying', 'Foraging/Grazing', 'Drinking', 'Ruminating', 'Grooming', 'Other']
```

---

## 4. External Dependencies

These are NOT bundled in the repo. Install them as part of setup.

### 4.1 OC-SORT (required for script 08b)

```bash
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
```

`third_party/` is gitignored. The tracker imports directly from this source tree.

Install OC-SORT's Python deps (lap, filterpy — already in requirements.txt):

```bash
pip install lap filterpy
```

### 4.2 SAM2 (required for script 07)

```bash
pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'
```

Download the SAM2 checkpoint:

```bash
mkdir -p models/sam2/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -O models/sam2/sam2.1_hiera_large.pt
```

---

## 5. Model Weight Downloads

All weights are hosted on HuggingFace: **https://huggingface.co/sakifkhan98/cattle-vision-framework**

```bash
pip install huggingface-hub   # already in requirements.txt
mkdir -p models/rf-detr/ models/sam2/
```

### RF-DETR backbone (COCO pretrained — required for detector training)

```bash
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-medium.pth \
  --local-dir models/rf-detr/
```

### RF-DETR cattle detector (Phase 1 — required for scripts 06+)

```bash
mkdir -p runs/detection/rfdetr_combined_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework rfdetr_combined_v1_best.pth \
  --local-dir runs/detection/rfdetr_combined_v1/
# Rename to match the expected checkpoint filename:
mv runs/detection/rfdetr_combined_v1/rfdetr_combined_v1_best.pth \
   runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth
```

### RF-DETR-Seg (Phase 3b — for SAM2-distilled segmentation model)

```bash
mkdir -p runs/seg_medium_lr5e5/
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-seg-medium.pt \
  --local-dir runs/seg_medium_lr5e5/
mv runs/seg_medium_lr5e5/rf-detr-seg-medium.pt \
   runs/seg_medium_lr5e5/checkpoint_best_ema.pth
```

### VideoMAE behavior classifiers — v1 (required for scripts 11–12)

Trained on SAM2 tubelets. Config 5 (combined) is the primary v1 model.

```bash
# Config 5: combined training — best for analytics (macro-F1 = 0.7537)
mkdir -p runs/behavior/videomae_combined_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_combined_v1.pt \
  --local-dir runs/behavior/videomae_combined_v1/
mv runs/behavior/videomae_combined_v1/videomae_combined_v1.pt \
   runs/behavior/videomae_combined_v1/checkpoint_best.pt

# Config 2: CVB in-domain — macro-F1 = 0.7607
mkdir -p runs/behavior/videomae_cvb_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_v1.pt \
  --local-dir runs/behavior/videomae_cvb_v1/
mv runs/behavior/videomae_cvb_v1/videomae_cvb_v1.pt \
   runs/behavior/videomae_cvb_v1/checkpoint_best.pt

# Config 1: CBVD-5 in-domain — macro-F1 = 0.3149
mkdir -p runs/behavior/videomae_cbvd5_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_v1.pt \
  --local-dir runs/behavior/videomae_cbvd5_v1/
mv runs/behavior/videomae_cbvd5_v1/videomae_cbvd5_v1.pt \
   runs/behavior/videomae_cbvd5_v1/checkpoint_best.pt

# Config 3: CBVD-5→CVB cross-domain — macro-F1 = 0.1690
mkdir -p runs/behavior/videomae_cbvd5_to_cvb_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_to_cvb_v1.pt \
  --local-dir runs/behavior/videomae_cbvd5_to_cvb_v1/
mv runs/behavior/videomae_cbvd5_to_cvb_v1/videomae_cbvd5_to_cvb_v1.pt \
   runs/behavior/videomae_cbvd5_to_cvb_v1/checkpoint_best.pt

# Config 4: CVB→CBVD-5 cross-domain — macro-F1 = 0.1789
mkdir -p runs/behavior/videomae_cvb_to_cbvd5_v1/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_to_cbvd5_v1.pt \
  --local-dir runs/behavior/videomae_cvb_to_cbvd5_v1/
mv runs/behavior/videomae_cvb_to_cbvd5_v1/videomae_cvb_to_cbvd5_v1.pt \
   runs/behavior/videomae_cvb_to_cbvd5_v1/checkpoint_best.pt
```

### VideoMAE behavior classifiers — v2 (required for scripts 11b–12b)

Trained on RF-DETR-Seg tubelets. These are the **primary thesis results** for Configs 1–4.

```bash
# Config 5: combined training — macro-F1 = 0.7507
mkdir -p runs/behavior/videomae_combined_v2/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_combined_v2.pt \
  --local-dir runs/behavior/videomae_combined_v2/
mv runs/behavior/videomae_combined_v2/videomae_combined_v2.pt \
   runs/behavior/videomae_combined_v2/checkpoint_best.pt

# Config 2: CVB in-domain — macro-F1 = 0.7770
mkdir -p runs/behavior/videomae_cvb_v2/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_v2.pt \
  --local-dir runs/behavior/videomae_cvb_v2/
mv runs/behavior/videomae_cvb_v2/videomae_cvb_v2.pt \
   runs/behavior/videomae_cvb_v2/checkpoint_best.pt

# Config 1: CBVD-5 in-domain — macro-F1 = 0.4511
mkdir -p runs/behavior/videomae_cbvd5_v2/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_v2.pt \
  --local-dir runs/behavior/videomae_cbvd5_v2/
mv runs/behavior/videomae_cbvd5_v2/videomae_cbvd5_v2.pt \
   runs/behavior/videomae_cbvd5_v2/checkpoint_best.pt

# Config 3: CBVD-5→CVB cross-domain — macro-F1 = 0.1722
mkdir -p runs/behavior/videomae_cbvd5_to_cvb_v2/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_to_cvb_v2.pt \
  --local-dir runs/behavior/videomae_cbvd5_to_cvb_v2/
mv runs/behavior/videomae_cbvd5_to_cvb_v2/videomae_cbvd5_to_cvb_v2.pt \
   runs/behavior/videomae_cbvd5_to_cvb_v2/checkpoint_best.pt

# Config 4: CVB→CBVD-5 cross-domain — macro-F1 = 0.2253
mkdir -p runs/behavior/videomae_cvb_to_cbvd5_v2/
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_to_cbvd5_v2.pt \
  --local-dir runs/behavior/videomae_cvb_to_cbvd5_v2/
mv runs/behavior/videomae_cvb_to_cbvd5_v2/videomae_cvb_to_cbvd5_v2.pt \
   runs/behavior/videomae_cvb_to_cbvd5_v2/checkpoint_best.pt
```

---

## 6. Dataset Downloads

See [docs/datasets.md](datasets.md) for full dataset descriptions and download instructions.

```bash
mkdir -p data/raw/cbvd5/ data/raw/cvb/
# CBVD-5: see docs/datasets.md §1 for official download URL
# CVB:    see docs/datasets.md §2 for official download URL
```

---

## 7. Quickstart — Skip GPU Inference

To skip scripts 06, 07a, and 08b (detection inference + RF-DETR-Seg + OC-SORT tracking,
totaling several hours of GPU time), download pre-computed tracking outputs from HuggingFace:

```bash
pip install huggingface-hub
mkdir -p data/processed/tracking_v2/

# Download tarballs (~661 MB total compressed)
huggingface-cli download sakifkhan98/cattle-vision-data \
  tracking_v2_cbvd5.tar.gz tracking_v2_cvb.tar.gz \
  --repo-type dataset --local-dir /tmp/tracking_v2/

# Extract
tar -xf /tmp/tracking_v2/tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2/
tar -xf /tmp/tracking_v2/tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2/
```

After extraction, you can skip scripts 06–08 and go directly to `scripts/09_generate_tubelets.sh`.
Or skip 09 too by downloading pretrained weights and running `scripts/11_evaluate.sh`.

---

## 8. Expected Directory Structure After Full Setup

```
cattle-vision-framework/
├── data/
│   ├── label_map.json           ← committed
│   ├── raw/
│   │   ├── cbvd5/               ← ~12 GB (download via docs/datasets.md)
│   │   └── cvb/                 ← ~15 GB (download via docs/datasets.md)
│   └── processed/               ← generated by scripts 01–09 (or download tracking_v2 from HF)
├── models/
│   ├── rf-detr/rf-detr-medium.pth       ← HuggingFace download (§5 above)
│   └── sam2/sam2.1_hiera_large.pt       ← direct wget (§4.2 above)
├── runs/
│   ├── seg_medium_lr5e5/checkpoint_best_ema.pth  ← HuggingFace download (§5 above)
│   ├── detection/rfdetr_combined_v1/checkpoint_best_total.pth  ← HuggingFace download
│   └── behavior/videomae_*/checkpoint_best.pt                  ← HuggingFace download
└── third_party/
    └── OC_SORT/                 ← git clone (§4.1 above)
```

---

## 9. Running the Web App

The Phase 9 inference UI is a FastAPI backend + React frontend.

### Production (one command)

```bash
# 1. Build the React frontend (once, or after any UI change)
cd ui && npm run build && cd ..

# 2. Start the server and open the browser
bash scripts/start_app.sh
```

`start_app.sh` will:

- Verify `ui/dist/` exists (error with instructions if not)
- Activate the `cattletransformer` conda environment
- Start FastAPI on `http://localhost:8000` (serves the built React bundle)
- Open `localhost:8000` in your default browser after 2 seconds

Use `--port PORT` to change the port:

```bash
bash scripts/start_app.sh --port 8080
```

### Development (hot-reload)

Run the backend and frontend in separate terminals:

```bash
# Terminal 1 — FastAPI backend (auto-reloads on Python changes)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Vite dev server (hot-reloads on React changes)
cd ui && npm run dev
```

The Vite dev server runs on `http://localhost:5173` and proxies `/jobs` and
`/results` requests to port 8000. Open `localhost:5173` in your browser during
development.

---

## 10. Verification

```bash
# Imports work
python -c "from src.data.label_utils import BEHAVIOR_NAMES; print(BEHAVIOR_NAMES)"

# OC-SORT path is correct (does not raise ImportError)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('third_party/OC_SORT')))
from trackers.ocsort_tracker.ocsort import OCSort
print('OC-SORT OK')
"

# Weights present
ls models/

# Run data inspection (no GPU needed)
bash scripts/01_inspect_data.sh
```
