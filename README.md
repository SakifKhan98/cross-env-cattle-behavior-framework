# Cross-Environment Cattle Behavior Framework

Code and artifacts accompanying the manuscript **"Cross-environment generalization of an
identity-aware vision framework for cattle behavior analysis"** (Sakif Khan, Texas State
University, 2026).

The pipeline recognizes 7 dairy-cattle behaviors from surveillance video: RF-DETR detection →
RF-DETR-Seg instance segmentation → OC-SORT tracking → VideoMAE video classification →
activity-budget analytics. It is evaluated in-domain (CBVD-5, CVB), cross-domain (CBVD-5↔CVB),
and out-of-distribution on five additional external datasets, plus a perturbation robustness
study across 5 corruption types × 2 severities × 6 datasets.

This is a curated public release of a larger private research repository — it contains
everything needed to reproduce the manuscript's reported numbers and figures, trimmed of
internal planning docs, thesis drafts, and non-reproducible working artifacts.

---

## Key Results

Primary numbers (v2, RF-DETR-Seg tracks — see `docs/results.md` for v1/v2 comparison):

| Stage                                        | Metric            | Value      |
| --------------------------------------------- | ----------------- | ---------- |
| Detection (combined, checkpoint-selection val) | mAP@50            | 70.4%      |
| Detection (CBVD-5 test)                        | mAP@50            | 45.9%      |
| Detection (CVB test)                           | mAP@50            | 5.7%       |
| Tracking (CVB, v1 run)                         | IDF1 / MOTA       | 67.31% / 36.61% |
| Behavior — Config 1: CBVD-5 in-domain          | macro-F1          | **0.4511** |
| Behavior — Config 2: CVB in-domain             | macro-F1          | **0.7770** |
| Behavior — Config 3: CBVD-5 → CVB transfer     | macro-F1          | 0.1722     |
| Behavior — Config 4: CVB → CBVD-5 transfer     | macro-F1          | 0.2253     |
| Behavior — Config 5: combined training         | macro-F1          | 0.7507     |

Out-of-distribution detection (mAP@50): CattleEyeView 47.0%, Freeman Center (angled ranch) 73.0%,
OpenCows2020 33.3%, Cows2021 27.3%, mmCows 32.4%.

Full per-class and per-dataset breakdowns: `results/behavior/f1_per_class.csv`,
`results/behavior/v1_v2_comparison.csv`, `results/detection/*.json`, `results/generalization/`.
Figure source panels: `results/figures/`.

See `docs/results.md` for narrative discussion, including the acknowledged limitation that the
v1 tracking IDF1/MOTA figures come from a mask-recovery run distinct from the box-only v2 run
that actually produced the tubelets behind the reported behavior numbers (see `docs/pipeline.md`).

---

## Repository Layout

| Directory    | Contents                                                                  |
| ------------ | -------------------------------------------------------------------------- |
| `src/`       | Library modules — detection, segmentation, tracking, behavior, analytics  |
| `scripts/`   | Numbered shell wrappers (01–19) that run the pipeline end-to-end          |
| `configs/`   | YAML experiment configs — all hyperparameters live here                   |
| `docker/`    | One Dockerfile per pipeline stage group + `docker-compose.yml`            |
| `tests/`     | Unit tests (`pytest`)                                                      |
| `notebooks/` | Figure/chart generation (`figure_plates.ipynb`, `figure_charts.ipynb`)    |
| `ui/`        | React frontend for the inference demo app                                 |
| `api/`       | FastAPI backend for the inference demo app                                |
| `results/`   | Committed result files (JSON/CSV/PNG) that back manuscript claims/figures |
| `logs/`      | Training logs (informational)                                             |
| `data/`      | `label_map.json` only — raw/processed data is gitignored, see below       |
| `docs/`      | Setup, pipeline, dataset, Docker, results, and reproduction guides        |

Two parallel pipeline paths exist: **v1** (SAM2 segmentation in the tracking loop, early
results) and **v2** (`b`-suffix scripts, single-stage RF-DETR-Seg, the manuscript's primary
numbers). Full execution order and per-script I/O contracts: `docs/pipeline.md`.

---

## Setup & Quick Start

```bash
git clone https://github.com/SakifKhan98/cross-env-cattle-behavior-framework.git
cd cross-env-cattle-behavior-framework

conda env create -f environment.yml
conda activate cattletransformer

# External dependencies (not pip-installable as part of this repo)
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'
```

Full step-by-step instructions (dataset downloads, checkpoint downloads, Docker, HiPE1
training commands): `docs/setup.md`, `docs/datasets.md`, `docs/docker.md`,
`docs/reproduction_guide.md`.

---

## Pretrained Weights

All checkpoints: **https://huggingface.co/sakifkhan98/cross-env-cattle-behavior-framework**

| Weight                                  | Description                                      | macro-F1 |
| ---------------------------------------- | ------------------------------------------------- | -------- |
| `rf-detr-medium.pth`                     | RF-DETR backbone (pretrained, before fine-tuning) | —        |
| `rfdetr_combined_v1_best.pth`            | RF-DETR detector, standalone (all detection/OOD/perturbation numbers) | — |
| `rf-detr-seg-medium.pt`                  | RF-DETR-Seg (segmentation + v2 tracking/tubelets) | —        |
| `videomae_cbvd5_v2.pt`                   | Config 1: CBVD-5 in-domain                        | 0.4511   |
| `videomae_cvb_v2.pt`                     | Config 2: CVB in-domain                           | 0.7770   |
| `videomae_cbvd5_to_cvb_v2.pt`            | Config 3: CBVD-5 → CVB transfer                   | 0.1722   |
| `videomae_cvb_to_cbvd5_v2.pt`            | Config 4: CVB → CBVD-5 transfer                   | 0.2253   |
| `videomae_combined_v2.pt`                | Config 5: combined training                       | 0.7507   |

```bash
huggingface-cli download sakifkhan98/cross-env-cattle-behavior-framework \
    rfdetr_combined_v1_best.pth rf-detr-seg-medium.pt videomae_combined_v2.pt \
    --local-dir weights/
```

Precomputed tracking/tubelet data (where available): **https://huggingface.co/datasets/sakifkhan98/cross-env-cattle-behavior-data**

---

## Datasets

| Dataset | Domain          | Behaviors   | License (original)       |
| ------- | --------------- | ----------- | ------------------------- |
| CBVD-5  | Indoor barn     | 5 (IDs 0–4) | CC BY 4.0 (paper); dataset license on Kaggle — verify before redistributing derivatives |
| CVB     | Outdoor pasture | 7 (IDs 0–6) | CC BY-NC-SA 4.0 (CSIRO)   |

Both datasets are downloaded separately and gitignored — see `docs/datasets.md` for sources
and directory layout. **See `THIRD_PARTY_LICENSES.md` before redistributing any CVB-derived
metadata (tubelet/prediction CSVs) — it carries NonCommercial + ShareAlike + attribution
obligations distinct from this repo's own MIT license.**

---

## Inference Demo App

`ui/` (React) + `api/` (FastAPI) implement the web demo referenced in the manuscript: upload a
video, run the full pipeline, and view behavior timelines/activity budgets in the browser.

```bash
bash scripts/start_app.sh
```

---

## Citation

If you use this work, please cite the manuscript (citation details to follow publication):

```
Khan, S. "Cross-environment generalization of an identity-aware vision framework for
cattle behavior analysis." (2026, in review).
```

---

## License

Code in this repository is licensed under the MIT License — see `LICENSE`.

Dataset-derived metadata is subject to the original datasets' licenses, which are **not**
MIT — see `THIRD_PARTY_LICENSES.md`.
