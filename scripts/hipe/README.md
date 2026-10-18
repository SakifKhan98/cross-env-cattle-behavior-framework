# HiPE1 Training Scripts

These scripts run on HiPE1, the lab server with 2× Tesla V100 GPUs.

## Server Details

- **Host:** `hipe1` (see `docs/hipe_ops.md` for SSH setup)
- **Home dir:** `/home/zxs12/`
- **Docker runtime:** NVIDIA Container Toolkit installed

## Scripts

| Script | Description |
|--------|-------------|
| `train_rfdetr_seg_medium.py` | RF-DETR-Seg-Medium fine-tuning (Config B, best result: ep59) |
| `train_rfdetr_seg_large.py` | RF-DETR-Seg-Large fine-tuning (experimental) |
| `train_rfdetr_seg_parameterized.py` | Env-var-driven general trainer; used for hyperparameter sweeps |

## Usage

Run via Docker on HiPE1:

```bash
# Medium model (standard training run)
docker compose -f docker/docker-compose.yml run --rm rfdetr_seg \
  scripts/hipe/train_rfdetr_seg_medium.py

# Parameterized (set hyperparams via env vars)
LR=1e-4 EPOCHS=60 BATCH=4 \
docker compose -f docker/docker-compose.yml run --rm rfdetr_seg \
  scripts/hipe/train_rfdetr_seg_parameterized.py
```

## Volume Requirements

Before running, ensure the following are populated on HiPE1:

- `data/rfdetr_seg/cattle/` — SAM2 pseudo-labels (training data for distillation)
- `models/rf-detr/rf-detr-medium.pth` — RF-DETR backbone (download from HuggingFace)

See `docs/setup.md` and `docs/hipe_ops.md` for full setup instructions.
