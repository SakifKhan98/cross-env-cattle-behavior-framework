# Training and Evaluation Logs

HiPE1 logs from VideoMAE Phase 6 training and evaluation. All runs used
`docker run --gpus all --shm-size=16g cattle-behavior ...` on HiPE1 Tesla V100.

Naming convention: `{verb}_{dataset}_{version}.log`
- verb: `train` or `eval`
- dataset: `cbvd5`, `cvb`, `combined`, `cbvd5_to_cvb`, `cvb_to_cbvd5`
- version: `v1` (SAM2 tubelets) or `v2` (RF-DETR-Seg tubelets)

---

## Training Logs

| File | Config | Best Macro-F1 | Best Epoch | Notes |
|------|--------|---------------|------------|-------|
| `train_cbvd5_v1.log` | Config 1: CBVD-5 in-domain | 0.3149 | 4 | v1 SAM2 path |
| `train_cvb_v1.log` | Config 2: CVB in-domain | 0.7607 | 4 | Highest per-dataset F1 (v1) |
| `train_combined_v1.log` | Config 5: CBVD-5 + CVB joint | 0.7537 | 4 | Primary analytics model (v1) |
| `train_cbvd5_to_cvb_v1.log` | Config 3: CBVD-5 → CVB cross-domain | 0.1690 | — | Large domain gap finding |
| `train_cvb_to_cbvd5_v1.log` | Config 4: CVB → CBVD-5 cross-domain | 0.1789 | 7 | Large domain gap finding |
| `train_cbvd5_v2.log` | Config 1: CBVD-5 in-domain | 0.4511 | — | v2 RF-DETR-Seg path |
| `train_cvb_v2.log` | Config 2: CVB in-domain | 0.7770 | — | v2 RF-DETR-Seg path |
| `train_combined_v2.log` | Config 5: CBVD-5 + CVB joint | 0.7507 | — | v2 RF-DETR-Seg path |
| `train_cbvd5_to_cvb_v2.log` | Config 3: CBVD-5 → CVB cross-domain | 0.1722 | — | v2 RF-DETR-Seg path |
| `train_cvb_to_cbvd5_v2.log` | Config 4: CVB → CBVD-5 cross-domain | 0.2253 | — | v2 RF-DETR-Seg path |

## Evaluation Logs

| File | Config | Notes |
|------|--------|-------|
| `eval_combined_v1.log` | Config 5 evaluation | Wrote `results/behavior/predictions/combined_val.csv` |
| `eval_cvb_v1.log` | Config 2 evaluation | Wrote `results/behavior/predictions/cvb_val.csv` |
| `eval_cbvd5_v1.log` | Config 1 evaluation | Wrote `results/behavior/predictions/cbvd5_val.csv` |
| `eval_cbvd5_to_cvb_v1.log` | Config 3 evaluation | Wrote `results/behavior/predictions/cbvd5_to_cvb_val.csv` |
| `eval_cvb_to_cbvd5_v1.log` | Config 4 evaluation | Wrote `results/behavior/predictions/cvb_to_cbvd5_val.csv` |

## Other

| File | Notes |
|------|-------|
| `tubelet_reexport_cvb.log` | CVB tubelet re-export (2026-04-29) after relaxing frame presence threshold to 12/16 |
| `train_rfdetr_seg_hipe1.log` | RF-DETR-Seg Phase 3b training log (1.3 MB JSON Lines); Config B = `seg_medium_lr5e5` |

---

## What to Look For

- **Training:** epoch-by-epoch macro-F1 is the key metric. Loss curves can be plotted with
  `notebooks/04_behavior_results.ipynb` or `scripts/plot_training_results.py`.
- **Evaluation:** final per-class F1 summary is printed at the end of each eval log.
  Ground truth numbers are in `results/behavior/f1_per_class.csv`.
- **cudnn warnings** (`UserWarning: TF32 ... cudnn`) appear in all logs — harmless.
  PyTorch uses a fallback convolution plan; results are unaffected.
