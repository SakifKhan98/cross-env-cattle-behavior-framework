# Third-Party Dataset Licenses

This repository's own code, configs, and `data/label_map.json` are licensed MIT (see
`LICENSE`). The two datasets this project builds on carry their own, separate license terms,
which govern any dataset-derived content (e.g. tubelet/prediction CSVs, tracking metadata).

## CBVD-5

- Paper: Li et al., *Scientific Reports* (2024), DOI [10.1038/s41598-024-65953-x](https://doi.org/10.1038/s41598-024-65953-x) — licensed **CC BY 4.0**.
- Dataset: hosted on Kaggle (`fandaoerji/cbvd-5cow-behavior-video-dataset`); the dataset
  listing appears to be **CC0**, but this has not been independently confirmed against
  Kaggle's rendered license badge. **Confirm the Kaggle license directly before relying on
  this for redistribution decisions.**
- Our own preprocessing scripts/configs that process CBVD-5 are unrestricted (MIT, this
  repo's own code). Any CBVD-5-derived rows in committed metadata (tubelet/prediction CSVs)
  should carry attribution to Li et al. as a matter of good practice, consistent with CC BY 4.0.

## CVB (Cattle Video Behaviors)

- Dataset: Zia et al., CSIRO/ANU, [arXiv:2305.16555](https://arxiv.org/abs/2305.16555), hosted
  by CSIRO, DOI [10.25919/3g3t-p068](https://doi.org/10.25919/3g3t-p068) — licensed
  **CC BY-NC-SA 4.0**.
- Any CVB-derived rows in committed metadata (tubelet/prediction CSVs, `labels.csv`) inherit
  CVB's license terms: **NonCommercial, ShareAlike, and Attribution to CSIRO** are required
  for any redistribution of that derived content — this is a stricter and different license
  than this repository's own MIT code license.
- Full paper reprints of the CVB dataset paper are **not** included in this repository
  (copyright risk) — see the original paper link above instead.

## Summary

| Content                                              | License                         |
| ----------------------------------------------------- | -------------------------------- |
| This repo's own code, scripts, configs, `label_map.json` | MIT                            |
| CBVD-5-derived dataset metadata                        | CC BY 4.0 (attribution requested) |
| CVB-derived dataset metadata                           | CC BY-NC-SA 4.0 (NonCommercial, ShareAlike, attribution to CSIRO required) |

If you redistribute any file under `results/behavior/`, `data/processed/tubelets*/`, or
similar CVB-derived metadata, you must comply with CC BY-NC-SA 4.0 for that content,
regardless of this repository's overall MIT license.
