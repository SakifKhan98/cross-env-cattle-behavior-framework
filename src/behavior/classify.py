"""VideoMAE tubelet classification — Phase 9 inference pipeline."""

from __future__ import annotations

import csv
import sys
import tempfile
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

NUM_CLASSES = 7
_VIDEOMAE_BASE = "MCG-NJU/videomae-base"


def _build_model(num_classes: int):
    from transformers import VideoMAEForVideoClassification
    import torch.nn as nn

    model = VideoMAEForVideoClassification.from_pretrained(_VIDEOMAE_BASE)
    model.classifier = nn.Linear(768, num_classes)
    return model


def classify_tubelets(
    tubelet_rows: list,
    checkpoint: str | Path,
    output_dir: str | Path,
    job_id: str,
    num_classes: int = NUM_CLASSES,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    progress: Optional[Callable] = None,
    _model=None,
) -> Path:
    """Run VideoMAE inference on tubelet clips and write predictions.csv.

    Args:
        tubelet_rows: from export_tubelets_from_tracks — list of dicts with
                      tubelet_dir, track_id, start_frame, end_frame.
        checkpoint:   Path to VideoMAE .pt checkpoint file.
        output_dir:   Where to write predictions.csv.
        job_id:       Used as video_id in predictions.csv.
        num_classes:  Output classes (default 7).
        batch_size:   Inference batch size.
        device:       torch.device. Defaults to CUDA if available.
        progress:     Optional callable(n_done: int, n_total: int).
        _model:       Pre-built model for testing — bypasses checkpoint loading.

    Returns:
        Path to predictions.csv.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.csv"
    logit_cols = [f"logit_{i}" for i in range(num_classes)]

    if not tubelet_rows:
        with open(preds_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset", "video_id", "tubelet_dir",
                "start_frame", "end_frame", "label_id", "pred_label_id",
            ] + logit_cols)
        return preds_path

    # Write a temporary labels CSV that TubeletDataset can read.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    try:
        tw = csv.DictWriter(tmp, fieldnames=[
            "dataset", "video_id", "tubelet_dir",
            "start_frame", "end_frame", "label_id", "split",
        ])
        tw.writeheader()
        for row in tubelet_rows:
            tw.writerow({
                "dataset": "inference",
                "video_id": job_id,
                "tubelet_dir": row["tubelet_dir"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "label_id": 0,
                "split": "inference",
            })
        tmp.close()

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from behavior.dataset import TubeletDataset

        # TubeletDataset resolves tubelet_dir relative to root.
        # Absolute tubelet_dir + root=Path(".") works because
        # Path(".") / "/abs/path" == Path("/abs/path") in Python.
        ds = TubeletDataset(tmp.name)

        if _model is None:
            model = _build_model(num_classes).to(device)
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state", ckpt)
            # Remap keys saved with older HF VideoMAE (fc_norm → videomae.layernorm)
            state = {
                k.replace("fc_norm.", "videomae.layernorm."): v
                for k, v in state.items()
            }
            model.load_state_dict(state)
        else:
            model = _model.to(device) if hasattr(_model, "to") else _model

        model.eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        all_preds: list = []
        all_logits_list: list = []
        n_done = 0

        with torch.no_grad():
            for videos, _ in loader:
                videos = videos.permute(0, 2, 1, 3, 4).to(device)
                out = model(pixel_values=videos)
                logits = out.logits.cpu()
                preds = logits.argmax(dim=-1).tolist()
                all_preds.extend(preds)
                all_logits_list.append(logits.numpy())
                n_done += len(preds)
                if progress:
                    progress(n_done, len(ds))

        all_logits_np = np.concatenate(all_logits_list, axis=0)

        with open(preds_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dataset", "video_id", "tubelet_dir",
                "start_frame", "end_frame", "label_id", "pred_label_id",
            ] + logit_cols)
            for i, row in enumerate(tubelet_rows):
                w.writerow([
                    "inference",
                    job_id,
                    row["tubelet_dir"],
                    row["start_frame"],
                    row["end_frame"],
                    -1,
                    all_preds[i],
                ] + [f"{v:.6f}" for v in all_logits_np[i]])

    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return preds_path
