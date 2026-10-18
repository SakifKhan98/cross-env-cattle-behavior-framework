import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABEL_NAMES = {0: "Standing", 1: "Lying", 2: "Foraging", 3: "Drinking", 4: "Ruminating", 5: "Grooming", 6: "Other"}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class TubeletDataset(Dataset):
    """
    Reads labels.csv and serves (tensor[3,16,224,224], label_id) pairs.

    Args:
        labels_csv:     Path to labels.csv (absolute or relative to cwd).
        dataset_filter: None | "cbvd5" | "cvb"  — keep only this dataset.
        split_filter:   None | "train" | "val" | "test"  — keep only this split.
        label_subset:   None | list[int]  — keep only these label IDs.
        root:           Root prepended to tubelet_dir. Defaults to parent of labels_csv.
    """

    def __init__(
        self,
        labels_csv: str,
        dataset_filter=None,
        split_filter=None,
        label_subset=None,
        root: str = None,
    ):
        df = pd.read_csv(labels_csv)

        if dataset_filter is not None:
            df = df[df["dataset"] == dataset_filter]
        if split_filter is not None:
            df = df[df["split"] == split_filter]
        if label_subset is not None:
            df = df[df["label_id"].isin(label_subset)]

        self.df = df.reset_index(drop=True)

        # tubelet_dir in labels.csv is relative to the repo root
        self.root = Path(root) if root is not None else Path(".")

        # precompute per-channel normalization tensors for speed
        self._mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1, 1)  # [3,1,1,1]
        self._std  = torch.tensor(_STD,  dtype=torch.float32).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tubelet_dir = self.root / row["tubelet_dir"]
        label_id = int(row["label_id"])

        frames = []
        for i in range(16):
            img_path = tubelet_dir / f"frame_{i:02d}.jpg"
            img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
            frames.append(np.array(img, dtype=np.float32) / 255.0)  # [224,224,3]

        # stack → [16,224,224,3] → [16,3,224,224] → [3,16,224,224]
        tensor = torch.from_numpy(np.stack(frames))          # [16,224,224,3]
        tensor = tensor.permute(0, 3, 1, 2)                  # [16,3,224,224]
        tensor = tensor.permute(1, 0, 2, 3)                  # [3,16,224,224]
        tensor = (tensor - self._mean) / self._std

        return tensor, label_id

    def class_weights(self, num_classes: int = 7) -> torch.Tensor:
        """Inverse-frequency weights for nn.CrossEntropyLoss."""
        counts = torch.zeros(num_classes)
        for lid in self.df["label_id"]:
            counts[int(lid)] += 1
        total = counts.sum()
        weights = total / (num_classes * counts.clamp(min=1))
        return weights
