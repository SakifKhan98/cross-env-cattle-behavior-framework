"""
src/segmentation/mask_utils.py
Phase 3 — Mask Utility Functions

PURPOSE:
    Helper functions for converting SAM2 binary masks to/from COCO RLE format.
    Kept in a separate file so both segment.py (Phase 3) and track.py (Phase 4)
    can import these without duplicating code.

WHY RLE (Run-Length Encoding):
    A raw binary mask for a 1920×1080 image is 2,073,600 bytes (~2MB).
    The same mask in COCO RLE format is typically 3–8KB — a 200–500x reduction.
    With 687 CBVD-5 videos × 6 frames × ~5 cows = ~20,000 masks, storing raw
    masks would require ~40GB. RLE keeps the whole dataset under 1GB.

COCO RLE format:
    {
        "counts": "...",        # RLE-encoded string
        "size": [height, width] # image dimensions (NOTE: height first!)
    }
"""

import numpy as np
from pycocotools import mask as mask_utils


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """
    Encode a binary mask to COCO RLE format.

    Args:
        binary_mask: (H, W) numpy array of dtype bool or uint8.
                     1 = foreground (cow), 0 = background.

    Returns:
        dict with keys "counts" (str) and "size" ([H, W]).

    Example:
        rle = mask_to_rle(my_mask)
        # rle = {"counts": "abc123...", "size": [1080, 1920]}
    """
    # Squeeze out any extra dimensions SAM2 may have added
    # SAM2 sometimes returns (1, H, W) or (1, 1, H, W) instead of (H, W)
    binary_mask = np.squeeze(binary_mask)

    # Ensure exactly 2D
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape: {binary_mask.shape}")

    # pycocotools requires:
    #   1. Fortran-contiguous memory layout (column-major, not row-major)
    #   2. uint8 dtype (not bool)
    binary_mask_f = np.asfortranarray(binary_mask.astype(np.uint8))

    rle = mask_utils.encode(binary_mask_f)

    if rle is None:
        raise RuntimeError(
            "pycocotools mask_utils.encode() returned None. "
            f"Mask dtype={binary_mask_f.dtype}, "
            f"shape={binary_mask_f.shape}, "
            f"contiguous={binary_mask_f.flags['F_CONTIGUOUS']}"
        )

    # pycocotools returns counts as bytes — convert to str for JSON serialization
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Decode a COCO RLE dict back to a binary mask.

    Args:
        rle: dict with "counts" (str) and "size" ([H, W]).

    Returns:
        (H, W) numpy array of dtype uint8 (0 or 1).

    Example:
        mask = rle_to_mask(rle)
        # mask.shape == (1080, 1920), dtype == uint8
    """
    # pycocotools expects counts as bytes, not str
    rle_copy = {"counts": rle["counts"].encode("utf-8"), "size": rle["size"]}
    return mask_utils.decode(rle_copy)


def mask_to_bbox(binary_mask: np.ndarray) -> list:
    """
    Compute the tight bounding box of a binary mask.

    Useful in Phase 3 when propagating masks: the previous frame's mask
    gives us a rough bounding box to re-prompt SAM2 with.

    Args:
        binary_mask: (H, W) bool or uint8 array.

    Returns:
        [x_min, y_min, width, height] in COCO format (pixels).
        Returns None if the mask is empty.
    """
    rows = np.any(binary_mask, axis=1)  # which rows have any foreground
    cols = np.any(binary_mask, axis=0)  # which cols have any foreground

    if not rows.any():
        return None  # empty mask

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # r = row = y axis, c = col = x axis
    x = int(c_min)
    y = int(r_min)
    w = int(c_max - c_min)
    h = int(r_max - r_min)

    return [x, y, w, h]


def mask_area(binary_mask: np.ndarray) -> int:
    """
    Count foreground pixels in a binary mask.

    Args:
        binary_mask: (H, W) bool or uint8 array.

    Returns:
        Integer pixel count.
    """
    return int(binary_mask.astype(bool).sum())
