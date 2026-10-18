"""Video Ingestor — Phase 9 inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


class VideoIngestor:
    """Opens a video file and yields (frame_index, frame) pairs.

    FPS and resolution are read from the file header — never hardcoded.
    """

    SUPPORTED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if self.video_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported video format: {self.video_path.suffix}. "
                f"Supported: {self.SUPPORTED_SUFFIXES}"
            )
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open: {self.video_path}")

        self._fps: float = cap.get(cv2.CAP_PROP_FPS)
        self._width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield (frame_index, BGR frame) for every frame in the video."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open: {self.video_path}")
        try:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()
