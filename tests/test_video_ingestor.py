"""Unit tests for the Phase 9 Video Ingestor."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

cv2 = pytest.importorskip("cv2")

from src.inference.video_ingestor import VideoIngestor


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_video(path: str, *, fps: float, width: int, height: int, n_frames: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ── fixtures ──────────────────────────────────────────────────────────────────

FPS = 25.0
WIDTH = 64
HEIGHT = 48
N_FRAMES = 30


@pytest.fixture(scope="module")
def synthetic_mp4(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("videos") / "test.mp4")
    _make_synthetic_video(path, fps=FPS, width=WIDTH, height=HEIGHT, n_frames=N_FRAMES)
    return path


@pytest.fixture(scope="module")
def synthetic_avi(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("videos") / "test.avi")
    _make_synthetic_video(path, fps=FPS, width=WIDTH, height=HEIGHT, n_frames=N_FRAMES)
    return path


# ── header metadata ───────────────────────────────────────────────────────────

class TestVideoIngestorMetadata:
    def test_fps_read_from_header(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        assert ingestor.fps == pytest.approx(FPS, rel=0.01)

    def test_resolution_read_from_header(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        assert ingestor.width == WIDTH
        assert ingestor.height == HEIGHT

    def test_frame_count_reported(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        # CAP_PROP_FRAME_COUNT can be off by one on some codecs; allow ±1.
        assert abs(ingestor.frame_count - N_FRAMES) <= 1

    def test_avi_fps_read_from_header(self, synthetic_avi):
        ingestor = VideoIngestor(synthetic_avi)
        assert ingestor.fps == pytest.approx(FPS, rel=0.01)


# ── frame iteration ───────────────────────────────────────────────────────────

class TestVideoIngestorFrames:
    def test_yields_correct_frame_count(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        frames = list(ingestor.frames())
        assert len(frames) == N_FRAMES

    def test_frame_indices_sequential(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        indices = [idx for idx, _ in ingestor.frames()]
        assert indices == list(range(N_FRAMES))

    def test_frames_are_numpy_arrays(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        for _idx, frame in ingestor.frames():
            assert isinstance(frame, np.ndarray)
            break  # check just the first frame

    def test_frame_shape_matches_resolution(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        for _idx, frame in ingestor.frames():
            assert frame.shape == (HEIGHT, WIDTH, 3)
            break

    def test_avi_yields_correct_frame_count(self, synthetic_avi):
        ingestor = VideoIngestor(synthetic_avi)
        frames = list(ingestor.frames())
        assert len(frames) == N_FRAMES

    def test_can_iterate_twice(self, synthetic_mp4):
        ingestor = VideoIngestor(synthetic_mp4)
        first_run = [idx for idx, _ in ingestor.frames()]
        second_run = [idx for idx, _ in ingestor.frames()]
        assert first_run == second_run


# ── error handling ────────────────────────────────────────────────────────────

class TestVideoIngestorErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoIngestor(tmp_path / "nonexistent.mp4")

    def test_unsupported_extension_raises(self, tmp_path):
        fake = tmp_path / "video.xyz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported video format"):
            VideoIngestor(fake)
