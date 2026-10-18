"""Tests for the Phase 9 FastAPI backend.

Unit tests: JobStore state transitions.
Integration tests: HTTP endpoints via FastAPI TestClient with mocked pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── skip if FastAPI / httpx not available ─────────────────────────────────────
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")

# ── JobStore unit tests ───────────────────────────────────────────────────────

def test_job_store_create_unique_ids():
    from api.job_store import JobStore
    s = JobStore()
    id1 = s.create()
    id2 = s.create()
    assert id1 != id2


def test_job_store_state_transitions():
    from api.job_store import JobStore
    s = JobStore()
    job_id = s.create()

    job = s.get(job_id)
    assert job is not None
    assert job.status == "pending"

    s.update_event(job_id, {"stage": 1, "status": "running"})
    job = s.get(job_id)
    assert job.status == "running"
    assert job.last_event == {"stage": 1, "status": "running"}

    s.complete(job_id, {"annotated_video": "/tmp/annotated.mp4"})
    job = s.get(job_id)
    assert job.status == "complete"
    assert job.result_paths["annotated_video"] == "/tmp/annotated.mp4"


def test_job_store_fail_transition():
    from api.job_store import JobStore
    s = JobStore()
    job_id = s.create()
    s.fail(job_id, "something broke")
    job = s.get(job_id)
    assert job.status == "failed"
    assert job.error == "something broke"


def test_job_store_missing_job_returns_none():
    from api.job_store import JobStore
    s = JobStore()
    assert s.get("nonexistent-id") is None


# ── synthetic video helper ────────────────────────────────────────────────────

_FPS = 25.0
_W, _H = 320, 240
_N_FRAMES = 75


def _make_video_bytes() -> bytes:
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, _FPS, (_W, _H))
    rng = np.random.default_rng(42)
    for _ in range(_N_FRAMES):
        frame = rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data


# ── mock pipeline ─────────────────────────────────────────────────────────────

_COW_BBOX = [40.0, 40.0, 160.0, 200.0]


def _mock_run_pipeline(config, progress=None):
    """Fake pipeline: emits progress events and writes minimal output files."""
    import time
    job_id = config["input"]["job_id"]
    output_root = Path(config["output"]["output_root"])
    job_dir = output_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "timelines").mkdir(exist_ok=True)

    if progress:
        for stage in range(1, 8):
            progress({"stage": stage, "stage_name": f"Stage{stage}",
                      "total_stages": 7, "frame": 10, "total_frames": 75,
                      "status": "running"})

    # Write expected output files
    (job_dir / "annotated.mp4").write_bytes(b"fake_video")
    with open(job_dir / "activity_budget.csv", "w", newline="") as f:
        csv.writer(f).writerow(["track_id", "behavior", "seconds", "pct"])
    with open(job_dir / "behavior_deviation.csv", "w", newline="") as f:
        csv.writer(f).writerow(["track_id", "behavior", "deviation"])
    (job_dir / "timelines" / "track_1.csv").write_text("frame,label\n0,Standing\n")


# ── integration tests ─────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path):
    """TestClient with the job store and output root reset for each test."""
    import api.job_store as js_mod
    import api.main as main_mod
    from fastapi.testclient import TestClient

    # Fresh store per test
    fresh_store = js_mod.JobStore()
    js_mod.store = fresh_store
    main_mod.store = fresh_store
    main_mod._OUTPUT_ROOT = tmp_path / "results" / "inference"

    with patch("api.main._run_pipeline_thread", side_effect=_patched_thread(main_mod)):
        yield TestClient(main_mod.app)


def _patched_thread(main_mod):
    """Return a side_effect that calls the mock pipeline then updates the store."""
    def _side(job_id, video_path, config):
        _mock_run_pipeline(config)
        result_paths = {
            "annotated_video": str(main_mod._OUTPUT_ROOT / job_id / "annotated.mp4"),
        }
        main_mod.store.complete(job_id, result_paths)
    return _side


def test_submit_job_returns_job_id(client):
    video_bytes = _make_video_bytes()
    resp = client.post(
        "/jobs",
        files={"video": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert len(body["job_id"]) == 36  # UUID4


def test_status_endpoint(client):
    video_bytes = _make_video_bytes()
    resp = client.post(
        "/jobs",
        files={"video": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/jobs/{job_id}/status")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("pending", "running", "complete", "failed")


def test_unknown_job_returns_404(client):
    resp = client.get("/jobs/nonexistent-uuid/status")
    assert resp.status_code == 404


def test_results_endpoint_returns_manifest(client, tmp_path):
    """After pipeline completes, /jobs/{id}/results returns a valid manifest."""
    import api.job_store as js_mod
    import api.main as main_mod

    # Create a job and manually complete it with real output files
    job_id = js_mod.store.create()
    job_dir = main_mod._OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "timelines").mkdir()
    (job_dir / "annotated.mp4").write_bytes(b"fake")
    (job_dir / "activity_budget.csv").write_text("a,b\n")
    (job_dir / "behavior_deviation.csv").write_text("a,b\n")
    (job_dir / "timelines" / "track_1.csv").write_text("frame,label\n")

    js_mod.store.complete(job_id, {"annotated_video": str(job_dir / "annotated.mp4")})

    resp = client.get(f"/jobs/{job_id}/results")
    assert resp.status_code == 200
    manifest = resp.json()
    assert f"/results/{job_id}/annotated.mp4" == manifest["annotated_video"]
    assert f"/results/{job_id}/activity_budget.csv" == manifest["activity_budget"]
    assert len(manifest["timelines"]) == 1


def test_sse_stream_terminates_with_complete(client, tmp_path):
    """SSE stream ends with status=complete for a finished job."""
    import api.job_store as js_mod
    import api.main as main_mod

    job_id = js_mod.store.create()
    job_dir = main_mod._OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "annotated.mp4").write_bytes(b"fake")
    (job_dir / "activity_budget.csv").write_text("a,b\n")
    (job_dir / "behavior_deviation.csv").write_text("a,b\n")

    js_mod.store.complete(job_id, {"annotated_video": str(job_dir / "annotated.mp4")})

    with client.stream("GET", f"/jobs/{job_id}/stream") as resp:
        assert resp.status_code == 200
        lines = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                lines.append(line[len("data:"):].strip())
                break  # first data line is the final event

    assert len(lines) >= 1
    event = json.loads(lines[0])
    assert event["status"] == "complete"
    assert "result_url" in event
