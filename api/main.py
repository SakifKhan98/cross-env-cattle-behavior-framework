"""FastAPI backend — Phase 9 inference pipeline server."""

from __future__ import annotations

import asyncio
import csv
import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from api.job_store import store

app = FastAPI(title="Cattle Vision Inference API")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "inference" / "default.yaml"
_OUTPUT_ROOT = _REPO_ROOT / "results" / "inference"


def _load_default_config() -> Dict[str, Any]:
    with open(_DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def _run_pipeline_thread(job_id: str, video_path: str, config: Dict[str, Any]) -> None:
    """Execute the pipeline in a background thread, updating the store."""
    from src.inference.pipeline import run_pipeline

    def _on_progress(event: Dict[str, Any]) -> None:
        store.update_event(job_id, event)

    try:
        run_pipeline(config, progress=_on_progress)
        job_dir = Path(config["output"]["output_root"]) / job_id
        result_paths = {
            "annotated_video": str(job_dir / "annotated.mp4"),
            "activity_budget": str(job_dir / "activity_budget.csv"),
            "behavior_deviation": str(job_dir / "behavior_deviation.csv"),
            "timelines": str(job_dir / "timelines"),
        }
        store.complete(job_id, result_paths)
    except Exception as exc:
        store.fail(job_id, str(exc))


@app.post("/jobs")
async def submit_job(
    video: UploadFile = File(...),
    rfdetr_checkpoint: str = Form(default=""),
    videomae_checkpoint: str = Form(default=""),
    confidence_threshold: float = Form(default=0.3),
    cleanup: bool = Form(default=False),
) -> JSONResponse:
    """Accept a video upload, start the inference pipeline in a background thread."""
    job_id = store.create(video_filename=video.filename)

    # Save uploaded video to a temp file that persists until the job finishes
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"cvf_job_{job_id}_"))
    video_path = tmp_dir / (video.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cfg = _load_default_config()
    cfg["input"]["video_path"] = str(video_path)
    cfg["input"]["job_id"] = job_id
    cfg["output"]["output_root"] = str(_OUTPUT_ROOT)
    cfg["output"]["cleanup"] = cleanup
    cfg["detection"]["confidence_threshold"] = confidence_threshold
    if rfdetr_checkpoint:
        cfg["models"]["rfdetr_seg_checkpoint"] = rfdetr_checkpoint
    if videomae_checkpoint:
        cfg["models"]["videomae_checkpoint"] = videomae_checkpoint

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, str(video_path), cfg),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream that emits progress events until the job completes or fails."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def _generate() -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            job = store.get(job_id)
            if job is None:
                break

            if job.status == "complete":
                yield {
                    "data": json.dumps({
                        "status": "complete",
                        "result_url": f"/jobs/{job_id}/results",
                    })
                }
                break
            elif job.status == "failed":
                yield {
                    "data": json.dumps({
                        "status": "failed",
                        "error": job.error or "unknown error",
                    })
                }
                break
            elif job.last_event is not None:
                yield {"data": json.dumps(job.last_event)}

            # Yield control and wait for next update
            await asyncio.get_event_loop().run_in_executor(
                None, store.wait_for_update, job_id, 2.0
            )

    return EventSourceResponse(_generate())


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str) -> JSONResponse:
    """Poll fallback — returns current job state + last progress snapshot."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "job_id": job_id,
        "status": job.status,
        "last_event": job.last_event,
        "error": job.error,
    })


def _job_dir_exists(job_id: str) -> bool:
    """True if a completed result directory exists on disk for job_id."""
    job_dir = _OUTPUT_ROOT / job_id
    return job_dir.is_dir() and (job_dir / "annotated.mp4").exists()


@app.get("/jobs/{job_id}/results")
async def job_results(job_id: str) -> JSONResponse:
    """Return manifest of result file URLs once the job is complete."""
    job = store.get(job_id)
    if job is not None and job.status != "complete":
        raise HTTPException(status_code=409, detail=f"Job status is '{job.status}', not 'complete'")
    if job is None and not _job_dir_exists(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = _OUTPUT_ROOT / job_id
    manifest: Dict[str, Any] = {
        "annotated_video": f"/results/{job_id}/annotated.mp4",
        "activity_budget": f"/results/{job_id}/activity_budget.csv",
        "behavior_deviation": f"/results/{job_id}/behavior_deviation.csv",
        "timelines": [],
    }
    timelines_dir = job_dir / "timelines"
    if timelines_dir.exists():
        manifest["timelines"] = [
            f"/results/{job_id}/timelines/{p.name}"
            for p in sorted(timelines_dir.iterdir())
        ]
    return JSONResponse(manifest)


@app.get("/results/{job_id}/{filename:path}")
async def serve_result(job_id: str, filename: str) -> FileResponse:
    """Serve a static result file."""
    job_dir = _OUTPUT_ROOT / job_id
    job = store.get(job_id)
    if job is None and not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = job_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Prevent path traversal
    try:
        file_path.resolve().relative_to(job_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    return FileResponse(str(file_path))


@app.get("/jobs")
async def list_jobs() -> JSONResponse:
    """Return all jobs (id, status, video filename, created_at)."""
    return JSONResponse(store.list_all())


@app.post("/jobs/seed")
async def seed_job(source_dir: str = Form(...)) -> JSONResponse:
    """Register an existing result directory as a completed job (for demos/screenshots).

    source_dir: absolute or repo-relative path to a directory that contains
                activity_budget.csv, behavior_deviation.csv, timelines/, annotated.mp4
    """
    src = Path(source_dir)
    if not src.is_absolute():
        src = _REPO_ROOT / src
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Source dir not found: {src}")

    job_id = str(uuid.uuid4())
    link_path = _OUTPUT_ROOT / job_id
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(src.resolve())

    store_job_id = store.create(video_filename=src.name)
    # Overwrite the auto-generated id with our chosen one so URLs match
    with store._lock:
        job = store._jobs.pop(store_job_id)
        job.job_id = job_id
        store._jobs[job_id] = job
        store._conds[job_id] = store._conds.pop(store_job_id)
    store.complete(job_id, {})

    return JSONResponse({"job_id": job_id, "results_url": f"/jobs/{job_id}/results"})


def _parse_activity_budget(path: Path) -> List[Dict[str, Any]]:
    """Parse activity_budget.csv → list of per-track budget dicts."""
    tracks: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("track_id", "")
            if tid not in tracks:
                tracks[tid] = {"track_id": tid, "is_outlier": False, "behaviors": []}
            try:
                pct = float(row.get("pct_time", row.get("pct", 0)))
                label_id = int(row.get("label_id", -1))
            except (ValueError, TypeError):
                continue
            tracks[tid]["behaviors"].append({
                "label_id": label_id,
                "behavior": row.get("behavior", row.get("label_name", "")),
                "pct_time": pct,
            })
    return list(tracks.values())


def _parse_deviation(path: Path) -> List[Dict[str, Any]]:
    """Parse behavior_deviation.csv → list of outlier rows."""
    outlier_tracks: set = set()
    all_rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_out = row.get("is_outlier", "False").strip().lower() in ("true", "1", "yes")
            if is_out:
                try:
                    pct = float(row.get("pct_time", 0))
                    baseline = float(row.get("baseline_median", 0))
                    deviation = float(row.get("deviation", 0))
                    label_id = int(row.get("label_id", -1))
                except (ValueError, TypeError):
                    continue
                outlier_tracks.add(row.get("track_id", ""))
                all_rows.append({
                    "track_id": row.get("track_id", ""),
                    "behavior": row.get("behavior", ""),
                    "label_id": label_id,
                    "pct_time": pct,
                    "baseline_median": baseline,
                    "deviation": deviation,
                })
    return all_rows, outlier_tracks


def _parse_timelines(timelines_dir: Path) -> tuple:
    """Parse all timeline CSVs under timelines_dir → per-track segment lists."""
    by_track: Dict[str, List[Dict[str, Any]]] = {}
    for csv_path in sorted(timelines_dir.rglob("*.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("track_id", csv_path.stem)
                if tid not in by_track:
                    by_track[tid] = []
                try:
                    start_sec = float(row.get("start_sec", 0))
                    end_sec = float(row.get("end_sec", 0))
                    duration_sec = float(row.get("duration_sec", end_sec - start_sec))
                    label_id = int(row.get("label_id", -1))
                except (ValueError, TypeError):
                    continue
                by_track[tid].append({
                    "label_id": label_id,
                    "behavior": row.get("label_name", row.get("behavior", "")),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                })
    return by_track


@app.get("/jobs/{job_id}/analytics")
async def job_analytics(job_id: str) -> JSONResponse:
    """Parse result CSVs and return structured analytics data for the UI."""
    job = store.get(job_id)
    if job is not None and job.status != "complete":
        raise HTTPException(status_code=409, detail=f"Job status is '{job.status}', not 'complete'")
    if job is None and not _job_dir_exists(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = _OUTPUT_ROOT / job_id

    # Activity budget
    budget_list: List[Dict[str, Any]] = []
    budget_path = job_dir / "activity_budget.csv"
    if budget_path.exists():
        budget_list = _parse_activity_budget(budget_path)

    # Outliers
    outlier_rows: List[Dict[str, Any]] = []
    outlier_tracks: set = set()
    deviation_path = job_dir / "behavior_deviation.csv"
    if deviation_path.exists():
        outlier_rows, outlier_tracks = _parse_deviation(deviation_path)

    # Mark outlier flag on budget entries
    for entry in budget_list:
        entry["is_outlier"] = entry["track_id"] in outlier_tracks

    # Timelines
    timeline_list: List[Dict[str, Any]] = []
    total_duration = 0.0
    timelines_dir = job_dir / "timelines"
    if timelines_dir.exists():
        by_track = _parse_timelines(timelines_dir)
        for tid, segs in by_track.items():
            if segs:
                track_end = max(s["end_sec"] for s in segs)
                total_duration = max(total_duration, track_end)
            timeline_list.append({
                "track_id": tid,
                "is_outlier": tid in outlier_tracks,
                "segments": segs,
            })

    return JSONResponse({
        "activity_budget": budget_list,
        "outliers": outlier_rows,
        "timelines": timeline_list,
        "total_duration_sec": total_duration,
    })


# ── Production static file serving (SPA) ─────────────────────────────────────
# Only mounted when the built frontend exists (i.e. production mode).
# In dev mode the Vite dev server handles static assets via its own port.
_DIST_DIR = _REPO_ROOT / "ui" / "dist"

if (_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="ui-assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str, request: Request) -> FileResponse:
    """Catch-all: serve static dist files if they exist, otherwise the SPA index."""
    index = _DIST_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not built. Run: cd ui && npm run build",
        )
    # Serve real files (logos, favicons, icons) from dist root
    static_file = _DIST_DIR / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))
    return FileResponse(str(index))
