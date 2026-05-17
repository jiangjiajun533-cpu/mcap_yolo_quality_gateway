"""
FastAPI application entry point (FR-API-001).

Start with:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.mcap import router as mcap_router
from app.api.jobs import router as jobs_router
from app.api.yolo import router as yolo_preview_router
from app.api.pipeline_review import router as pipeline_review_router
from app.api.metrics import router as metrics_router
from app.core.paths import OUTPUTS_ROOT, PROJECT_ROOT, resolve_output_dir

_PROJECT_ROOT = PROJECT_ROOT


app = FastAPI(
    title="MCAP YOLO Quality Gateway",
    description=(
        "MCAP image quality assessment and YOLO object detection pipeline. "
        "Supports async job submission, quality scanning, and YOLO inference."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(mcap_router)
app.include_router(jobs_router)
app.include_router(yolo_preview_router)
app.include_router(pipeline_review_router)
app.include_router(metrics_router)

# Serve generated reports and sample images (always from project outputs/)
_outputs = OUTPUTS_ROOT
_outputs.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(_outputs), html=False), name="outputs")

# Serve dashboard static files
_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/report/quality", include_in_schema=False)
async def report_quality(output_dir: str = Query("outputs/yolo_test_v2")):
    root = resolve_output_dir(output_dir)
    p = root / "quality_report.html"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {p}")
    return FileResponse(str(p), media_type="text/html")


@app.get("/report/yolo", include_in_schema=False)
async def report_yolo(output_dir: str = Query("outputs/yolo_test_v2")):
    root = resolve_output_dir(output_dir)
    p = root / "yolo_report.html"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {p}")
    return FileResponse(str(p), media_type="text/html")


@app.get("/report/metrics", include_in_schema=False)
async def report_metrics(output_dir: str = Query("outputs/yolo_test_v2")):
    root = resolve_output_dir(output_dir)
    p = root / "metrics.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Metrics not found: {p}")
    return FileResponse(str(p), media_type="application/json")


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the interactive dashboard."""
    index = _static / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "Dashboard not found. Place index.html in app/static/"}
