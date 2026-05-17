"""
MCAP-related API endpoints (FR-API-003, FR-API-004, FR-API-005).

POST /mcap/inspect       — parse MCAP metadata
POST /mcap/quality_scan  — launch async quality scan job
POST /mcap/yolo_infer    — launch async YOLO inference job
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.paths import resolve_mcap_path
from app.mcap_io.reader import read_mcap_summary
from app.jobs.manager import job_manager
from app.jobs.worker import launch_worker

router = APIRouter(prefix="/mcap", tags=["mcap"])

_UPLOAD_DIR = Path("test_data")


@router.post("/upload")
async def upload_mcap(file: UploadFile = File(...)):
    """Accept an uploaded MCAP file and save to test_data/."""
    if not file.filename or not file.filename.lower().endswith(".mcap"):
        raise HTTPException(status_code=400, detail="Only .mcap files are accepted")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"mcap_path": str(dest.resolve()), "filename": file.filename}


# ── FR-API-003: /mcap/inspect ─────────────────────────────────────────────

class InspectRequest(BaseModel):
    mcap_path: str

@router.post("/inspect")
def mcap_inspect(req: InspectRequest):
    p = resolve_mcap_path(req.mcap_path)
    if not p.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"MCAP file not found: {req.mcap_path} (resolved: {p})",
        )

    summary = read_mcap_summary(p)
    return {
        "mcap_file": summary.mcap_file,
        "duration_sec": round(summary.duration_sec, 3),
        "topics": [
            {
                "topic": t.topic,
                "message_type": t.message_type,
                "message_count": t.message_count,
                "is_image_topic": t.is_image_topic,
            }
            for t in summary.topics
        ],
    }


# ── FR-API-004: /mcap/quality_scan ────────────────────────────────────────

class QualityScanRequest(BaseModel):
    mcap_path: str
    topics: Optional[List[str]] = None
    sample_every_n: int = Field(default=1, ge=1)
    target_fps: float = Field(default=0.0, ge=0.0)
    start_sec: float = Field(default=0.0, ge=0.0, description="Relative start sec from MCAP beginning (FR-MCAP-003)")
    end_sec: float = Field(default=0.0, ge=0.0, description="Relative end sec; 0 = no upper limit")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_frames: int = Field(default=0, ge=0)
    output_dir: Optional[str] = None

@router.post("/quality_scan")
def mcap_quality_scan(req: QualityScanRequest):
    p = resolve_mcap_path(req.mcap_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"MCAP file not found: {req.mcap_path}")

    job = job_manager.create("quality_scan", params=req.model_dump())
    if req.output_dir is None:
        job.params["output_dir"] = str(Path(settings.output_dir) / job.job_id)
    launch_worker(job)

    return {"job_id": job.job_id, "status": job.status.value}


# ── FR-API-005: /mcap/yolo_infer ──────────────────────────────────────────

class YoloInferRequest(BaseModel):
    mcap_path: str
    topics: Optional[List[str]] = None
    model_path: str = Field(default="models/yolov8n.onnx")
    labels_path: str = Field(default="models/coco_classes.txt")
    target_classes: Optional[List[str]] = None
    sample_every_n: int = Field(default=1, ge=1)
    target_fps: float = Field(default=0.0, ge=0.0)
    start_sec: float = Field(default=0.0, ge=0.0, description="Relative start sec from MCAP beginning (FR-MCAP-003)")
    end_sec: float = Field(default=0.0, ge=0.0, description="Relative end sec; 0 = no upper limit")
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    conf_threshold: float = Field(default=settings.conf_threshold, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=settings.nms_threshold, ge=0.0, le=1.0)
    min_box_side_px: int = Field(default=settings.min_box_side_px, ge=0)
    skip_depth_topics_for_yolo: bool = Field(default=settings.skip_depth_topics_for_yolo)
    infer_low_quality: bool = False
    max_frames: int = Field(default=0, ge=0)
    output_dir: Optional[str] = None

@router.post("/yolo_infer")
def mcap_yolo_infer(req: YoloInferRequest):
    p = resolve_mcap_path(req.mcap_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"MCAP file not found: {req.mcap_path}")

    job = job_manager.create("yolo_infer", params=req.model_dump())
    if req.output_dir is None:
        job.params["output_dir"] = str(Path(settings.output_dir) / job.job_id)
    launch_worker(job)

    return {"job_id": job.job_id, "status": job.status.value}
