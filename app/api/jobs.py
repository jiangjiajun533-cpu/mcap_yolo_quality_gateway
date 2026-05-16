"""GET /jobs/{job_id} endpoint (FR-API-006)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.jobs.manager import job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@router.get("")
def list_jobs():
    return job_manager.list_jobs()
