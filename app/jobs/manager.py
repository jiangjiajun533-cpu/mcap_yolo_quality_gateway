"""
In-memory job state machine for async task management.

States: pending → running → finished | failed
Thread-safe via a lock protecting the shared dict.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    job_type: str                        # "quality_scan" | "yolo_infer"
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0                # 0.0 – 1.0
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    result_path: str = ""
    report_path: str = ""
    error: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress": round(self.progress, 4),
            "created_at": self.created_at,
        }
        if self.started_at:
            d["started_at"] = self.started_at
        if self.finished_at:
            d["finished_at"] = self.finished_at
            d["elapsed_sec"] = round(self.finished_at - self.started_at, 3)
        if self.result_path:
            d["result_path"] = self.result_path
        if self.report_path:
            d["report_path"] = self.report_path
        if self.error:
            d["error"] = self.error
        return d


class JobManager:
    """Thread-safe singleton job registry."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_type: str, params: Optional[dict] = None) -> Job:
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            job_type=job_type,
            created_at=time.time(),
            params=params or {},
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

    def set_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = min(1.0, max(0.0, progress))

    def set_finished(
        self,
        job_id: str,
        result_path: str = "",
        report_path: str = "",
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FINISHED
                job.progress = 1.0
                job.finished_at = time.time()
                job.result_path = result_path
                job.report_path = report_path

    def set_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.error = error

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]


job_manager = JobManager()
