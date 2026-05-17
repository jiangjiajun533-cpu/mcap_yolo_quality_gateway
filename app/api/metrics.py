"""Prometheus-compatible /metrics endpoint (bonus: monitoring)."""

from __future__ import annotations

import threading
import time

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()


class _Counters:
    """Simple thread-safe counters for Prometheus exposition."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, float] = {
            "jobs_submitted_total": 0,
            "jobs_completed_total": 0,
            "jobs_failed_total": 0,
            "frames_sampled_total": 0,
            "frames_inferred_total": 0,
            "frames_skipped_quality_total": 0,
            "detections_total": 0,
        }
        self._start = time.time()

    def inc(self, name: str, value: float = 1):
        with self._lock:
            self._data[name] = self._data.get(name, 0) + value

    def set_val(self, name: str, value: float):
        with self._lock:
            self._data[name] = value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            d = dict(self._data)
        d["uptime_seconds"] = round(time.time() - self._start, 1)
        return d


counters = _Counters()


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Expose metrics in Prometheus text exposition format."""
    snap = counters.snapshot()
    lines: list[str] = []
    for key, val in sorted(snap.items()):
        prom_name = f"mcap_gateway_{key}"
        lines.append(f"# TYPE {prom_name} gauge")
        lines.append(f"{prom_name} {val}")
    lines.append("")
    return "\n".join(lines)
