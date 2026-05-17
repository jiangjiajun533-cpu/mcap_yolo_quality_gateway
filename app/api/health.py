"""GET /health endpoint (FR-API-002)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings
from app.core.paths import path_hints

router = APIRouter()


@router.get("/health")
def health_check():
    model_loaded = Path(settings.model_path).exists()
    mcap_reader_available = True
    try:
        import mcap  # noqa: F401
    except ImportError:
        mcap_reader_available = False

    yolo_backend = "onnxruntime"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        yolo_backend = "unavailable"

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "mcap_reader_available": mcap_reader_available,
        "yolo_backend": yolo_backend,
    }


@router.get("/path_hints")
def get_path_hints():
    """Dashboard: how to fill MCAP / output paths for this server (Docker vs local)."""
    return path_hints()
