"""GET /health endpoint (FR-API-002)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings

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
