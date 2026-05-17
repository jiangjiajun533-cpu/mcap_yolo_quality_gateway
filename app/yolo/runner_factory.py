"""
Factory for selecting YOLO inference backend (onnxruntime or tensorrt).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger("yolo.runner_factory")


def create_runner(
    model_path: str | Path,
    backend: str = "onnxruntime",
    *,
    labels_path: Optional[str | Path] = None,
    target_classes: Optional[List[str]] = None,
    conf_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    input_size: int = 640,
    device: str = "cpu",
    min_box_side_px: Optional[int] = None,
):
    """
    Create a YOLO runner for the specified backend.

    Args:
        backend: "onnxruntime" (default) or "tensorrt"
    """
    if backend == "tensorrt":
        from app.yolo.trt_runner import YoloTrtRunner

        logger.info(f"Using TensorRT backend: {model_path}")
        return YoloTrtRunner(
            engine_path=model_path,
            labels_path=labels_path,
            target_classes=target_classes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            min_box_side_px=min_box_side_px,
        )
    else:
        from app.yolo.onnx_runner import YoloOnnxRunner

        logger.info(f"Using ONNX Runtime backend: {model_path}")
        return YoloOnnxRunner(
            model_path=model_path,
            labels_path=labels_path,
            target_classes=target_classes,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size,
            device=device,
            min_box_side_px=min_box_side_px,
        )
