"""
TensorRT inference runner for YOLO models (bonus: TRT acceleration).

Requires `tensorrt` and `pycuda` packages.
Falls back gracefully if unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.core.config import settings
from app.core.errors import InferenceError, ModelLoadError, ModelNotFoundError
from app.core.logging import get_logger
from app.yolo.labels import build_target_class_ids, load_labels
from app.yolo.postprocess import Detection, postprocess
from app.yolo.preprocess import preprocess

logger = get_logger("yolo.trt_runner")

TRT_AVAILABLE = False
try:
    import tensorrt as trt  # noqa: F401
    import pycuda.driver as cuda  # noqa: F401
    import pycuda.autoinit  # noqa: F401

    TRT_AVAILABLE = True
except ImportError:
    pass


class YoloTrtRunner:
    """
    TensorRT engine runner for YOLO.

    Expects a serialised TRT engine file (`.engine` / `.trt`).
    Build one from ONNX with:
        trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
    """

    def __init__(
        self,
        engine_path: str | Path,
        labels_path: Optional[str | Path] = None,
        target_classes: Optional[List[str]] = None,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        input_size: int = 640,
        min_box_side_px: Optional[int] = None,
    ):
        if not TRT_AVAILABLE:
            raise ModelLoadError(
                "TensorRT backend requested but tensorrt/pycuda not installed. "
                "Install with: pip install tensorrt pycuda"
            )

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise ModelNotFoundError(f"TRT engine not found: {engine_path}")

        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                self._engine = runtime.deserialize_cuda_engine(f.read())
            self._context = self._engine.create_execution_context()
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load TRT engine {engine_path}: {exc}"
            ) from exc

        self._input_idx = 0
        self._output_idx = 1
        input_shape = self._engine.get_binding_shape(self._input_idx)
        output_shape = self._engine.get_binding_shape(self._output_idx)

        self._input_shape = list(input_shape)
        self._output_shape = list(output_shape)
        self._input_size_bytes = int(np.prod(input_shape) * np.float32().nbytes)
        self._output_size_bytes = int(np.prod(output_shape) * np.float32().nbytes)

        self._d_input = cuda.mem_alloc(self._input_size_bytes)
        self._d_output = cuda.mem_alloc(self._output_size_bytes)
        self._stream = cuda.Stream()
        self._h_output = np.empty(output_shape, dtype=np.float32)

        self.model_name = engine_path.stem
        self.model_path = engine_path
        self.conf_threshold = (
            settings.conf_threshold if conf_threshold is None else conf_threshold
        )
        self.nms_threshold = (
            settings.nms_threshold if nms_threshold is None else nms_threshold
        )
        self.min_box_side_px = (
            settings.min_box_side_px if min_box_side_px is None else min_box_side_px
        )
        self.input_size = input_size
        self.device = "gpu"

        self.class_names: List[str] = load_labels(labels_path)
        self.target_class_ids: Set[int] = build_target_class_ids(
            self.class_names, target_classes
        )

        logger.info(
            f"YoloTrtRunner: engine={self.model_name} "
            f"input={self._input_shape} output={self._output_shape} "
            f"classes={len(self.class_names)} targets={len(self.target_class_ids)}"
        )

    def infer(self, img: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        t_start = time.perf_counter()

        try:
            tensor, meta = preprocess(img, self.input_size)
        except Exception as exc:
            raise InferenceError(f"Preprocess failed: {exc}") from exc
        t_pre = time.perf_counter()

        try:
            h_input = np.ascontiguousarray(tensor)
            cuda.memcpy_htod_async(self._d_input, h_input, self._stream)
            self._context.execute_async_v2(
                bindings=[int(self._d_input), int(self._d_output)],
                stream_handle=self._stream.handle,
            )
            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self._stream)
            self._stream.synchronize()
        except Exception as exc:
            raise InferenceError(f"TRT inference failed: {exc}") from exc
        t_inf = time.perf_counter()

        try:
            detections = postprocess(
                raw_output=self._h_output,
                meta=meta,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.nms_threshold,
                target_class_ids=self.target_class_ids,
                min_box_side_px=self.min_box_side_px,
            )
        except Exception as exc:
            raise InferenceError(f"Postprocess failed: {exc}") from exc
        t_end = time.perf_counter()

        latency = {
            "preprocess_ms": round((t_pre - t_start) * 1000, 2),
            "inference_ms": round((t_inf - t_pre) * 1000, 2),
            "postprocess_ms": round((t_end - t_inf) * 1000, 2),
            "total_ms": round((t_end - t_start) * 1000, 2),
        }
        return detections, latency

    def model_info(self) -> dict:
        return {
            "name": self.model_name,
            "format": "tensorrt",
            "input_size": [self.input_size, self.input_size],
            "backend": "tensorrt",
            "device": "gpu",
        }
