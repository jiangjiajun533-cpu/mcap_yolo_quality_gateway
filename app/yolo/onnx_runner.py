"""
ONNX Runtime inference runner for YOLO models (FR-YOLO-001).
Handles model loading, input name discovery, inference timing.
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
from app.yolo.preprocess import LetterboxMeta, preprocess

logger = get_logger("yolo.onnx_runner")


class YoloOnnxRunner:
    """
    Wraps an ONNX Runtime session for YOLO inference.
    Thread-safe for reading (session is read-only after init).
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: Optional[str | Path] = None,
        target_classes: Optional[List[str]] = None,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        input_size: int = 640,
        device: str = "cpu",
        min_box_side_px: Optional[int] = None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(f"YOLO model not found: {model_path}")

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 0   # 0 = use all cores
            opts.inter_op_num_threads = 0
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device == "gpu" else ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=providers,
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to load ONNX model {model_path}: {exc}") from exc

        self._input_name: str = self._session.get_inputs()[0].name
        self._input_shape: List[int] = list(self._session.get_inputs()[0].shape)

        self.model_name = model_path.stem
        self.model_path = model_path
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
        self.device = device

        self.class_names: List[str] = load_labels(labels_path)
        self.target_class_ids: Set[int] = build_target_class_ids(
            self.class_names, target_classes
        )

        logger.info(
            f"YoloOnnxRunner: model={self.model_name} "
            f"input={self._input_name}{self._input_shape} "
            f"classes={len(self.class_names)} targets={len(self.target_class_ids)} "
            f"device={device}"
        )

    def infer(self, img: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        """
        Run full inference pipeline on a BGR numpy image.

        Returns:
            detections: list of Detection objects (original image coords)
            latency:    dict with preprocess_ms, inference_ms, postprocess_ms, total_ms
        """
        t_start = time.perf_counter()

        # --- Preprocess ---
        try:
            tensor, meta = preprocess(img, self.input_size)
        except Exception as exc:
            raise InferenceError(f"Preprocess failed: {exc}") from exc
        t_pre = time.perf_counter()

        # --- Inference ---
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc
        t_inf = time.perf_counter()

        # --- Postprocess ---
        try:
            detections = postprocess(
                raw_output=outputs[0],
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
            "inference_ms":  round((t_inf - t_pre)   * 1000, 2),
            "postprocess_ms": round((t_end - t_inf)  * 1000, 2),
            "total_ms":      round((t_end - t_start) * 1000, 2),
        }
        return detections, latency

    def model_info(self) -> dict:
        """Return model metadata dict for report output (FR-YOLO-006)."""
        return {
            "name": self.model_name,
            "format": "onnx",
            "input_size": [self.input_size, self.input_size],
            "backend": "onnxruntime",
            "device": self.device,
        }
