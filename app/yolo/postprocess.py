"""
YOLO output tensor post-processing (FR-YOLO-004).

Supports YOLOv8 / YOLO11 output format:
  - shape (1, 84, 8400) — [cx, cy, w, h, cls0..cls79]

Also supports YOLOv5 format:
  - shape (1, 25200, 85) — [cx, cy, w, h, obj_conf, cls0..cls79]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from app.yolo.nms import batched_nms
from app.yolo.preprocess import LetterboxMeta, unscale_coords


@dataclass
class Detection:
    """Single detected object (FR-YOLO-004 output format)."""

    label: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
        }


def _decode_yolov8(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode YOLOv8/v11 tensor: shape (1, 84, N) or (84, N).
    Returns boxes (N,4) cxcywh, obj_scores (N,), class_ids (N,).
    """
    pred = output[0] if output.ndim == 3 else output  # (84, N)
    pred = pred.T  # (N, 84)

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]  # (N, 80)

    class_ids = class_scores.argmax(axis=1)  # (N,)
    obj_scores = class_scores.max(axis=1)  # (N,)
    return boxes_cxcywh, obj_scores, class_ids


def _decode_yolov5(output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode YOLOv5 tensor: shape (1, 25200, 85) or (25200, 85).
    Returns boxes (N,4) cxcywh, obj_scores (N,), class_ids (N,).
    """
    pred = output[0] if output.ndim == 3 else output  # (25200, 85)

    boxes_cxcywh = pred[:, :4]
    obj_conf = pred[:, 4]
    class_scores = pred[:, 5:]  # (N, 80)
    class_ids = class_scores.argmax(axis=1)
    obj_scores = obj_conf * class_scores.max(axis=1)
    return boxes_cxcywh, obj_scores, class_ids


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def postprocess(
    raw_output: np.ndarray,
    meta: LetterboxMeta,
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_class_ids: Optional[Set[int]] = None,
    min_box_side_px: int = 0,
) -> List[Detection]:
    """
    Full post-processing pipeline (FR-YOLO-004):
      1. Detect output format (YOLOv8 vs YOLOv5)
      2. Confidence filtering
      3. Class-aware batched NMS
      4. Coordinate unscale → original image pixels
      5. Target-class filtering (FR-YOLO-005)
      6. Minimum bbox size filter (optional)
      7. Clip bbox to image bounds

    Args:
        raw_output:       first output tensor from ONNX session
        meta:             LetterboxMeta from preprocess()
        class_names:      list of class label strings
        conf_threshold:   minimum confidence to keep a detection
        iou_threshold:    NMS IoU threshold
        target_class_ids: if set, only return detections in this set

    Returns:
        List of Detection objects with original-image pixel coordinates.
    """
    if raw_output is None or raw_output.size == 0:
        return []

    # --- Detect output format ---
    shape = raw_output.shape
    if raw_output.ndim == 3:
        _, dim1, dim2 = shape
        if dim1 < dim2:
            # YOLOv8: (1, 84, 8400)
            boxes_cxcywh, scores, class_ids = _decode_yolov8(raw_output)
        else:
            # YOLOv5: (1, 25200, 85)
            boxes_cxcywh, scores, class_ids = _decode_yolov5(raw_output)
    else:
        # Fallback: treat as YOLOv8 transposed
        boxes_cxcywh, scores, class_ids = _decode_yolov8(raw_output)

    # --- Confidence filter ---
    mask = scores >= conf_threshold
    if not mask.any():
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # --- cx,cy,w,h → x1,y1,x2,y2 ---
    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)

    # --- Batched NMS ---
    kept_indices = batched_nms(boxes_xyxy, scores, class_ids, iou_threshold)
    if not kept_indices:
        return []

    detections: List[Detection] = []
    for idx in kept_indices:
        cid = int(class_ids[idx])

        # Target-class filter (FR-YOLO-005)
        if target_class_ids is not None and cid not in target_class_ids:
            continue

        label = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        conf = float(scores[idx])

        bx1, by1, bx2, by2 = (
            float(boxes_xyxy[idx, 0]),
            float(boxes_xyxy[idx, 1]),
            float(boxes_xyxy[idx, 2]),
            float(boxes_xyxy[idx, 3]),
        )

        # Map back to original image coordinates
        ox1, oy1, ox2, oy2 = unscale_coords(bx1, by1, bx2, by2, meta)

        # Skip degenerate boxes
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        if min_box_side_px > 0:
            w, h = ox2 - ox1, oy2 - oy1
            if min(w, h) < min_box_side_px:
                continue

        detections.append(
            Detection(
                label=label,
                class_id=cid,
                confidence=round(conf, 4),
                x1=ox1,
                y1=oy1,
                x2=ox2,
                y2=oy2,
            )
        )

    return detections
