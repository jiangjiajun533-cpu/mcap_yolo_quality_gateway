"""
Hand-written Non-Maximum Suppression (FR-YOLO-004).
No cv2.dnn or external NMS — implements IoU from scratch as required.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_iou(box_a: Tuple[float, float, float, float],
                box_b: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU of two bboxes given as (x1, y1, x2, y2).
    Returns value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def nms(
    boxes: np.ndarray,       # (N, 4)  float, x1y1x2y2
    scores: np.ndarray,      # (N,)    float, confidence
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Classic greedy NMS.

    Args:
        boxes:         shape (N, 4), each row [x1, y1, x2, y2]
        scores:        shape (N,)
        iou_threshold: suppress boxes whose IoU with kept box exceeds this

    Returns:
        List of kept indices (sorted by descending score).
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = scores.argsort()[::-1].tolist()
    kept: List[int] = []

    while order:
        best = order.pop(0)
        kept.append(best)
        remaining = []
        for idx in order:
            iou = compute_iou(
                tuple(boxes[best].tolist()),
                tuple(boxes[idx].tolist()),
            )
            if iou <= iou_threshold:
                remaining.append(idx)
        order = remaining

    return kept


def batched_nms(
    boxes: np.ndarray,       # (N, 4)
    scores: np.ndarray,      # (N,)
    class_ids: np.ndarray,   # (N,)  int
    iou_threshold: float = 0.45,
) -> List[int]:
    """
    Per-class NMS: applies NMS independently for each class.
    Uses an offset trick: shift boxes by class_id * large_offset so that
    boxes from different classes never suppress each other.
    """
    if len(boxes) == 0:
        return []

    offset = class_ids.astype(np.float32) * 4096.0  # large enough offset
    shifted_boxes = boxes.copy()
    shifted_boxes[:, 0] += offset
    shifted_boxes[:, 1] += offset
    shifted_boxes[:, 2] += offset
    shifted_boxes[:, 3] += offset

    return nms(shifted_boxes, scores, iou_threshold)
