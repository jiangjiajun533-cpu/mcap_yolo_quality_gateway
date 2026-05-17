"""
Draw bounding boxes, labels, and confidence scores on images (FR-REPORT-005).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.yolo.postprocess import Detection

# Colour palette (BGR) — cycles through 20 distinct colours
_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def _color_for_class(class_id: int):
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(
    img: np.ndarray,
    detections: List[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bbox, label, and confidence on a copy of img.
    Returns the annotated copy (does NOT modify the input).
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for det in detections:
        color = _color_for_class(det.class_id)
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        # Label text
        text = f"{det.label} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )
        # Background rectangle for text
        ty = max(y1 - th - baseline, 0)
        cv2.rectangle(out, (x1, ty), (x1 + tw, ty + th + baseline), color, -1)
        cv2.putText(
            out,
            text,
            (x1, ty + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_thickness,
            cv2.LINE_AA,
        )
    return out
