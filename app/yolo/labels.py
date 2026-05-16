"""
COCO class label management and target-class filtering (FR-YOLO-005).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from app.core.config import settings
from app.core.errors import ModelNotFoundError
from app.core.logging import get_logger

logger = get_logger("yolo.labels")

# Fallback COCO-80 class names (used when labels file is absent)
COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_labels(labels_path: Optional[str | Path] = None) -> List[str]:
    """
    Load class names from a text file (one class per line).
    Falls back to built-in COCO-80 if file not found or not specified.
    """
    path = Path(labels_path) if labels_path else None

    if path and path.exists():
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        logger.info(f"Loaded {len(names)} class labels from {path}")
        return names

    if path and not path.exists():
        logger.warning(f"Labels file not found: {path}. Using built-in COCO-80.")

    return list(COCO80_CLASSES)


def build_target_class_ids(
    class_names: List[str],
    target_classes: Optional[List[str]] = None,
) -> Set[int]:
    """
    Return the set of class_ids that match the target_classes list.
    If target_classes is None or empty, all classes are included.
    Matching is case-insensitive.
    """
    if not target_classes:
        return set(range(len(class_names)))

    target_lower = {t.lower().strip() for t in target_classes}
    ids: Set[int] = set()
    for i, name in enumerate(class_names):
        if name.lower().strip() in target_lower:
            ids.add(i)

    missing = target_lower - {class_names[i].lower().strip() for i in ids}
    if missing:
        logger.warning(f"Target classes not found in label list: {missing}")

    logger.info(f"Target class IDs: {sorted(ids)} ({len(ids)} classes)")
    return ids
