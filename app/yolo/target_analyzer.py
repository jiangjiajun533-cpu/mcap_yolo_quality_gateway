"""
Target-level quality impact analysis (FR-YOLO-008).

For each detected class, accumulates:
  - total detected count
  - average confidence
  - average quality score of frames where it was detected
  - count in low-quality frames vs normal-quality frames
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from app.yolo.pipeline import InferenceRecord


@dataclass
class TargetStats:
    label: str
    class_id: int
    detected_count: int = 0
    _confidences: List[float] = field(default_factory=list, repr=False)
    _quality_scores: List[float] = field(default_factory=list, repr=False)
    low_quality_frame_detected_count: int = 0
    normal_quality_frame_detected_count: int = 0

    @property
    def avg_confidence(self) -> float:
        return round(float(np.mean(self._confidences)), 4) if self._confidences else 0.0

    @property
    def avg_quality_score(self) -> float:
        return round(float(np.mean(self._quality_scores)), 4) if self._quality_scores else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "detected_count": self.detected_count,
            "avg_confidence": self.avg_confidence,
            "avg_quality_score": self.avg_quality_score,
            "low_quality_frame_detected_count": self.low_quality_frame_detected_count,
            "normal_quality_frame_detected_count": self.normal_quality_frame_detected_count,
        }


class TargetAnalyzer:
    """
    Feed InferenceRecords to accumulate per-class detection statistics.
    Call finalize() to get the FR-YOLO-008 output dict.
    """

    def __init__(self) -> None:
        self._stats: Dict[int, TargetStats] = {}   # class_id → TargetStats

    def update(self, record: InferenceRecord) -> None:
        """Process one InferenceRecord (inferred or skip_inference)."""
        if not record.objects:
            return

        for det in record.objects:
            cid = det.class_id
            if cid not in self._stats:
                self._stats[cid] = TargetStats(label=det.label, class_id=cid)

            s = self._stats[cid]
            s.detected_count += 1
            s._confidences.append(det.confidence)
            s._quality_scores.append(record.quality_score)

            if record.is_bad_quality:
                s.low_quality_frame_detected_count += 1
            else:
                s.normal_quality_frame_detected_count += 1

    def finalize(self) -> dict:
        """Return FR-YOLO-008 format dict."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.detected_count,
            reverse=True,
        )
        return {
            "target_analysis": [s.to_dict() for s in sorted_stats]
        }
