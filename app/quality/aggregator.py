"""
Per-topic quality aggregator (FR-QUALITY-003).
Collects QualityResult objects and produces the per-topic summary dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.quality.scoring import QualityResult


@dataclass
class TopicQualitySummary:
    """Aggregated quality stats for one Topic (FR-QUALITY-003)."""

    topic: str
    message_type: str = ""
    total_frames: int = 0  # original message count in MCAP
    processed_frames: int = 0  # actually decoded + analyzed
    decode_failed_frames: int = 0
    bad_quality_frames: int = 0

    # Running accumulators (not serialized directly)
    _scores: List[float] = field(default_factory=list, repr=False)
    _decode_ms: List[float] = field(default_factory=list, repr=False)
    _issue_counts: Dict[str, int] = field(default_factory=dict, repr=False)
    _worst_frames: List[QualityResult] = field(default_factory=list, repr=False)

    def add(self, result: QualityResult, decode_ms: float = 0.0) -> None:
        self.processed_frames += 1
        self._scores.append(result.quality_score)
        self._decode_ms.append(decode_ms)

        if result.is_bad_quality:
            self.bad_quality_frames += 1

        for tag in result.quality_tags:
            if tag != "normal":
                self._issue_counts[tag] = self._issue_counts.get(tag, 0) + 1

        # Keep worst frames (sorted by quality_score ascending)
        self._worst_frames.append(result)
        self._worst_frames.sort(key=lambda r: r.quality_score)
        if len(self._worst_frames) > 20:
            self._worst_frames = self._worst_frames[:20]

    def add_decode_failure(self) -> None:
        self.processed_frames += 1
        self.decode_failed_frames += 1
        self.bad_quality_frames += 1
        self._scores.append(0.0)

    @property
    def bad_quality_ratio(self) -> float:
        if self.processed_frames == 0:
            return 0.0
        return round(self.bad_quality_frames / self.processed_frames, 4)

    @property
    def avg_quality_score(self) -> float:
        if not self._scores:
            return 0.0
        return round(float(np.mean(self._scores)), 4)

    @property
    def p50_quality_score(self) -> float:
        if not self._scores:
            return 0.0
        return round(float(np.percentile(self._scores, 50)), 4)

    @property
    def p95_decode_ms(self) -> float:
        if not self._decode_ms:
            return 0.0
        return round(float(np.percentile(self._decode_ms, 95)), 2)

    @property
    def quality_issue_counts(self) -> Dict[str, int]:
        return dict(self._issue_counts)

    @property
    def top_worst_frames(self) -> List[QualityResult]:
        return self._worst_frames[:20]

    def to_dict(self) -> dict:
        """FR-QUALITY-003 output format."""
        return {
            "topic": self.topic,
            "message_type": self.message_type,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "decode_failed_frames": self.decode_failed_frames,
            "bad_quality_frames": self.bad_quality_frames,
            "bad_quality_ratio": self.bad_quality_ratio,
            "avg_quality_score": self.avg_quality_score,
            "p50_quality_score": self.p50_quality_score,
            "p95_decode_ms": self.p95_decode_ms,
            "quality_issue_counts": self.quality_issue_counts,
        }
