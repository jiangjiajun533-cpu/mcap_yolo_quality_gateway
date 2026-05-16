"""
Video sequence quality analysis per topic (FR-SEQ-001, FR-SEQ-002).
Detects: frame-rate estimation, timestamp jumps, long gaps, resolution changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("quality.sequence_analyzer")


@dataclass
class SequenceWarning:
    """A single sequence-level warning event."""
    level: str          # "warn" | "error"
    code: str           # e.g. FRAME_TIME_GAP, TIMESTAMP_JUMP, RESOLUTION_CHANGED
    topic: str
    message: str
    timestamp_ns: int


@dataclass
class SequenceSummary:
    """Per-topic video sequence statistics (FR-SEQ-001, FR-SEQ-002)."""
    topic: str
    duration_sec: float = 0.0
    total_frames: int = 0
    estimated_fps: float = 0.0
    frame_interval_ms_avg: float = 0.0
    frame_interval_ms_p95: float = 0.0
    frame_interval_ms_min: float = 0.0
    frame_interval_ms_max: float = 0.0
    timestamp_jump_count: int = 0
    long_gap_count: int = 0
    resolution_change_count: int = 0
    warnings: List[SequenceWarning] = field(default_factory=list)
    resolutions_seen: List[Tuple[int, int]] = field(default_factory=list)  # (w, h)


class TopicSequenceTracker:
    """
    Stateful per-topic tracker. Feed frames one by one via update().
    Call finalize() to get the SequenceSummary.
    """

    def __init__(
        self,
        topic: str,
        gap_threshold_ms: Optional[float] = None,
        jump_threshold_ms: Optional[float] = None,
    ):
        self.topic = topic
        self.gap_threshold_ms = gap_threshold_ms or settings.frame_gap_threshold_ms
        self.jump_threshold_ms = jump_threshold_ms or settings.timestamp_jump_threshold_ms

        self._timestamps_ns: List[int] = []
        self._resolutions: List[Tuple[int, int]] = []
        self._current_resolution: Optional[Tuple[int, int]] = None
        self._warnings: List[SequenceWarning] = []
        self._resolution_change_count = 0

    def update(self, timestamp_ns: int, width: int, height: int) -> None:
        """Record one frame's timestamp and resolution."""
        self._timestamps_ns.append(timestamp_ns)

        res = (width, height)
        if self._current_resolution is None:
            self._current_resolution = res
            self._resolutions.append(res)
        elif res != self._current_resolution:
            self._resolution_change_count += 1
            self._warnings.append(SequenceWarning(
                level="warn",
                code="RESOLUTION_CHANGED",
                topic=self.topic,
                message=(
                    f"resolution changed from "
                    f"{self._current_resolution[0]}x{self._current_resolution[1]} "
                    f"to {res[0]}x{res[1]}"
                ),
                timestamp_ns=timestamp_ns,
            ))
            self._current_resolution = res
            if res not in self._resolutions:
                self._resolutions.append(res)

    def finalize(self) -> SequenceSummary:
        """Compute all statistics from accumulated data."""
        summary = SequenceSummary(topic=self.topic)
        summary.total_frames = len(self._timestamps_ns)
        summary.warnings = self._warnings
        summary.resolutions_seen = list(self._resolutions)
        summary.resolution_change_count = self._resolution_change_count

        if summary.total_frames < 2:
            return summary

        ts = np.array(self._timestamps_ns, dtype=np.int64)
        ts_sorted = np.sort(ts)

        duration_ns = int(ts_sorted[-1] - ts_sorted[0])
        summary.duration_sec = duration_ns / 1e9

        # Frame intervals in ms
        intervals_ms = np.diff(ts_sorted).astype(np.float64) / 1e6

        summary.frame_interval_ms_avg = float(np.mean(intervals_ms))
        summary.frame_interval_ms_p95 = float(np.percentile(intervals_ms, 95))
        summary.frame_interval_ms_min = float(np.min(intervals_ms))
        summary.frame_interval_ms_max = float(np.max(intervals_ms))

        if summary.duration_sec > 0:
            summary.estimated_fps = round(
                (summary.total_frames - 1) / summary.duration_sec, 2
            )

        # Detect long gaps
        long_gap_mask = intervals_ms > self.gap_threshold_ms
        for idx in np.where(long_gap_mask)[0]:
            summary.long_gap_count += 1
            self._warnings.append(SequenceWarning(
                level="warn",
                code="FRAME_TIME_GAP",
                topic=self.topic,
                message=(
                    f"frame interval {intervals_ms[idx]:.0f}ms is larger than "
                    f"threshold {self.gap_threshold_ms:.0f}ms"
                ),
                timestamp_ns=int(ts_sorted[idx + 1]),
            ))

        # Detect timestamp jumps (backwards or large forward)
        raw_intervals = np.diff(ts).astype(np.float64) / 1e6  # preserve original order
        for i, iv in enumerate(raw_intervals):
            if iv < 0:
                summary.timestamp_jump_count += 1
                self._warnings.append(SequenceWarning(
                    level="warn",
                    code="TIMESTAMP_BACKWARD",
                    topic=self.topic,
                    message=f"timestamp went backward by {abs(iv):.1f}ms",
                    timestamp_ns=int(ts[i + 1]),
                ))
            elif iv > self.jump_threshold_ms:
                summary.timestamp_jump_count += 1
                self._warnings.append(SequenceWarning(
                    level="warn",
                    code="TIMESTAMP_JUMP",
                    topic=self.topic,
                    message=f"timestamp jumped forward {iv:.0f}ms",
                    timestamp_ns=int(ts[i + 1]),
                ))

        return summary


def sequence_summary_to_dict(s: SequenceSummary) -> dict:
    """Serialize SequenceSummary to JSON-compatible dict."""
    return {
        "topic": s.topic,
        "duration_sec": round(s.duration_sec, 3),
        "total_frames": s.total_frames,
        "estimated_fps": s.estimated_fps,
        "frame_interval_ms_avg": round(s.frame_interval_ms_avg, 2),
        "frame_interval_ms_p95": round(s.frame_interval_ms_p95, 2),
        "frame_interval_ms_min": round(s.frame_interval_ms_min, 2),
        "frame_interval_ms_max": round(s.frame_interval_ms_max, 2),
        "timestamp_jump_count": s.timestamp_jump_count,
        "long_gap_count": s.long_gap_count,
        "resolution_change_count": s.resolution_change_count,
        "resolutions_seen": [f"{w}x{h}" for w, h in s.resolutions_seen],
        "warnings": [
            {
                "level": w.level,
                "code": w.code,
                "topic": w.topic,
                "message": w.message,
                "timestamp_ns": w.timestamp_ns,
            }
            for w in s.warnings
        ],
    }
