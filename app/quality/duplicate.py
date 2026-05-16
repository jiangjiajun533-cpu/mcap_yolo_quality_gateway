"""
Near-duplicate frame detection using perceptual hashing (FR-SEQ-003, bonus).
Uses a simple average-hash (aHash) which is fast and requires no extra deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger("quality.duplicate")

_HASH_SIZE = 8          # 8×8 = 64-bit hash
_DUP_THRESHOLD = 10     # Hamming distance ≤ this → near-duplicate


def _ahash(img: np.ndarray, hash_size: int = _HASH_SIZE) -> np.ndarray:
    """Compute average hash (aHash): returns bool array of length hash_size²."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    return (small > mean).flatten()


def _hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


@dataclass
class DuplicateGroup:
    start_frame_seq: int
    end_frame_seq: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    reason: str = "near-identical frames"

    @property
    def duration_sec(self) -> float:
        return (self.end_timestamp_ns - self.start_timestamp_ns) / 1e9


class DuplicateDetector:
    """
    Stateful per-topic duplicate detector.
    Feed frames via update(); call finalize() for results.
    """

    def __init__(self, threshold: int = _DUP_THRESHOLD):
        self.threshold = threshold
        self._hashes: List[np.ndarray] = []
        self._seqs: List[int] = []
        self._timestamps: List[int] = []

    def update(self, img: np.ndarray, frame_seq: int, timestamp_ns: int) -> None:
        h = _ahash(img)
        self._hashes.append(h)
        self._seqs.append(frame_seq)
        self._timestamps.append(timestamp_ns)

    def finalize(self) -> List[DuplicateGroup]:
        """Return list of duplicate groups (consecutive near-identical frames)."""
        if len(self._hashes) < 2:
            return []

        groups: List[DuplicateGroup] = []
        in_group = False
        group_start_idx = 0

        for i in range(1, len(self._hashes)):
            dist = _hamming(self._hashes[i - 1], self._hashes[i])
            is_dup = dist <= self.threshold

            if is_dup and not in_group:
                in_group = True
                group_start_idx = i - 1
            elif not is_dup and in_group:
                in_group = False
                if i - 1 > group_start_idx:  # at least 2 consecutive dups
                    groups.append(DuplicateGroup(
                        start_frame_seq=self._seqs[group_start_idx],
                        end_frame_seq=self._seqs[i - 1],
                        start_timestamp_ns=self._timestamps[group_start_idx],
                        end_timestamp_ns=self._timestamps[i - 1],
                    ))

        # Close open group at end
        if in_group and len(self._hashes) - 1 > group_start_idx:
            groups.append(DuplicateGroup(
                start_frame_seq=self._seqs[group_start_idx],
                end_frame_seq=self._seqs[-1],
                start_timestamp_ns=self._timestamps[group_start_idx],
                end_timestamp_ns=self._timestamps[-1],
            ))

        return groups


def duplicate_groups_to_dict(topic: str, groups: List[DuplicateGroup]) -> dict:
    return {
        "topic": topic,
        "duplicate_frame_groups": [
            {
                "start_frame_seq": g.start_frame_seq,
                "end_frame_seq": g.end_frame_seq,
                "duration_sec": round(g.duration_sec, 3),
                "reason": g.reason,
            }
            for g in groups
        ],
    }
