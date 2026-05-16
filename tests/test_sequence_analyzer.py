"""Tests for sequence analyzer (FR-SEQ-001, FR-SEQ-002)."""
import pytest

from app.quality.sequence_analyzer import TopicSequenceTracker, SequenceSummary


class TestTopicSequenceTracker:
    def test_single_frame(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=200, jump_threshold_ms=500)
        t.update(1_000_000_000, 640, 480)
        s = t.finalize()
        assert s.total_frames == 1
        assert s.estimated_fps == 0.0
        assert s.duration_sec == 0.0

    def test_constant_fps(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=200, jump_threshold_ms=500)
        interval_ns = int(1e9 / 30)  # 30fps
        for i in range(100):
            t.update(1_000_000_000 + i * interval_ns, 640, 480)
        s = t.finalize()
        assert 28 < s.estimated_fps < 32
        assert 30 < s.frame_interval_ms_avg < 35
        assert s.timestamp_jump_count == 0
        assert s.long_gap_count == 0

    def test_long_gap_detected(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=100, jump_threshold_ms=500)
        t.update(1_000_000_000, 640, 480)
        t.update(1_500_000_000, 640, 480)  # 500ms gap
        s = t.finalize()
        assert s.long_gap_count >= 1

    def test_resolution_change(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=200, jump_threshold_ms=500)
        t.update(1_000_000_000, 640, 480)
        t.update(1_033_000_000, 640, 480)
        t.update(1_066_000_000, 320, 240)  # resolution change
        s = t.finalize()
        assert s.resolution_change_count == 1
        assert len(s.resolutions_seen) == 2
        assert (640, 480) in s.resolutions_seen
        assert (320, 240) in s.resolutions_seen

    def test_timestamp_backward(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=200, jump_threshold_ms=500)
        t.update(2_000_000_000, 640, 480)
        t.update(1_000_000_000, 640, 480)  # backward
        s = t.finalize()
        assert s.timestamp_jump_count >= 1
        codes = [w.code for w in s.warnings]
        assert "TIMESTAMP_BACKWARD" in codes

    def test_no_warnings_for_normal_sequence(self):
        t = TopicSequenceTracker("/cam", gap_threshold_ms=200, jump_threshold_ms=500)
        interval_ns = int(1e9 / 30)
        for i in range(50):
            t.update(1_000_000_000 + i * interval_ns, 1280, 720)
        s = t.finalize()
        assert s.timestamp_jump_count == 0
        assert s.long_gap_count == 0
        assert s.resolution_change_count == 0
