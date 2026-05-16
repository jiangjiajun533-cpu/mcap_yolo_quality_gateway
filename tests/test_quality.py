"""Tests for quality analyzer (FR-QUALITY-001, FR-QUALITY-002)."""
import numpy as np
import pytest

from app.quality.metrics import compute_all_metrics
from app.quality.scoring import QualityResult, compute_quality_score
from app.quality.analyzer import analyze_frame
from app.mcap_io.message_types import FrameRecord


def _make_frame(img: np.ndarray, **kwargs) -> FrameRecord:
    h, w = img.shape[:2]
    defaults = dict(
        mcap_file="test.mcap", topic="/cam", frame_seq=0,
        log_time_ns=1_000_000_000, publish_time_ns=1_000_000_000,
        ros_stamp_ns=1_000_000_000, width=w, height=h,
        encoding="bgr8", frame_id="test", image=img, decode_ms=1.0,
    )
    defaults.update(kwargs)
    return FrameRecord(**defaults)


class TestMetrics:
    def test_normal_image(self):
        img = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        assert 50 < m["brightness_mean"] < 200
        assert m["blur_score"] > 0
        assert m["width"] == 640
        assert m["height"] == 480

    def test_dark_image(self):
        img = np.full((480, 640, 3), 10, dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        assert m["brightness_mean"] < 30

    def test_bright_image(self):
        img = np.full((480, 640, 3), 245, dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        assert m["brightness_mean"] > 225

    def test_solid_color(self):
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        assert m["is_solid_color"] is True


class TestScoring:
    def test_normal_score(self):
        img = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        result = compute_quality_score(m)
        assert 0.0 <= result.quality_score <= 1.0
        assert isinstance(result.quality_tags, list)

    def test_corrupted_score_zero(self):
        m = {"width": 640, "height": 480, "brightness_mean": 0, "brightness_std": 0,
             "blur_score": 0, "contrast_score": 0, "saturation_mean": 0,
             "is_solid_color": False, "is_color_channel_anomaly": False,
             "is_aspect_ratio_anomaly": False}
        result = compute_quality_score(m, is_corrupted=True)
        assert result.quality_score == 0.0
        assert "corrupted" in result.quality_tags

    def test_dark_image_penalty(self):
        img = np.full((480, 640, 3), 10, dtype=np.uint8)
        m = compute_all_metrics(img, 640, 480)
        result = compute_quality_score(m)
        assert result.is_too_dark is True
        assert result.penalties.get("exposure", 0) > 0


class TestAnalyzer:
    def test_analyze_good_frame(self):
        img = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        frame = _make_frame(img)
        result = analyze_frame(frame, quality_threshold=0.6)
        assert isinstance(result, QualityResult)
        assert result.quality_score > 0

    def test_analyze_empty_image(self):
        img = np.zeros((0, 0, 3), dtype=np.uint8)
        frame = _make_frame(img, width=0, height=0)
        result = analyze_frame(frame)
        assert result.is_corrupted is True
        assert result.quality_score == 0.0
