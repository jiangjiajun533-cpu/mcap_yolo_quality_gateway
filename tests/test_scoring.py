"""Tests for quality scoring module (FR-QUALITY-002)."""

import pytest

from app.quality.scoring import QualityResult, compute_quality_score


def _make_normal_metrics() -> dict:
    return {
        "width": 1280,
        "height": 720,
        "brightness_mean": 120.0,
        "brightness_std": 50.0,
        "blur_score": 200.0,
        "contrast_score": 50.0,
        "saturation_mean": 60.0,
        "is_solid_color": False,
        "is_color_channel_anomaly": False,
        "is_aspect_ratio_anomaly": False,
    }


class TestComputeQualityScore:
    def test_normal_image_high_score(self):
        result = compute_quality_score(_make_normal_metrics())
        assert result.quality_score > 0.8
        assert "normal" in result.quality_tags
        assert result.is_bad_quality is False

    def test_corrupted_yields_zero(self):
        result = compute_quality_score(_make_normal_metrics(), is_corrupted=True)
        assert result.quality_score == 0.0
        assert result.is_corrupted is True
        assert "corrupted" in result.quality_tags

    def test_blurry_image(self):
        m = _make_normal_metrics()
        m["blur_score"] = 20.0
        result = compute_quality_score(m)
        assert result.is_blurry is True
        assert result.penalties["blur"] > 0
        assert result.quality_score < 1.0

    def test_too_dark(self):
        m = _make_normal_metrics()
        m["brightness_mean"] = 10.0
        result = compute_quality_score(m)
        assert result.is_too_dark is True
        assert "too_dark" in result.quality_tags

    def test_too_bright(self):
        m = _make_normal_metrics()
        m["brightness_mean"] = 250.0
        result = compute_quality_score(m)
        assert result.is_too_bright is True
        assert "too_bright" in result.quality_tags

    def test_low_contrast(self):
        m = _make_normal_metrics()
        m["contrast_score"] = 5.0
        result = compute_quality_score(m)
        assert result.is_low_contrast is True
        assert result.penalties["contrast"] > 0

    def test_low_resolution(self):
        m = _make_normal_metrics()
        m["width"] = 32
        m["height"] = 32
        result = compute_quality_score(m)
        assert result.is_low_resolution is True
        assert result.penalties["resolution"] > 0

    def test_penalties_sum(self):
        m = _make_normal_metrics()
        m["blur_score"] = 10.0
        m["brightness_mean"] = 5.0
        result = compute_quality_score(m)
        total_penalty = sum(result.penalties.values())
        expected_score = max(0.0, round(1.0 - total_penalty, 4))
        assert abs(result.quality_score - expected_score) < 0.01

    def test_score_range(self):
        for bm in [0, 50, 128, 200, 255]:
            m = _make_normal_metrics()
            m["brightness_mean"] = float(bm)
            result = compute_quality_score(m)
            assert 0.0 <= result.quality_score <= 1.0
