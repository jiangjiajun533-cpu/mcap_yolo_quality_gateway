"""
Quality scoring: converts raw metrics into a 0.0–1.0 score with
explainable penalties (FR-QUALITY-002).

score = 1.0 - blur_penalty - exposure_penalty - contrast_penalty
              - resolution_penalty - corruption_penalty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.core.config import settings


@dataclass
class QualityResult:
    """Full quality assessment for one frame (FR-QUALITY-001 + FR-QUALITY-002)."""
    # --- Frame identity ---
    mcap_file: str = ""
    topic: str = ""
    frame_seq: int = 0
    timestamp_ns: int = 0

    # --- Dimensions ---
    width: int = 0
    height: int = 0

    # --- Raw metrics ---
    brightness_mean: float = 0.0
    brightness_std: float = 0.0
    blur_score: float = 0.0
    contrast_score: float = 0.0
    saturation_mean: float = 0.0

    # --- Boolean flags ---
    is_too_dark: bool = False
    is_too_bright: bool = False
    is_blurry: bool = False
    is_low_contrast: bool = False
    is_low_resolution: bool = False
    is_corrupted: bool = False
    is_solid_color: bool = False
    is_color_channel_anomaly: bool = False
    is_aspect_ratio_anomaly: bool = False
    is_depth_image: bool = False

    # --- Score ---
    quality_score: float = 1.0
    quality_tags: List[str] = field(default_factory=list)
    penalties: Dict[str, float] = field(default_factory=dict)

    # --- Pass / fail ---
    is_bad_quality: bool = False


def compute_quality_score(
    metrics: dict,
    is_corrupted: bool = False,
    is_depth: bool = False,
) -> QualityResult:
    """
    Compute quality_score from raw metrics dict (from metrics.compute_all_metrics).
    All thresholds come from settings (config.py) — fully configurable.
    """
    cfg = settings
    result = QualityResult()

    # Populate raw metrics
    result.width = metrics.get("width", 0)
    result.height = metrics.get("height", 0)
    result.brightness_mean = metrics.get("brightness_mean", 0.0)
    result.brightness_std = metrics.get("brightness_std", 0.0)
    result.blur_score = metrics.get("blur_score", 0.0)
    result.contrast_score = metrics.get("contrast_score", 0.0)
    result.saturation_mean = metrics.get("saturation_mean", 0.0)
    result.is_solid_color = metrics.get("is_solid_color", False)
    result.is_color_channel_anomaly = metrics.get("is_color_channel_anomaly", False)
    result.is_aspect_ratio_anomaly = metrics.get("is_aspect_ratio_anomaly", False)
    result.is_depth_image = is_depth

    penalties: Dict[str, float] = {}
    tags: List[str] = []

    # --- Corruption penalty (binary) ---
    if is_corrupted:
        result.is_corrupted = True
        penalties["corruption"] = cfg.corruption_penalty_max
        tags.append("corrupted")
        result.quality_score = 0.0
        result.quality_tags = tags
        result.penalties = penalties
        result.is_bad_quality = True
        return result

    # --- Blur penalty ---
    blur = result.blur_score
    if blur < cfg.blur_threshold:
        result.is_blurry = True
        tags.append("blurry")
        # linear penalty: 0 at threshold, max at 0
        ratio = max(0.0, 1.0 - blur / cfg.blur_threshold)
        penalties["blur"] = round(cfg.blur_penalty_max * ratio, 4)
    else:
        penalties["blur"] = 0.0

    # --- Exposure penalty ---
    bm = result.brightness_mean
    if bm < cfg.brightness_low:
        result.is_too_dark = True
        tags.append("too_dark")
        ratio = max(0.0, 1.0 - bm / cfg.brightness_low)
        penalties["exposure"] = round(cfg.exposure_penalty_max * ratio, 4)
    elif bm > cfg.brightness_high:
        result.is_too_bright = True
        tags.append("too_bright")
        ratio = max(0.0, (bm - cfg.brightness_high) / (255.0 - cfg.brightness_high + 1e-6))
        penalties["exposure"] = round(cfg.exposure_penalty_max * ratio, 4)
    else:
        penalties["exposure"] = 0.0

    # --- Contrast penalty ---
    cs = result.contrast_score
    if cs < cfg.contrast_threshold:
        result.is_low_contrast = True
        tags.append("low_contrast")
        ratio = max(0.0, 1.0 - cs / cfg.contrast_threshold)
        penalties["contrast"] = round(cfg.contrast_penalty_max * ratio, 4)
    else:
        penalties["contrast"] = 0.0

    # --- Resolution penalty ---
    if result.width < cfg.min_width or result.height < cfg.min_height:
        result.is_low_resolution = True
        tags.append("low_resolution")
        penalties["resolution"] = cfg.resolution_penalty_max
    else:
        penalties["resolution"] = 0.0

    # --- Solid color / channel anomaly (additional tags, small penalty) ---
    if result.is_solid_color and "too_dark" not in tags and "too_bright" not in tags:
        tags.append("solid_color")
        penalties["solid_color"] = 0.10
    else:
        penalties.setdefault("solid_color", 0.0)

    if result.is_color_channel_anomaly:
        tags.append("channel_anomaly")
        penalties["channel_anomaly"] = 0.05
    else:
        penalties.setdefault("channel_anomaly", 0.0)

    # --- Aspect ratio anomaly ---
    if result.is_aspect_ratio_anomaly:
        tags.append("aspect_ratio_anomaly")
        penalties["aspect_ratio"] = 0.10
    else:
        penalties["aspect_ratio"] = 0.0

    # --- Compressed payload too small (bonus) ---
    payload_size = metrics.get("compressed_payload_size", 0)
    if payload_size > 0 and payload_size < 500:
        tags.append("small_payload")
        penalties["small_payload"] = 0.15
    else:
        penalties["small_payload"] = 0.0

    # --- Timestamp anomaly (bonus, frame-level) ---
    if metrics.get("is_timestamp_anomaly", False):
        tags.append("timestamp_anomaly")
        penalties["timestamp"] = 0.05
    else:
        penalties["timestamp"] = 0.0

    penalties["corruption"] = 0.0

    # --- Final score ---
    total_penalty = sum(penalties.values())
    score = max(0.0, round(1.0 - total_penalty, 4))

    result.quality_score = score
    result.quality_tags = tags if tags else ["normal"]
    result.penalties = penalties
    result.is_bad_quality = score < cfg.quality_threshold

    return result
