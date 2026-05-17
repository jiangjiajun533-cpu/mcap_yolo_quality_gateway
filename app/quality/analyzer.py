"""
Per-frame quality analyzer: integrates metrics + scoring + gate control
(FR-QUALITY-001, FR-QUALITY-002, FR-YOLO-007).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.errors import DecodeError, QualityError
from app.core.logging import get_logger
from app.mcap_io.message_types import FrameRecord
from app.quality.metrics import compute_all_metrics
from app.quality.scoring import QualityResult, compute_quality_score

logger = get_logger("quality.analyzer")


def analyze_frame(frame: FrameRecord, quality_threshold: Optional[float] = None) -> QualityResult:
    """
    Run quality analysis on a decoded FrameRecord.
    Returns a QualityResult with score, tags, penalties, and pass/fail flag.
    """
    t0 = time.perf_counter()
    threshold = quality_threshold if quality_threshold is not None else settings.quality_threshold

    img = frame.image
    is_corrupted = False

    # Guard: empty or None image
    if img is None or img.size == 0:
        result = QualityResult(
            mcap_file=frame.mcap_file,
            topic=frame.topic,
            frame_seq=frame.frame_seq,
            timestamp_ns=frame.ros_stamp_ns or frame.publish_time_ns or frame.log_time_ns,
            log_time_ns=frame.log_time_ns,
            publish_time_ns=frame.publish_time_ns,
            ros_stamp_ns=frame.ros_stamp_ns,
            timestamp_source=frame.timestamp_source,
            width=frame.width,
            height=frame.height,
            is_corrupted=True,
            quality_score=0.0,
            quality_tags=["corrupted"],
            penalties={"corruption": 1.0},
            is_bad_quality=True,
            is_depth_image=frame.is_depth,
        )
        return result

    try:
        metrics = compute_all_metrics(img, frame.width, frame.height)
    except Exception as exc:
        logger.warning(f"Metrics computation failed for {frame.topic} seq={frame.frame_seq}: {exc}")
        is_corrupted = True
        metrics = {
            "width": frame.width, "height": frame.height,
            "brightness_mean": 0.0, "brightness_std": 0.0,
            "blur_score": 0.0, "contrast_score": 0.0,
            "saturation_mean": 0.0,
            "is_solid_color": False, "is_color_channel_anomaly": False,
            "is_aspect_ratio_anomaly": False,
        }

    # Bonus: compressed payload size (FR-QUALITY-001 item 12)
    metrics["compressed_payload_size"] = getattr(frame, "compressed_payload_size", 0)
    # Bonus: timestamp anomaly (FR-QUALITY-001 item 13)
    metrics["is_timestamp_anomaly"] = getattr(frame, "is_timestamp_anomaly", False)

    result = compute_quality_score(
        metrics=metrics,
        is_corrupted=is_corrupted,
        is_depth=frame.is_depth,
    )

    # Override threshold from caller
    result.is_bad_quality = result.quality_score < threshold

    # Fill identity fields
    result.mcap_file = frame.mcap_file
    result.topic = frame.topic
    result.frame_seq = frame.frame_seq
    result.timestamp_ns = frame.ros_stamp_ns or frame.publish_time_ns or frame.log_time_ns
    result.log_time_ns = frame.log_time_ns
    result.publish_time_ns = frame.publish_time_ns
    result.ros_stamp_ns = frame.ros_stamp_ns
    result.timestamp_source = frame.timestamp_source

    quality_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        f"quality topic={frame.topic} seq={frame.frame_seq} "
        f"score={result.quality_score:.3f} tags={result.quality_tags} "
        f"quality_ms={quality_ms:.1f}"
    )
    return result


def quality_result_to_dict(result: QualityResult) -> dict:
    """Serialize QualityResult to JSON-compatible dict (FR-QUALITY-001 + FR-IMG-003)."""
    d = {
        "mcap_file": result.mcap_file,
        "topic": result.topic,
        "frame_seq": result.frame_seq,
        "timestamp_ns": result.timestamp_ns,
        "log_time_ns": result.log_time_ns,
        "ros_stamp_ns": result.ros_stamp_ns,
        "timestamp_source": result.timestamp_source,
        "width": result.width,
        "height": result.height,
        "brightness_mean": result.brightness_mean,
        "brightness_std": result.brightness_std,
        "blur_score": result.blur_score,
        "contrast_score": result.contrast_score,
        "saturation_mean": result.saturation_mean,
        "is_too_dark": result.is_too_dark,
        "is_too_bright": result.is_too_bright,
        "is_blurry": result.is_blurry,
        "is_low_contrast": result.is_low_contrast,
        "is_low_resolution": result.is_low_resolution,
        "is_corrupted": result.is_corrupted,
        "is_depth_image": result.is_depth_image,
        "quality_score": result.quality_score,
        "quality_tags": result.quality_tags,
        "penalties": result.penalties,
        "is_bad_quality": result.is_bad_quality,
    }
    if result.publish_time_ns is not None:
        d["publish_time_ns"] = result.publish_time_ns
    return d
