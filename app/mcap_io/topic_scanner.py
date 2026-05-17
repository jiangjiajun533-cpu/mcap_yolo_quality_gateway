"""Auto-discover image topics from MCAP files (FR-MCAP-001, FR-IMG-001/002)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app.core.errors import McapFileNotFoundError
from app.core.logging import get_logger
from app.mcap_io.message_types import (
    COMPRESSED_IMAGE_SCHEMAS,
    IMAGE_SCHEMA_NAMES,
    RAW_IMAGE_SCHEMAS,
    McapSummary,
    TopicInfo,
)
from app.mcap_io.reader import read_mcap_summary

logger = get_logger("mcap_io.topic_scanner")


def scan_image_topics(mcap_path: str | Path) -> List[TopicInfo]:
    """
    Auto-detect all image topics in an MCAP file (FR-MCAP-001).
    Returns only topics with recognized image schema names.
    """
    summary = read_mcap_summary(mcap_path)
    image_topics = summary.image_topics
    logger.info(
        f"{summary.mcap_file}: found {len(image_topics)} image topic(s) "
        f"out of {summary.topic_count} total"
    )
    for t in image_topics:
        logger.debug(f"  {t.topic} | {t.message_type} | {t.message_count} msgs")
    return image_topics


def filter_topics(
    available: List[TopicInfo],
    requested: Optional[List[str]] = None,
) -> List[TopicInfo]:
    """
    If requested topic names are provided, filter to those.
    If a requested topic is not found, log a warning (do not crash).
    """
    if not requested:
        return available

    available_names = {t.topic: t for t in available}
    result: List[TopicInfo] = []
    for name in requested:
        if name in available_names:
            result.append(available_names[name])
        else:
            logger.warning(f"Requested topic not found in MCAP: {name}")
    return result


def is_compressed_image(topic: TopicInfo) -> bool:
    return topic.schema_name in COMPRESSED_IMAGE_SCHEMAS


def is_raw_image(topic: TopicInfo) -> bool:
    return topic.schema_name in RAW_IMAGE_SCHEMAS


def is_depth_image_topic(topic: str) -> bool:
    """True for aligned depth / disparity topics — poor fit for COCO RGB detectors."""
    name = topic.lower()
    return "depth" in name or "disparity" in name


def build_topic_summary(summary: McapSummary) -> Dict:
    """Build JSON-serialisable topic summary (FR-MCAP-001 output format)."""
    return {
        "mcap_file": summary.mcap_file,
        "start_time_ns": summary.start_time_ns,
        "end_time_ns": summary.end_time_ns,
        "duration_sec": round(summary.duration_sec, 3),
        "topic_count": summary.topic_count,
        "topics": [
            {
                "topic": t.topic,
                "message_type": t.message_type,
                "message_count": t.message_count,
                "is_image_topic": t.is_image_topic,
            }
            for t in summary.topics
        ],
        "detected_topics": [
            {
                "topic": t.topic,
                "message_type": t.message_type,
                "message_count": t.message_count,
                "is_image_topic": True,
            }
            for t in summary.image_topics
        ],
    }
