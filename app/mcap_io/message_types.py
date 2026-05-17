"""Data classes for frame records and MCAP metadata (FR-IMG-003)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FrameRecord:
    """Single decoded image frame with all metadata (FR-IMG-003)."""

    mcap_file: str
    topic: str
    frame_seq: int

    log_time_ns: int
    publish_time_ns: Optional[int]
    ros_stamp_ns: Optional[int]

    width: int
    height: int
    encoding: str
    frame_id: Optional[str]

    image: np.ndarray  # BGR uint8 (H, W, 3) or grayscale (H, W)
    decode_ms: float

    # Source info
    message_type: str = ""  # e.g. sensor_msgs/CompressedImage
    is_depth: bool = False  # True for 16UC1 depth images
    timestamp_source: str = "log_time"  # "ros_header" | "publish_time" | "log_time"
    compressed_payload_size: int = (
        0  # raw bytes of compressed data (bonus: small payload)
    )
    is_timestamp_anomaly: bool = (
        False  # bonus: set by pipeline if timestamp inconsistent
    )


@dataclass
class TopicInfo:
    """Metadata for a single MCAP topic."""

    topic: str
    message_type: str  # e.g. sensor_msgs/CompressedImage
    message_count: int
    is_image_topic: bool
    schema_name: str = ""
    message_encoding: str = ""  # "ros1" | "ros2" | "cdr"


@dataclass
class McapSummary:
    """MCAP file-level summary (FR-MCAP-001)."""

    mcap_file: str
    start_time_ns: int
    end_time_ns: int
    duration_sec: float
    topic_count: int
    topics: List[TopicInfo] = field(default_factory=list)

    @property
    def image_topics(self) -> List[TopicInfo]:
        return [t for t in self.topics if t.is_image_topic]


# Recognized image schema names (ros1 and ros2 variants)
IMAGE_SCHEMA_NAMES = {
    "sensor_msgs/CompressedImage",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/Image",
    "sensor_msgs/msg/Image",
}

COMPRESSED_IMAGE_SCHEMAS = {
    "sensor_msgs/CompressedImage",
    "sensor_msgs/msg/CompressedImage",
}

RAW_IMAGE_SCHEMAS = {
    "sensor_msgs/Image",
    "sensor_msgs/msg/Image",
}
