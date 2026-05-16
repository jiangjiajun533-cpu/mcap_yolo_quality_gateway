"""
ROS image message decoder.
Supports CompressedImage (FR-IMG-001) and raw Image (FR-IMG-002).
Preserves timestamps (FR-IMG-003).
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app.core.errors import (
    CompressedImageDecodeError,
    EmptyFrameError,
    RawImageDecodeError,
    UnsupportedEncodingError,
)
from app.core.logging import get_logger
from app.mcap_io.message_types import (
    COMPRESSED_IMAGE_SCHEMAS,
    RAW_IMAGE_SCHEMAS,
    FrameRecord,
    TopicInfo,
)

logger = get_logger("mcap_io.ros_image_decoder")

# Raw image encodings → OpenCV conversion code (or None for special handling)
_ENCODING_TO_CONVERT = {
    "rgb8":   cv2.COLOR_RGB2BGR,
    "bgr8":   None,              # already BGR
    "mono8":  None,              # grayscale, no conversion needed
    "rgba8":  cv2.COLOR_RGBA2BGR,
    "bgra8":  cv2.COLOR_BGRA2BGR,
}

# Depth / special encodings (supported with grayscale fallback)
_DEPTH_ENCODINGS = {"16UC1", "32FC1"}

# Channels per encoding
_ENCODING_CHANNELS = {
    "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4,
    "mono8": 1, "yuv422": 2,
    "16UC1": 1, "32FC1": 1,
}

_ENCODING_DTYPE = {
    "rgb8": np.uint8, "bgr8": np.uint8, "rgba8": np.uint8, "bgra8": np.uint8,
    "mono8": np.uint8, "yuv422": np.uint8,
    "16UC1": np.uint16, "32FC1": np.float32,
}


def _ros_stamp_to_ns(stamp) -> Optional[int]:
    """Convert ROS header stamp to nanoseconds."""
    try:
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    except Exception:
        return None


def decode_compressed_image(
    ros_msg,
    mcap_file: str,
    topic: str,
    frame_seq: int,
    log_time_ns: int,
    publish_time_ns: Optional[int] = None,
) -> FrameRecord:
    """
    Decode sensor_msgs/CompressedImage to BGR numpy array (FR-IMG-001).
    Raises CompressedImageDecodeError on failure (non-fatal per frame).
    """
    t0 = time.perf_counter()

    try:
        raw_data = bytes(ros_msg.data)
    except Exception as exc:
        raise CompressedImageDecodeError(f"Cannot read data field: {exc}") from exc

    if not raw_data:
        raise EmptyFrameError(f"CompressedImage data is empty: topic={topic} seq={frame_seq}")

    # Decode JPEG / PNG bytes → numpy BGR
    buf = np.frombuffer(raw_data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # always returns BGR uint8

    if img is None:
        raise CompressedImageDecodeError(
            f"cv2.imdecode returned None: topic={topic} seq={frame_seq} "
            f"format={getattr(ros_msg, 'format', 'unknown')}"
        )

    decode_ms = (time.perf_counter() - t0) * 1000.0

    # Timestamps
    ros_stamp_ns: Optional[int] = None
    frame_id: Optional[str] = None
    try:
        ros_stamp_ns = _ros_stamp_to_ns(ros_msg.header.stamp)
        frame_id = ros_msg.header.frame_id
    except Exception:
        pass

    timestamp_source = "ros_header" if ros_stamp_ns else (
        "publish_time" if publish_time_ns else "log_time"
    )

    return FrameRecord(
        mcap_file=mcap_file,
        topic=topic,
        frame_seq=frame_seq,
        log_time_ns=log_time_ns,
        publish_time_ns=publish_time_ns,
        ros_stamp_ns=ros_stamp_ns,
        width=img.shape[1],
        height=img.shape[0],
        encoding=getattr(ros_msg, "format", "jpeg"),
        frame_id=frame_id,
        image=img,
        decode_ms=decode_ms,
        message_type="sensor_msgs/CompressedImage",
        is_depth=False,
        timestamp_source=timestamp_source,
        compressed_payload_size=len(raw_data),
    )


def decode_raw_image(
    ros_msg,
    mcap_file: str,
    topic: str,
    frame_seq: int,
    log_time_ns: int,
    publish_time_ns: Optional[int] = None,
) -> FrameRecord:
    """
    Decode sensor_msgs/Image to numpy array (FR-IMG-002).
    Supports: rgb8, bgr8, mono8, rgba8, bgra8, 16UC1, 32FC1.
    Raises RawImageDecodeError on failure.
    """
    t0 = time.perf_counter()

    encoding: str = ros_msg.encoding
    width: int = ros_msg.width
    height: int = ros_msg.height
    step: int = ros_msg.step

    raw_data = bytes(ros_msg.data)
    if not raw_data:
        raise EmptyFrameError(f"Image data is empty: topic={topic} seq={frame_seq}")

    # Determine dtype and channels
    dtype = _ENCODING_DTYPE.get(encoding)
    channels = _ENCODING_CHANNELS.get(encoding)

    if dtype is None or channels is None:
        raise UnsupportedEncodingError(
            f"Unsupported encoding '{encoding}' on topic={topic} seq={frame_seq}"
        )

    expected_len = step * height
    if len(raw_data) != expected_len:
        raise RawImageDecodeError(
            f"Data length mismatch: got {len(raw_data)} expected {expected_len} "
            f"(topic={topic} seq={frame_seq} encoding={encoding})"
        )

    # Reshape raw bytes to image array
    if channels == 1:
        img_raw = np.frombuffer(raw_data, dtype=dtype).reshape((height, step // dtype(0).itemsize))
        img_raw = img_raw[:, :width]  # strip padding if step > width * itemsize
    else:
        img_raw = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, step))
        img_raw = img_raw[:, : width * channels].reshape((height, width, channels))

    is_depth = encoding in _DEPTH_ENCODINGS

    if is_depth:
        # Convert 16UC1 depth to 8-bit grayscale for quality analysis (visual representation)
        if encoding == "16UC1":
            max_val = img_raw.max()
            if max_val > 0:
                img_bgr = (img_raw.astype(np.float32) / max_val * 255).astype(np.uint8)
            else:
                img_bgr = np.zeros((height, width), dtype=np.uint8)
            # Keep as grayscale (H, W) — quality analyzer will handle it
        elif encoding == "32FC1":
            finite = img_raw[np.isfinite(img_raw)]
            max_val = finite.max() if finite.size > 0 else 1.0
            img_bgr = (img_raw / (max_val + 1e-6) * 255).astype(np.uint8)
        else:
            img_bgr = img_raw.astype(np.uint8)
    elif encoding == "yuv422":
        yuyv = img_raw.astype(np.uint8)
        img_bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    else:
        convert_code = _ENCODING_TO_CONVERT.get(encoding)
        if convert_code is not None:
            img_bgr = cv2.cvtColor(img_raw.astype(np.uint8), convert_code)
        else:
            img_bgr = img_raw.astype(np.uint8)

    decode_ms = (time.perf_counter() - t0) * 1000.0

    # Timestamps
    ros_stamp_ns: Optional[int] = None
    frame_id: Optional[str] = None
    try:
        ros_stamp_ns = _ros_stamp_to_ns(ros_msg.header.stamp)
        frame_id = ros_msg.header.frame_id
    except Exception:
        pass

    timestamp_source = "ros_header" if ros_stamp_ns else (
        "publish_time" if publish_time_ns else "log_time"
    )

    return FrameRecord(
        mcap_file=mcap_file,
        topic=topic,
        frame_seq=frame_seq,
        log_time_ns=log_time_ns,
        publish_time_ns=publish_time_ns,
        ros_stamp_ns=ros_stamp_ns,
        width=width,
        height=height,
        encoding=encoding,
        frame_id=frame_id,
        image=img_bgr,
        decode_ms=decode_ms,
        message_type="sensor_msgs/Image",
        is_depth=is_depth,
        timestamp_source=timestamp_source,
    )


def decode_ros_message(
    ros_msg,
    schema_name: str,
    mcap_file: str,
    topic: str,
    frame_seq: int,
    log_time_ns: int,
    publish_time_ns: Optional[int] = None,
) -> FrameRecord:
    """
    Unified entry point: dispatches to the correct decoder based on schema name.
    """
    if schema_name in COMPRESSED_IMAGE_SCHEMAS:
        return decode_compressed_image(
            ros_msg, mcap_file, topic, frame_seq, log_time_ns, publish_time_ns
        )
    elif schema_name in RAW_IMAGE_SCHEMAS:
        return decode_raw_image(
            ros_msg, mcap_file, topic, frame_seq, log_time_ns, publish_time_ns
        )
    else:
        from app.core.errors import UnsupportedMessageTypeError
        raise UnsupportedMessageTypeError(
            f"Unknown image schema '{schema_name}' on topic={topic}"
        )
