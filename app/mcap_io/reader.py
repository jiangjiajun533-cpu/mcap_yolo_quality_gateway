"""MCAP file reader using mcap + mcap-ros1-support (FR-MCAP-001, FR-MCAP-002)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

from mcap.reader import make_reader

from app.core.errors import McapCorruptedError, McapFileNotFoundError, McapReadError
from app.core.logging import get_logger
from app.mcap_io.message_types import (
    IMAGE_SCHEMA_NAMES,
    McapSummary,
    TopicInfo,
)

logger = get_logger("mcap_io.reader")


def _build_topic_infos(summary) -> List[TopicInfo]:
    """Build TopicInfo list from mcap summary object."""
    topics: List[TopicInfo] = []
    for ch_id, ch in summary.channels.items():
        schema = summary.schemas.get(ch.schema_id)
        schema_name = schema.name if schema else ""
        msg_count = (
            summary.statistics.channel_message_counts.get(ch_id, 0)
            if summary.statistics
            else 0
        )
        is_image = schema_name in IMAGE_SCHEMA_NAMES
        topics.append(
            TopicInfo(
                topic=ch.topic,
                message_type=schema_name,
                message_count=msg_count,
                is_image_topic=is_image,
                schema_name=schema_name,
                message_encoding=ch.message_encoding,
            )
        )
    return sorted(topics, key=lambda t: t.topic)


def read_mcap_summary(mcap_path: str | Path) -> McapSummary:
    """
    Read MCAP metadata without decoding any messages (FR-MCAP-001).
    Returns McapSummary with topic list and timing info.
    """
    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise McapFileNotFoundError(f"MCAP file not found: {mcap_path}")

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

        if summary is None:
            raise McapReadError(f"MCAP file has no summary section: {mcap_path}")

        stats = summary.statistics
        start_ns = stats.message_start_time if stats else 0
        end_ns = stats.message_end_time if stats else 0
        duration_sec = (end_ns - start_ns) / 1e9 if end_ns > start_ns else 0.0
        topic_count = len(summary.channels)
        topics = _build_topic_infos(summary)

        return McapSummary(
            mcap_file=mcap_path.name,
            start_time_ns=start_ns,
            end_time_ns=end_ns,
            duration_sec=duration_sec,
            topic_count=topic_count,
            topics=topics,
        )
    except (McapFileNotFoundError, McapReadError):
        raise
    except Exception as exc:
        raise McapCorruptedError(f"Failed to read MCAP: {mcap_path}: {exc}") from exc


def scan_mcap_directory(directory: str | Path, recursive: bool = True) -> List[Path]:
    """
    Scan a directory for .mcap files (FR-MCAP-002).
    Returns sorted list of paths.
    """
    directory = Path(directory)
    if not directory.exists():
        raise McapFileNotFoundError(f"Directory not found: {directory}")

    pattern = "**/*.mcap" if recursive else "*.mcap"
    files = sorted(directory.glob(pattern))
    logger.info(f"Found {len(files)} MCAP file(s) in {directory}")
    return files


def iter_messages(
    mcap_path: str | Path,
    topics: Optional[List[str]] = None,
    start_time_ns: Optional[int] = None,
    end_time_ns: Optional[int] = None,
) -> Iterator[tuple]:
    """
    Iterate raw (schema, channel, message) tuples from an MCAP file.
    Applies optional topic filter and time range filter (FR-MCAP-003).

    Yields: (schema, channel, message)  — message.data is raw bytes.
    """
    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise McapFileNotFoundError(f"MCAP file not found: {mcap_path}")

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=topics):
                log_time = message.log_time
                if start_time_ns is not None and log_time < start_time_ns:
                    continue
                if end_time_ns is not None and log_time > end_time_ns:
                    continue
                yield schema, channel, message
    except (McapFileNotFoundError,):
        raise
    except Exception as exc:
        raise McapReadError(f"Error iterating messages in {mcap_path}: {exc}") from exc


def iter_decoded_messages(
    mcap_path: str | Path,
    topics: Optional[List[str]] = None,
    start_time_ns: Optional[int] = None,
    end_time_ns: Optional[int] = None,
):
    """
    Iterate decoded ROS messages using mcap-ros1-support.
    Yields: (schema, channel, message, ros_msg)
    Falls back gracefully if decoder not available.
    """
    mcap_path = Path(mcap_path)
    if not mcap_path.exists():
        raise McapFileNotFoundError(f"MCAP file not found: {mcap_path}")

    try:
        from mcap_ros1.decoder import DecoderFactory

        decoder_factories = [DecoderFactory()]
    except ImportError:
        logger.warning(
            "mcap-ros1-support not available; falling back to raw message iteration"
        )
        decoder_factories = []

    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(
                topics=topics
            ):
                log_time = message.log_time
                if start_time_ns is not None and log_time < start_time_ns:
                    continue
                if end_time_ns is not None and log_time > end_time_ns:
                    continue
                yield schema, channel, message, ros_msg
    except (McapFileNotFoundError,):
        raise
    except Exception as exc:
        raise McapReadError(
            f"Error iterating decoded messages in {mcap_path}: {exc}"
        ) from exc
