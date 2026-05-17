"""Custom exception hierarchy for the application."""

from __future__ import annotations


class AppError(Exception):
    """Base exception for all application errors."""


# --- MCAP ---
class McapError(AppError):
    """Base class for MCAP-related errors."""


class McapFileNotFoundError(McapError):
    """MCAP file does not exist."""


class McapReadError(McapError):
    """Failed to read or parse an MCAP file."""


class McapCorruptedError(McapReadError):
    """MCAP file is corrupted."""


class TopicNotFoundError(McapError):
    """Requested topic not found in MCAP."""


class UnsupportedMessageTypeError(McapError):
    """Message type is not supported (non-fatal, skip with warning)."""


# --- Image decoding ---
class DecodeError(AppError):
    """Base class for image decode errors (non-fatal per frame)."""


class CompressedImageDecodeError(DecodeError):
    """Failed to decode CompressedImage frame."""


class RawImageDecodeError(DecodeError):
    """Failed to decode raw Image frame."""


class UnsupportedEncodingError(DecodeError):
    """Image encoding not supported."""


class EmptyFrameError(DecodeError):
    """Frame data is empty."""


# --- Quality ---
class QualityError(AppError):
    """Base class for quality analysis errors."""


# --- YOLO ---
class YoloError(AppError):
    """Base class for YOLO inference errors."""


class ModelNotFoundError(YoloError):
    """YOLO model file not found."""


class ModelLoadError(YoloError):
    """Failed to load YOLO model."""


class InferenceError(YoloError):
    """Runtime inference failure."""


# --- Report ---
class ReportError(AppError):
    """Base class for report generation errors."""


# --- API ---
class JobNotFoundError(AppError):
    """Job ID not found."""


class InvalidParameterError(AppError):
    """Invalid API or CLI parameter."""
