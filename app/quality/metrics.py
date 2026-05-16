"""
Single-frame image quality metrics (FR-QUALITY-001).
All functions take a numpy array (BGR or grayscale) and return scalar values.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR or already-gray image to single-channel uint8."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert BGR to HSV. Returns grayscale unchanged."""
    if img.ndim == 2:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def brightness_stats(img: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) brightness from grayscale version."""
    gray = to_gray(img)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return mean, std


def blur_score(img: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    Common threshold: < 100 is blurry (configurable).
    """
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def contrast_score(img: np.ndarray) -> float:
    """
    RMS contrast = std dev of grayscale pixel values.
    Lower = less contrast.
    """
    gray = to_gray(img)
    return float(np.std(gray.astype(np.float32)))


def saturation_mean(img: np.ndarray) -> float:
    """Mean HSV saturation (0–255). Returns 0.0 for grayscale images."""
    if img.ndim == 2:
        return 0.0
    hsv = to_hsv(img)
    if hsv is None:
        return 0.0
    return float(np.mean(hsv[:, :, 1]))


def is_solid_color(img: np.ndarray, std_threshold: float = 5.0) -> bool:
    """Detect near-uniform / solid-colour frames (all channels very low std)."""
    gray = to_gray(img)
    return float(np.std(gray)) < std_threshold


def color_channel_stats(img: np.ndarray) -> Tuple[float, float, float]:
    """
    Mean per channel (B, G, R). Returns (0, 0, 0) for grayscale.
    Used to detect channel anomalies (e.g., green-only frames).
    """
    if img.ndim == 2:
        return 0.0, 0.0, 0.0
    b = float(np.mean(img[:, :, 0]))
    g = float(np.mean(img[:, :, 1]))
    r = float(np.mean(img[:, :, 2]))
    return b, g, r


def is_color_channel_anomaly(img: np.ndarray, ratio_threshold: float = 3.0) -> bool:
    """
    Detect channel imbalance (e.g., one channel dominates by ratio_threshold×).
    Returns False for grayscale images.
    """
    if img.ndim == 2:
        return False
    b, g, r = color_channel_stats(img)
    vals = [b, g, r]
    mx, mn = max(vals), min(vals)
    if mn < 1.0:
        return mx > 30.0   # one strong channel, others near zero
    return mx / mn > ratio_threshold


def aspect_ratio_anomaly(width: int, height: int,
                          min_ratio: float = 0.1,
                          max_ratio: float = 10.0) -> bool:
    """Check if width/height ratio is outside expected range."""
    if height == 0:
        return True
    ratio = width / height
    return ratio < min_ratio or ratio > max_ratio


def compute_all_metrics(img: np.ndarray, width: int, height: int) -> dict:
    """
    Compute all quality metrics for a single frame.
    Returns dict matching FR-QUALITY-001 output schema.
    """
    bright_mean, bright_std = brightness_stats(img)
    b_score = blur_score(img)
    c_score = contrast_score(img)
    sat_mean = saturation_mean(img)
    solid = is_solid_color(img)
    chan_anomaly = is_color_channel_anomaly(img)
    asp_anomaly = aspect_ratio_anomaly(width, height)

    return {
        "width": width,
        "height": height,
        "brightness_mean": round(bright_mean, 2),
        "brightness_std": round(bright_std, 2),
        "blur_score": round(b_score, 2),
        "contrast_score": round(c_score, 2),
        "saturation_mean": round(sat_mean, 2),
        "is_solid_color": solid,
        "is_color_channel_anomaly": chan_anomaly,
        "is_aspect_ratio_anomaly": asp_anomaly,
    }
