"""Centralized configuration using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    workspace_dir: Path = Field(
        default=Path("/workspace"), description="Container workspace root"
    )
    output_dir: Path = Field(
        default=Path("outputs"), description="Default output directory"
    )
    models_dir: Path = Field(
        default=Path("models"), description="Directory for model files"
    )
    test_data_dir: Path = Field(
        default=Path("test_data"), description="Directory for test MCAP files"
    )

    # --- MCAP reading ---
    auto_detect_topics: bool = Field(default=True)
    sample_every_n: int = Field(default=1, ge=1)
    target_fps: float = Field(
        default=0.0, ge=0.0, description="0 means disabled, use sample_every_n instead"
    )
    start_sec: float = Field(default=0.0, ge=0.0)
    end_sec: float = Field(default=0.0, ge=0.0, description="0 means no limit")
    max_frames: int = Field(default=0, ge=0, description="0 means no limit")

    # --- Quality thresholds ---
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    blur_threshold: float = Field(
        default=100.0, description="Laplacian variance below this = blurry"
    )
    brightness_low: float = Field(
        default=30.0, description="Mean brightness below this = too dark"
    )
    brightness_high: float = Field(
        default=225.0, description="Mean brightness above this = too bright"
    )
    contrast_threshold: float = Field(
        default=20.0, description="Std dev below this = low contrast"
    )
    min_width: int = Field(default=64, description="Width below this = low resolution")
    min_height: int = Field(
        default=64, description="Height below this = low resolution"
    )

    # --- Quality penalty weights ---
    blur_penalty_max: float = Field(default=0.35)
    exposure_penalty_max: float = Field(default=0.25)
    contrast_penalty_max: float = Field(default=0.15)
    resolution_penalty_max: float = Field(default=0.15)
    corruption_penalty_max: float = Field(default=1.0)

    # --- YOLO ---
    model_path: Path = Field(default=Path("models/yolov8n.onnx"))
    labels_path: Path = Field(default=Path("models/coco_classes.txt"))
    yolo_input_size: int = Field(default=640)
    conf_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="YOLO detection conf filter (FR-YOLO-001 example default 0.25)",
    )
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_box_side_px: int = Field(
        default=32,
        ge=0,
        description="Drop detections whose shorter bbox side is below this (px); 0 disables",
    )
    skip_depth_topics_for_yolo: bool = Field(
        default=True,
        description="Do not run YOLO on topics whose name contains depth/disparity",
    )
    detection_sample_min_conf: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Only export detection_samples when max object conf >= this",
    )
    infer_low_quality: bool = Field(default=False)
    target_classes: List[str] = Field(
        default=[
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "traffic light",
            "stop sign",
            "dog",
            "cat",
        ]
    )

    # --- Output limits ---
    max_bad_samples: int = Field(default=200, ge=0)
    max_detection_samples: int = Field(default=200, ge=0)
    html_gallery_preview_limit: int = Field(
        default=40,
        ge=0,
        description="Max thumbnails embedded in HTML reports (full set remains in outputs/)",
    )

    # --- Sequence analysis ---
    frame_gap_threshold_ms: float = Field(
        default=200.0, description="Gap above this triggers FRAME_TIME_GAP warning"
    )
    timestamp_jump_threshold_ms: float = Field(default=500.0)

    # --- API ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")


settings = Settings()
