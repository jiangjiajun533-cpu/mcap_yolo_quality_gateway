"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure root logger and return a named logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
    )
    logger = logging.getLogger(name or "mcap_yolo_gateway")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a child logger."""
    return logging.getLogger(f"mcap_yolo_gateway.{name}")
