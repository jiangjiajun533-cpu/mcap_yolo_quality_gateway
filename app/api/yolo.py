"""
Single-frame preview endpoints (FR-API-007, FR-API-008) — bonus.

GET /mcap/frame        — return a single decoded frame as JPEG
GET /mcap/frame_yolo   — return a single frame with YOLO detections drawn
GET /mcap/frame_info   — return frame metadata, quality, and detections as JSON
GET /mcap/topic_frames — return total frame count for a topic
GET /mcap/resolve_frame — map pipeline timestamp_ns to raw message index
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.errors import DecodeError, UnsupportedEncodingError
from app.mcap_io.reader import iter_decoded_messages
from app.mcap_io.message_types import COMPRESSED_IMAGE_SCHEMAS, RAW_IMAGE_SCHEMAS
from app.mcap_io.ros_image_decoder import decode_ros_message
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.postprocess import Detection
from app.yolo.visualizer import draw_detections

router = APIRouter(prefix="/mcap", tags=["preview"])

_runner_cache: dict[str, YoloOnnxRunner] = {}

# Project root for resolving relative MCAP paths regardless of uvicorn cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_TIMESTAMP_TOLERANCE_NS = 500_000_000  # 500 ms — ros_stamp vs log_time can differ


@dataclass
class _DecodedFrame:
    image: np.ndarray
    frame_seq: int
    timestamp_ns: int
    encoding: str = ""


def _is_image_schema(schema_name: str) -> bool:
    return schema_name in COMPRESSED_IMAGE_SCHEMAS or schema_name in RAW_IMAGE_SCHEMAS


def _resolve_mcap_path(mcap_path: str) -> Path:
    """Resolve MCAP path relative to cwd or project root."""
    raw = mcap_path.strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return p.resolve()
    candidates = [Path.cwd() / raw, _PROJECT_ROOT / raw, _PROJECT_ROOT / "test_data" / p.name]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return p


def _frame_timestamp_ns(frame, log_time_ns: int) -> int:
    """Match pipeline record.timestamp_ns: ros_stamp → publish → log_time."""
    return int(frame.ros_stamp_ns or frame.publish_time_ns or log_time_ns)


def _decode_at_index(
    p: Path,
    topic: str,
    frame_seq: int,
) -> _DecodedFrame:
    idx = 0
    for schema, channel, message, ros_msg in iter_decoded_messages(p, topics=[topic]):
        schema_name = schema.name
        if not _is_image_schema(schema_name):
            continue
        if idx == frame_seq:
            try:
                frame = decode_ros_message(
                    ros_msg=ros_msg,
                    schema_name=schema_name,
                    mcap_file=p.name,
                    topic=topic,
                    frame_seq=frame_seq,
                    log_time_ns=message.log_time,
                    publish_time_ns=message.publish_time,
                )
                ts = _frame_timestamp_ns(frame, message.log_time)
                enc = getattr(frame, "encoding", "") or schema_name
                return _DecodedFrame(
                    image=frame.image,
                    frame_seq=frame_seq,
                    timestamp_ns=ts,
                    encoding=enc,
                )
            except (DecodeError, UnsupportedEncodingError) as exc:
                raise HTTPException(status_code=422, detail=f"Decode failed: {exc}")
        idx += 1

    raise HTTPException(status_code=404, detail=f"Frame {frame_seq} not found on {topic}")


def _decode_by_timestamp(
    p: Path,
    topic: str,
    timestamp_ns: int,
) -> _DecodedFrame:
    best: Optional[_DecodedFrame] = None
    best_diff: Optional[int] = None
    idx = 0

    for schema, channel, message, ros_msg in iter_decoded_messages(p, topics=[topic]):
        schema_name = schema.name
        if not _is_image_schema(schema_name):
            continue

        log_time = message.log_time
        try:
            frame = decode_ros_message(
                ros_msg=ros_msg,
                schema_name=schema_name,
                mcap_file=p.name,
                topic=topic,
                frame_seq=idx,
                log_time_ns=log_time,
                publish_time_ns=message.publish_time,
            )
            ts = _frame_timestamp_ns(frame, log_time)
            diff = abs(ts - timestamp_ns)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                enc = getattr(frame, "encoding", "") or schema_name
                best = _DecodedFrame(
                    image=frame.image,
                    frame_seq=idx,
                    timestamp_ns=ts,
                    encoding=enc,
                )
        except (DecodeError, UnsupportedEncodingError):
            pass
        idx += 1

    if best is None or best_diff is None or best_diff > _TIMESTAMP_TOLERANCE_NS:
        raise HTTPException(
            status_code=404,
            detail=f"No frame within {_TIMESTAMP_TOLERANCE_NS}ns of timestamp {timestamp_ns} on {topic}",
        )
    return best


def _get_frame(
    mcap_path: str,
    topic: str,
    frame_seq: Optional[int] = None,
    timestamp_ns: Optional[int] = None,
    raw_frame_idx: Optional[int] = None,
) -> _DecodedFrame:
    p = _resolve_mcap_path(mcap_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"MCAP not found: {mcap_path} (tried {p})")

    # Prefer raw_frame_idx — exact match to pipeline sampling index
    if raw_frame_idx is not None and raw_frame_idx >= 0:
        try:
            return _decode_at_index(p, topic, raw_frame_idx)
        except HTTPException:
            pass

    if timestamp_ns is not None and timestamp_ns > 0:
        try:
            return _decode_by_timestamp(p, topic, timestamp_ns)
        except HTTPException:
            if frame_seq is not None and frame_seq >= 0:
                return _decode_at_index(p, topic, frame_seq)
            raise

    if frame_seq is None:
        frame_seq = 0
    return _decode_at_index(p, topic, frame_seq)


def _to_jpeg_stream(img: np.ndarray) -> StreamingResponse:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@router.get("/frame")
def get_frame(
    mcap_path: str = Query(...),
    topic: str = Query(...),
    frame_seq: int = Query(0, ge=0),
    timestamp_ns: Optional[int] = Query(None, ge=0),
    raw_frame_idx: Optional[int] = Query(None, ge=0),
):
    decoded = _get_frame(
        mcap_path, topic,
        frame_seq=frame_seq,
        timestamp_ns=timestamp_ns,
        raw_frame_idx=raw_frame_idx,
    )
    return _to_jpeg_stream(decoded.image)


class _BBoxIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class _ObjectIn(BaseModel):
    label: str
    class_id: int = 0
    confidence: float = 0.0
    bbox: _BBoxIn


class DrawFrameRequest(BaseModel):
    mcap_path: str
    topic: str
    timestamp_ns: int = Field(..., ge=1)
    raw_frame_idx: Optional[int] = Field(None, ge=0)
    frame_seq: Optional[int] = Field(None, ge=0)
    objects: list[_ObjectIn] = Field(default_factory=list)


def _objects_to_detections(objects: list[_ObjectIn]) -> list[Detection]:
    dets: list[Detection] = []
    for o in objects:
        bb = o.bbox
        dets.append(
            Detection(
                label=o.label,
                class_id=o.class_id,
                confidence=float(o.confidence),
                x1=int(bb.x1),
                y1=int(bb.y1),
                x2=int(bb.x2),
                y2=int(bb.y2),
            )
        )
    return dets


@router.post("/draw_frame")
def draw_frame(req: DrawFrameRequest):
    """Draw pipeline-stored bounding boxes on a decoded frame (no re-inference)."""
    decoded = _get_frame(
        req.mcap_path,
        req.topic,
        frame_seq=req.frame_seq,
        timestamp_ns=req.timestamp_ns,
        raw_frame_idx=req.raw_frame_idx,
    )
    dets = _objects_to_detections(req.objects)
    if not dets:
        return _to_jpeg_stream(decoded.image)
    annotated = draw_detections(decoded.image, dets)
    return _to_jpeg_stream(annotated)


@router.get("/frame_yolo")
def get_frame_yolo(
    mcap_path: str = Query(...),
    topic: str = Query(...),
    frame_seq: int = Query(0, ge=0),
    timestamp_ns: Optional[int] = Query(None, ge=0),
    raw_frame_idx: Optional[int] = Query(None, ge=0),
    model_path: str = Query("models/yolov8n.onnx"),
):
    decoded = _get_frame(
        mcap_path, topic,
        frame_seq=frame_seq,
        timestamp_ns=timestamp_ns,
        raw_frame_idx=raw_frame_idx,
    )
    img = decoded.image

    if model_path not in _runner_cache:
        mp = Path(model_path)
        if not mp.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {mp}")
        _runner_cache[model_path] = YoloOnnxRunner(model_path=mp)

    runner = _runner_cache[model_path]
    detections, _ = runner.infer(img)
    annotated = draw_detections(img, detections)
    return _to_jpeg_stream(annotated)


@router.get("/frame_info")
def get_frame_info(
    mcap_path: str = Query(...),
    topic: str = Query(...),
    frame_seq: int = Query(0, ge=0),
    timestamp_ns: Optional[int] = Query(None, ge=0),
    model_path: str = Query("models/yolov8n.onnx"),
    run_yolo: bool = Query(False),
):
    """Return frame metadata, quality metrics, and optionally YOLO detections as JSON."""
    decoded = _get_frame(mcap_path, topic, frame_seq=frame_seq, timestamp_ns=timestamp_ns)
    img = decoded.image
    h, w = img.shape[:2]

    from app.quality.metrics import compute_all_metrics
    from app.quality.scoring import compute_quality_score

    metrics = compute_all_metrics(img, w, h)
    quality = compute_quality_score(metrics)

    result = {
        "frame_seq": decoded.frame_seq,
        "pipeline_frame_seq": frame_seq if timestamp_ns else None,
        "topic": topic,
        "width": w,
        "height": h,
        "timestamp_ns": decoded.timestamp_ns,
        "timestamp": decoded.timestamp_ns / 1e9 if decoded.timestamp_ns else 0,
        "encoding": decoded.encoding,
        "quality_score": round(quality.quality_score, 4),
        "quality_tags": quality.quality_tags,
        "is_bad_quality": quality.is_bad_quality,
        "penalties": {k: round(v, 4) for k, v in quality.penalties.items()},
        "metrics": {
            "brightness_mean": round(quality.brightness_mean, 1),
            "brightness_std": round(quality.brightness_std, 1),
            "blur_score": round(quality.blur_score, 1),
            "contrast_score": round(quality.contrast_score, 1),
            "saturation_mean": round(quality.saturation_mean, 1),
            "is_solid_color": quality.is_solid_color,
            "is_color_channel_anomaly": quality.is_color_channel_anomaly,
        },
        "detections": [],
    }

    if run_yolo:
        if model_path not in _runner_cache:
            mp = Path(model_path)
            if not mp.exists():
                result["detections"] = []
                return result
            _runner_cache[model_path] = YoloOnnxRunner(model_path=mp)

        runner = _runner_cache[model_path]
        detections, latency = runner.infer(img)
        result["detections"] = [d.to_dict() for d in detections]
        result["inference_ms"] = round(latency * 1000, 2) if isinstance(latency, float) else 0

    return result


@router.get("/resolve_frame")
def resolve_frame(
    mcap_path: str = Query(...),
    topic: str = Query(...),
    timestamp_ns: int = Query(..., ge=1),
):
    """Map a pipeline timestamp to the raw message index on a topic (for timeline scrubbing)."""
    decoded = _decode_by_timestamp(_resolve_mcap_path(mcap_path), topic, timestamp_ns)
    return {
        "topic": topic,
        "timestamp_ns": decoded.timestamp_ns,
        "raw_frame_seq": decoded.frame_seq,
    }


@router.get("/topic_frames")
def get_topic_frames(
    mcap_path: str = Query(...),
    topic: str = Query(...),
):
    """Return total frame count for a topic (for timeline)."""
    p = _resolve_mcap_path(mcap_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"MCAP not found: {mcap_path}")

    count = 0
    for schema, channel, message, ros_msg in iter_decoded_messages(p, topics=[topic]):
        schema_name = schema.name
        if _is_image_schema(schema_name):
            count += 1

    return {"topic": topic, "total_frames": count}
