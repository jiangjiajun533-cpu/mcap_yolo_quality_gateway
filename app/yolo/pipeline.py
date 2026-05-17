"""
End-to-end MCAP → sample → decode → quality → gate → YOLO pipeline.

See SAMPLING_STRATEGY.md for the full design rationale.

Sampling strategy (FR-MCAP-004)
-------------------------------
Execution: unified n-based sampling → ``raw_frame_index % n == 0``.

n derivation priority:
  target_fps > 0 AND source_fps > 0  →  n = round(source_fps / target_fps)  (bonus)
  target_fps > 0 AND source_fps N/A  →  WARNING + fallback to sample_every_n
  target_fps = 0                     →  n = sample_every_n  (default 1 = all frames)

Both params supplied:  target_fps wins; sample_every_n is fallback.

Skipped frames (index % n != 0): only counted, never decoded/checked.
Sampled frames: decode → quality → gate → YOLO.
All quality statistics are reported over sampled frames only.
max_frames limits sampled frames, not raw frames.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

import os
import sys

from app.core.config import settings
from app.core.errors import DecodeError, InferenceError, UnsupportedEncodingError
from app.core.logging import get_logger
from app.mcap_io.message_types import COMPRESSED_IMAGE_SCHEMAS, RAW_IMAGE_SCHEMAS
from app.mcap_io.reader import iter_decoded_messages, read_mcap_summary
from app.mcap_io.topic_scanner import is_depth_image_topic
from app.mcap_io.ros_image_decoder import decode_ros_message
from app.quality.analyzer import analyze_frame
from app.quality.scoring import QualityResult
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.postprocess import Detection

logger = get_logger("yolo.pipeline")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def compute_sample_n(
    target_fps: float,
    source_fps_estimate: float,
    sample_every_n: int,
) -> int:
    """
    Derive the final N for every-N-frame sampling.

    Priority:
      target_fps > 0 AND source_fps > 0  →  n = round(source / target)   (bonus)
      target_fps > 0 AND source_fps <= 0  →  fallback to sample_every_n  (+ WARNING)
      target_fps = 0                      →  n = sample_every_n          (required)
    """
    if target_fps > 0:
        if source_fps_estimate <= 0:
            logger.warning(
                f"target_fps={target_fps} requested but source_fps is unavailable "
                f"(duration unknown or topic not in summary). "
                f"Falling back to sample_every_n={sample_every_n}."
            )
            return max(1, sample_every_n)
        n = max(1, round(source_fps_estimate / target_fps))
        if n == 1 and source_fps_estimate < target_fps:
            logger.warning(
                f"source_fps≈{source_fps_estimate:.1f} < target_fps={target_fps}, "
                f"cannot downsample. Processing all frames (n=1)."
            )
        else:
            logger.info(
                f"target_fps={target_fps} source_fps≈{source_fps_estimate:.1f} → n={n}"
            )
        return n
    return max(1, sample_every_n)


# ---------------------------------------------------------------------------
# Pipeline stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    """
    Counts for each layer of the pipeline.
    Quality stats are scoped to sampled frames only.
    """
    # Raw level
    total_raw_frames: int = 0
    skipped_by_sampling: int = 0

    # Sampled level
    sampled_frames: int = 0

    # Decode level (within sampled)
    decode_failed: int = 0

    # Quality level (within successfully decoded)
    quality_analyzed: int = 0
    quality_passed: int = 0
    quality_failed: int = 0

    # YOLO level
    infer_success: int = 0
    infer_failed: int = 0
    skipped_low_quality: int = 0
    skipped_depth_topic: int = 0

    # Sampling metadata (populated by run_pipeline)
    sampling_mode: str = "sample_every_n"
    target_fps: float = 0.0
    estimated_source_fps: float = 0.0
    computed_sample_every_n: int = 1
    estimated_actual_fps: float = 0.0
    max_frames: int = 0

    # FR-MCAP-003: user clip window (relative seconds from MCAP start; 0 = not set)
    clip_start_sec: float = 0.0
    clip_end_sec: float = 0.0
    mcap_metadata_duration_sec: float = 0.0
    # Actual log-time span of sampled frames processed in this run
    processed_start_time_ns: Optional[int] = None
    processed_end_time_ns: Optional[int] = None

    def to_dict(self) -> dict:
        proc_dur = None
        if (
            self.processed_start_time_ns is not None
            and self.processed_end_time_ns is not None
            and self.processed_end_time_ns > self.processed_start_time_ns
        ):
            proc_dur = round(
                (self.processed_end_time_ns - self.processed_start_time_ns) / 1e9, 3
            )
        meta_warn = None
        if self.mcap_metadata_duration_sec <= 0:
            meta_warn = (
                "MCAP summary duration_sec is 0 (invalid or missing time span); "
                "source FPS cannot be estimated; --target-fps falls back to --sample-every-n"
            )
        return {
            "processing_time_range": {
                "clip_start_sec": self.clip_start_sec,
                "clip_end_sec": self.clip_end_sec,
                "clip_end_unlimited": self.clip_end_sec <= 0,
                "processed_start_time_ns": self.processed_start_time_ns,
                "processed_end_time_ns": self.processed_end_time_ns,
                "processed_duration_sec": proc_dur,
                "mcap_metadata_duration_sec": round(self.mcap_metadata_duration_sec, 3),
                "metadata_warning": meta_warn,
            },
            "sampling": {
                "mode": self.sampling_mode,
                "target_fps": self.target_fps,
                "estimated_source_fps": round(self.estimated_source_fps, 2),
                "computed_sample_every_n": self.computed_sample_every_n,
                "estimated_actual_fps": round(self.estimated_actual_fps, 2),
                "max_frames": self.max_frames,
                "max_frames_applies_to": "sampled_frames",
            },
            "frames": {
                "raw_frames": self.total_raw_frames,
                "sampled_frames": self.sampled_frames,
                "skipped_by_sampling": self.skipped_by_sampling,
                "decode_failed_frames": self.decode_failed,
                "quality_analyzed_frames": self.quality_analyzed,
                "quality_passed_frames": self.quality_passed,
                "quality_failed_frames": self.quality_failed,
                "skipped_low_quality_frames": self.skipped_low_quality,
                "skipped_depth_topic_frames": self.skipped_depth_topic,
                "infer_success_frames": self.infer_success,
                "infer_failed_frames": self.infer_failed,
            },
            "note": "All quality and detection statistics are based on sampled frames only.",
        }


# ---------------------------------------------------------------------------
# Inference record (FR-YOLO-006 + FR-YOLO-007)
# ---------------------------------------------------------------------------

@dataclass
class InferenceRecord:
    """Single-frame output record (FR-YOLO-006)."""
    mcap_file: str = ""
    topic: str = ""
    frame_seq: int = 0       # sequential index within sampled frames for this topic
    raw_frame_idx: int = 0   # original message index in the MCAP topic (before sampling)
    timestamp_ns: int = 0
    # FR-IMG-003
    log_time_ns: int = 0
    publish_time_ns: Optional[int] = None
    ros_stamp_ns: Optional[int] = None
    timestamp_source: str = "log_time"

    quality_score: float = 0.0
    # Full per-frame quality (FR-QUALITY-001); used by CLI reports / aggregators
    quality: Optional[QualityResult] = field(default=None, repr=False)
    quality_tags: List[str] = field(default_factory=list)
    quality_penalties: dict = field(default_factory=dict)
    is_bad_quality: bool = False

    # "inferred" | "skip_inference" | "decode_error" | "infer_error" | "quality_only"
    action: str = "inferred"
    reason: str = ""

    objects: List[Detection] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

    # Retained image for sample export (only set when needed, cleared after export)
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(
        self,
        model_info: Optional[dict] = None,
        target_classes: Optional[List[str]] = None,
    ) -> dict:
        """Serialise to FR-YOLO-006 JSON format."""
        d: dict = {
            "mcap_file": self.mcap_file,
            "topic": self.topic,
            "frame_seq": self.frame_seq,
            "raw_frame_idx": self.raw_frame_idx,
            "timestamp_ns": self.timestamp_ns,
            "log_time_ns": self.log_time_ns,
            "ros_stamp_ns": self.ros_stamp_ns,
            "timestamp_source": self.timestamp_source,
            "quality_score": self.quality_score,
            "quality_tags": self.quality_tags,
            "action": self.action,
        }
        if self.action not in ("inferred",):
            d["reason"] = self.reason
        if model_info:
            d["model"] = model_info
        if target_classes is not None:
            d["target_classes"] = target_classes
        if self.publish_time_ns is not None:
            d["publish_time_ns"] = self.publish_time_ns
        d["objects"] = [det.to_dict() for det in self.objects]
        d["latency_ms"] = self.latency_ms
        return d


# ---------------------------------------------------------------------------
# Progress tracker (single-line, works in both TTY and pipe/non-TTY)
# ---------------------------------------------------------------------------

class _ProgressTracker:
    """
    Single-line progress on Unix/macOS TTY; on Windows PowerShell use sparse lines
    (\\r often does not overwrite, and per-frame refresh floods the console).
    """

    _REFRESH_SEC = 0.5
    _INTERVAL_PCT = 5  # print at 0%, 5%, 10%, ... 100%

    def __init__(self, total: int, desc: str = ""):
        self._total = max(1, total)
        self._desc = desc
        self._done = 0
        self._t0 = time.perf_counter()
        self._last_pct_shown = -1
        self._last_refresh = 0.0
        # Windows consoles rarely honor \\r in-place updates reliably
        self._sparse = os.name == "nt" or not sys.stderr.isatty()

    def update(self, postfix: str = "") -> None:
        self._done += 1
        pct = int(self._done * 100 / self._total)
        now = time.perf_counter()
        elapsed = now - self._t0
        fps = self._done / elapsed if elapsed > 0 else 0

        if self._sparse:
            step = self._INTERVAL_PCT
            if (
                pct // step > self._last_pct_shown // step
                or self._done >= self._total
            ):
                self._last_pct_shown = pct
                self._emit(pct, fps, postfix)
            return

        # Unix TTY: throttle refresh (not every frame)
        if (
            self._done < self._total
            and pct == self._last_pct_shown
            and (now - self._last_refresh) < self._REFRESH_SEC
        ):
            return

        self._last_pct_shown = pct
        self._last_refresh = now
        bar_len = 24
        filled = int(bar_len * self._done / self._total)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"{self._desc} |{bar}| {pct:3d}% "
            f"{self._done}/{self._total} {fps:.1f}fr/s {postfix}"
        )
        sys.stderr.write("\r" + msg.ljust(100))
        sys.stderr.flush()

    def _emit(self, pct: int, fps: float, postfix: str) -> None:
        print(
            f"[{self._desc}] {pct:3d}% ({self._done}/{self._total}) "
            f"{fps:.1f} fr/s | {postfix}",
            flush=True,
        )

    def close(self) -> None:
        elapsed = time.perf_counter() - self._t0
        if not self._sparse:
            sys.stderr.write("\n")
            sys.stderr.flush()
        rate = self._done / elapsed if elapsed > 0 else 0.0
        print(
            f"[{self._desc}] Done: {self._done} frames in {elapsed:.1f}s ({rate:.1f} fr/s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main pipeline iterator
# ---------------------------------------------------------------------------

def run_pipeline(
    mcap_path: str | Path,
    topics: Optional[List[str]],
    runner: Optional[YoloOnnxRunner],
    quality_threshold: float = 0.6,
    infer_low_quality: bool = False,
    target_fps: float = 0.0,
    sample_every_n: int = 1,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    max_frames: int = 0,
    skip_depth_topics_for_yolo: Optional[bool] = None,
    stats_out: Optional[List["PipelineStats"]] = None,
) -> Iterator[InferenceRecord]:
    """
    Core pipeline: yields one InferenceRecord per SAMPLED frame.

    Only sampled frames (raw_index % n == 0) are decoded and processed.
    Skipped frames are counted in stats but never decoded.

    Pass a list as ``stats_out`` to receive the final PipelineStats
    after the generator is exhausted.
    """
    mcap_path = Path(mcap_path)
    skip_depth_yolo = (
        settings.skip_depth_topics_for_yolo
        if skip_depth_topics_for_yolo is None
        else skip_depth_topics_for_yolo
    )
    stats = PipelineStats()
    stats.max_frames = max_frames
    stats.target_fps = target_fps

    # --- Warn if both target_fps and sample_every_n are explicitly set ---
    if target_fps > 0 and sample_every_n > 1:
        logger.warning(
            f"Both target_fps={target_fps} and sample_every_n={sample_every_n} "
            f"are provided. target_fps takes priority; sample_every_n is used "
            f"only as fallback if source_fps is unavailable."
        )

    # --- Resolve time range (FR-MCAP-003) ---
    summary = read_mcap_summary(mcap_path)
    abs_start_ns: Optional[int] = None
    abs_end_ns: Optional[int] = None
    if start_sec > 0:
        abs_start_ns = summary.start_time_ns + int(start_sec * 1e9)
    if end_sec > 0:
        abs_end_ns = summary.start_time_ns + int(end_sec * 1e9)
    if abs_start_ns is not None and abs_end_ns is not None and abs_start_ns >= abs_end_ns:
        raise ValueError(
            f"--start-sec ({start_sec}) must be less than --end-sec ({end_sec})"
        )

    stats.clip_start_sec = start_sec
    stats.clip_end_sec = end_sec
    stats.mcap_metadata_duration_sec = summary.duration_sec
    if summary.duration_sec <= 0:
        logger.warning(
            f"MCAP {mcap_path.name}: metadata duration_sec=0 "
            f"(start_ns={summary.start_time_ns}, end_ns={summary.end_time_ns}). "
            "File may have timestamp/metadata issues; --target-fps cannot derive N from FPS."
        )

    # --- Resolve topics ---
    if not topics:
        topics = [t.topic for t in summary.image_topics]
        logger.info(f"Auto-detected {len(topics)} image topics in {mcap_path.name}")

    # --- Compute n per topic from source FPS estimate ---
    topic_source_fps: dict[str, float] = {}
    summary_topic_map: dict[str, int] = {}
    for t_info in summary.image_topics:
        summary_topic_map[t_info.topic] = t_info.message_count
        if t_info.topic in topics and summary.duration_sec > 0:
            topic_source_fps[t_info.topic] = t_info.message_count / summary.duration_sec

    topic_n: dict[str, int] = {}
    for t in topics:
        msg_count = summary_topic_map.get(t, 0)
        if msg_count == 0:
            logger.warning(
                f"Topic {t} has 0 messages in MCAP summary, nothing to process."
            )

        src_fps = topic_source_fps.get(t, 0.0)
        n = compute_sample_n(target_fps, src_fps, sample_every_n)
        topic_n[t] = n
        logger.info(f"Topic {t}: source_fps≈{src_fps:.1f}  n={n}")

    # --- Populate sampling metadata on stats ---
    avg_src_fps = 0.0
    avg_n = 1
    if topic_n:
        avg_n = max(1, round(sum(topic_n.values()) / len(topic_n)))
        fps_vals = [v for v in topic_source_fps.values() if v > 0]
        avg_src_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0

    stats.sampling_mode = "target_fps" if target_fps > 0 else "sample_every_n"
    stats.estimated_source_fps = avg_src_fps
    stats.computed_sample_every_n = avg_n
    stats.estimated_actual_fps = avg_src_fps / avg_n if avg_src_fps > 0 else 0.0

    # Accumulate total raw frames from summary
    for t_info in summary.image_topics:
        if t_info.topic in topics:
            stats.total_raw_frames += t_info.message_count

    # Per-topic raw frame index (used for % n check) and sampled frame seq
    raw_idx: dict[str, int] = {t: 0 for t in topics}
    sampled_seq: dict[str, int] = {t: 0 for t in topics}
    last_ts_ns: dict[str, int] = {}
    total_sampled = 0

    # Estimate total sampled frames for progress bar
    expected_sampled = 0
    for t_info in summary.image_topics:
        if t_info.topic in topics:
            n = topic_n.get(t_info.topic, avg_n)
            expected_sampled += t_info.message_count // max(1, n)
    if max_frames > 0:
        expected_sampled = min(expected_sampled, max_frames)

    mode_label = "YOLO" if runner else "Quality"
    _progress = _ProgressTracker(expected_sampled, f"{mode_label} {mcap_path.name}")

    for schema, channel, message, ros_msg in iter_decoded_messages(
        mcap_path,
        topics=topics,
        start_time_ns=abs_start_ns,
        end_time_ns=abs_end_ns,
    ):
        topic = channel.topic
        schema_name = schema.name

        # Skip unknown schemas without crashing
        if schema_name not in COMPRESSED_IMAGE_SCHEMAS and schema_name not in RAW_IMAGE_SCHEMAS:
            logger.debug(f"Skipping non-image schema {schema_name} on {topic}")
            continue

        # Initialise per-topic state lazily (handles unexpected topics)
        if topic not in raw_idx:
            raw_idx[topic] = 0
            sampled_seq[topic] = 0
            src_fps = topic_source_fps.get(topic, 0.0)
            topic_n[topic] = compute_sample_n(target_fps, src_fps, sample_every_n)

        n = topic_n[topic]
        current_raw_idx = raw_idx[topic]
        raw_idx[topic] += 1

        # ── SAMPLING GATE ──────────────────────────────────────────────────
        # Frames NOT selected: only count, never decode, never quality-check.
        if current_raw_idx % n != 0:
            stats.skipped_by_sampling += 1
            continue
        # ───────────────────────────────────────────────────────────────────

        frame_seq = sampled_seq[topic]
        sampled_seq[topic] += 1
        stats.sampled_frames += 1
        total_sampled += 1
        log_time_ns = message.log_time
        if stats.processed_start_time_ns is None or log_time_ns < stats.processed_start_time_ns:
            stats.processed_start_time_ns = log_time_ns
        if stats.processed_end_time_ns is None or log_time_ns > stats.processed_end_time_ns:
            stats.processed_end_time_ns = log_time_ns

        record = InferenceRecord(
            mcap_file=mcap_path.name,
            topic=topic,
            frame_seq=frame_seq,
            raw_frame_idx=current_raw_idx,
        )

        # ── DECODE ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            frame = decode_ros_message(
                ros_msg=ros_msg,
                schema_name=schema_name,
                mcap_file=mcap_path.name,
                topic=topic,
                frame_seq=frame_seq,
                log_time_ns=log_time_ns,
                publish_time_ns=message.publish_time,
            )
            decode_ms = round((time.perf_counter() - t0) * 1000, 2)
        except (DecodeError, UnsupportedEncodingError) as exc:
            decode_ms = round((time.perf_counter() - t0) * 1000, 2)
            stats.decode_failed += 1
            logger.warning(f"Decode failed {topic} seq={frame_seq}: {exc}")
            record.log_time_ns = log_time_ns
            record.publish_time_ns = message.publish_time or None
            record.ros_stamp_ns = None
            record.timestamp_ns = log_time_ns
            record.timestamp_source = "log_time"
            record.action = "decode_error"
            record.reason = str(exc)
            record.latency_ms = _zero_latency(decode_ms)
            _progress.update(f"fail={stats.decode_failed}")
            yield record
            continue

        record.log_time_ns = frame.log_time_ns
        record.publish_time_ns = frame.publish_time_ns
        record.ros_stamp_ns = frame.ros_stamp_ns
        record.timestamp_ns = (
            frame.ros_stamp_ns or frame.publish_time_ns or log_time_ns
        )
        record.timestamp_source = frame.timestamp_source

        # Per-frame timestamp anomaly (reversed or >10s gap)
        ts = record.timestamp_ns
        if topic in last_ts_ns:
            delta = ts - last_ts_ns[topic]
            if delta < 0 or delta > 10_000_000_000:
                frame.is_timestamp_anomaly = True
        last_ts_ns[topic] = ts

        # ── QUALITY ANALYSIS ───────────────────────────────────────────────
        t1 = time.perf_counter()
        qr: QualityResult = analyze_frame(frame, quality_threshold)
        quality_ms = round((time.perf_counter() - t1) * 1000, 2)

        stats.quality_analyzed += 1
        record.quality = qr
        record.quality_score = qr.quality_score
        record.quality_tags = qr.quality_tags
        record.quality_penalties = qr.penalties
        record.is_bad_quality = qr.is_bad_quality

        if qr.is_bad_quality:
            stats.quality_failed += 1
        else:
            stats.quality_passed += 1

        # ── QUALITY GATE + YOLO ────────────────────────────────────────────
        if runner is None:
            # Quality-only mode (no model provided)
            record.action = "quality_only"
            record.latency_ms = _build_latency(decode_ms, quality_ms)
            if qr.is_bad_quality:
                record.image = frame.image
            _progress.update(f"ok={stats.quality_passed} bad={stats.quality_failed}")
            yield record

        elif qr.is_bad_quality and not infer_low_quality:
            # FR-YOLO-007: skip low-quality frame
            stats.skipped_low_quality += 1
            record.action = "skip_inference"
            record.reason = "quality score lower than threshold"
            record.objects = []
            record.latency_ms = _build_latency(decode_ms, quality_ms)
            record.image = frame.image
            _progress.update(f"ok={stats.quality_passed} bad={stats.quality_failed} skip={stats.skipped_low_quality}")
            yield record

        elif skip_depth_yolo and is_depth_image_topic(topic):
            stats.skipped_depth_topic += 1
            record.action = "skip_inference"
            record.reason = "depth/disparity topic excluded from YOLO"
            record.objects = []
            record.latency_ms = _build_latency(decode_ms, quality_ms)
            _progress.update(
                f"ok={stats.quality_passed} bad={stats.quality_failed} "
                f"depth_skip={stats.skipped_depth_topic}"
            )
            yield record

        else:
            # Run YOLO inference
            t2 = time.perf_counter()
            try:
                detections, yolo_lat = runner.infer(frame.image)
                stats.infer_success += 1
                record.action = "inferred"
                record.objects = detections
                if detections or qr.is_bad_quality:
                    record.image = frame.image
                record.latency_ms = {
                    "decode":      decode_ms,
                    "quality":     quality_ms,
                    "preprocess":  yolo_lat["preprocess_ms"],
                    "inference":   yolo_lat["inference_ms"],
                    "postprocess": yolo_lat["postprocess_ms"],
                    "total": round(
                        decode_ms + quality_ms
                        + yolo_lat["preprocess_ms"]
                        + yolo_lat["inference_ms"]
                        + yolo_lat["postprocess_ms"],
                        2,
                    ),
                }
            except InferenceError as exc:
                stats.infer_failed += 1
                logger.warning(f"Inference failed {topic} seq={frame_seq}: {exc}")
                record.action = "infer_error"
                record.reason = str(exc)
                record.latency_ms = _build_latency(
                    decode_ms, quality_ms,
                    yolo_ms=round((time.perf_counter() - t2) * 1000, 2),
                )
            _progress.update(
                f"ok={stats.quality_passed} bad={stats.quality_failed} "
                f"det={stats.infer_success} obj={len(record.objects)}"
            )
            yield record

        # ── MAX FRAMES LIMIT (NFR-002) ─────────────────────────────────────
        if max_frames > 0 and total_sampled >= max_frames:
            logger.info(f"Reached max_frames={max_frames}, stopping.")
            break

    _progress.close()

    logger.info(
        f"[pipeline] {mcap_path.name} | "
        f"mode={stats.sampling_mode} n={stats.computed_sample_every_n} | "
        f"raw={stats.total_raw_frames} "
        f"skipped_sampling={stats.skipped_by_sampling} "
        f"sampled={stats.sampled_frames} "
        f"decode_fail={stats.decode_failed} "
        f"q_pass={stats.quality_passed} q_fail={stats.quality_failed} "
        f"inferred={stats.infer_success} "
        f"infer_fail={stats.infer_failed}"
    )
    if stats_out is not None:
        stats_out.append(stats)


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def _zero_latency(decode_ms: float = 0.0) -> dict:
    return {
        "decode": decode_ms, "quality": 0.0,
        "preprocess": 0.0, "inference": 0.0,
        "postprocess": 0.0, "total": decode_ms,
    }


def _build_latency(
    decode_ms: float,
    quality_ms: float,
    preprocess_ms: float = 0.0,
    inference_ms: float = 0.0,
    postprocess_ms: float = 0.0,
    yolo_ms: float = 0.0,
) -> dict:
    pre = preprocess_ms or yolo_ms
    return {
        "decode":      decode_ms,
        "quality":     quality_ms,
        "preprocess":  pre,
        "inference":   inference_ms,
        "postprocess": postprocess_ms,
        "total":       round(decode_ms + quality_ms + pre + inference_ms + postprocess_ms, 2),
    }
