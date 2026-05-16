"""
Background worker: runs quality scan or YOLO inference in a thread.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.jobs.manager import Job, job_manager
from app.mcap_io.reader import read_mcap_summary
from app.quality.aggregator import TopicQualitySummary
from app.quality.scoring import QualityResult
from app.quality.sequence_analyzer import TopicSequenceTracker
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.pipeline import InferenceRecord, PipelineStats, run_pipeline
from app.yolo.target_analyzer import TargetAnalyzer
from app.report.json_report import (
    write_mcap_summary,
    write_quality_report,
    write_yolo_predictions,
    write_metrics,
)
from app.report.html_report import write_quality_html, write_yolo_html
from app.report.markdown_report import write_quality_md, write_yolo_md

logger = get_logger("jobs.worker")


def _run_quality_scan(job: Job) -> None:
    """Execute a quality-only scan (no YOLO model)."""
    p = job.params
    mcap_path = Path(p["mcap_path"])
    output_dir = Path(p.get("output_dir", f"outputs/{job.job_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_mcap_summary(mcap_path)
    topics = p.get("topics") or None

    stats_out: list[PipelineStats] = []
    records: list[InferenceRecord] = []
    topic_quality: dict[str, TopicQualitySummary] = {}
    seq_trackers: dict[str, TopicSequenceTracker] = {}

    t0 = time.perf_counter()
    for record in run_pipeline(
        mcap_path=mcap_path,
        topics=topics,
        runner=None,
        quality_threshold=p.get("quality_threshold", 0.6),
        target_fps=p.get("target_fps", 0.0),
        sample_every_n=p.get("sample_every_n", 1),
        max_frames=p.get("max_frames", 0),
        stats_out=stats_out,
    ):
        records.append(record)
        _accumulate_quality(record, summary, topic_quality, seq_trackers)

    wall_sec = time.perf_counter() - t0
    stats = stats_out[0] if stats_out else PipelineStats()

    write_mcap_summary(output_dir, [summary])
    write_quality_report(
        output_dir,
        list(topic_quality.values()),
        [t.finalize() for t in seq_trackers.values()],
        stats,
    )
    write_quality_html(
        output_dir,
        list(topic_quality.values()),
        [t.finalize() for t in seq_trackers.values()],
        stats,
    )
    write_quality_md(
        output_dir,
        list(topic_quality.values()),
        [t.finalize() for t in seq_trackers.values()],
        stats,
    )
    write_metrics(output_dir, stats, records, wall_time_sec=wall_sec)

    job_manager.set_finished(
        job.job_id,
        result_path=str(output_dir / "quality_report.json"),
        report_path=str(output_dir / "quality_report.html"),
    )


def _run_yolo_infer(job: Job) -> None:
    """Execute YOLO inference pipeline."""
    p = job.params
    mcap_path = Path(p["mcap_path"])
    output_dir = Path(p.get("output_dir", f"outputs/{job.job_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = YoloOnnxRunner(
        model_path=p.get("model_path", "models/yolov8n.onnx"),
        labels_path=p.get("labels_path", "models/coco_classes.txt"),
        target_classes=p.get("target_classes"),
        conf_threshold=p.get("conf_threshold", settings.conf_threshold),
        nms_threshold=p.get("nms_threshold", settings.nms_threshold),
        min_box_side_px=p.get("min_box_side_px", settings.min_box_side_px),
    )
    model_info = runner.model_info()

    summary = read_mcap_summary(mcap_path)
    topics = p.get("topics") or None

    stats_out: list[PipelineStats] = []
    records: list[InferenceRecord] = []
    topic_quality: dict[str, TopicQualitySummary] = {}
    seq_trackers: dict[str, TopicSequenceTracker] = {}
    target_analyzer = TargetAnalyzer()

    t0 = time.perf_counter()
    for record in run_pipeline(
        mcap_path=mcap_path,
        topics=topics,
        runner=runner,
        quality_threshold=p.get("quality_threshold", 0.6),
        infer_low_quality=p.get("infer_low_quality", False),
        target_fps=p.get("target_fps", 0.0),
        sample_every_n=p.get("sample_every_n", 1),
        max_frames=p.get("max_frames", 0),
        skip_depth_topics_for_yolo=p.get(
            "skip_depth_topics_for_yolo", settings.skip_depth_topics_for_yolo
        ),
        stats_out=stats_out,
    ):
        records.append(record)
        target_analyzer.update(record)
        _accumulate_quality(record, summary, topic_quality, seq_trackers)

    wall_sec = time.perf_counter() - t0
    stats = stats_out[0] if stats_out else PipelineStats()

    from app.report.json_report import _aggregate_latencies
    perf = {
        "wall_time_sec": round(wall_sec, 3),
        "processed_frames_per_sec": (
            round(stats.sampled_frames / wall_sec, 2) if wall_sec > 0 else 0.0
        ),
        **_aggregate_latencies([r for r in records if r.action == "inferred"]),
    }

    write_mcap_summary(output_dir, [summary])
    write_quality_report(
        output_dir,
        list(topic_quality.values()),
        [t.finalize() for t in seq_trackers.values()],
        stats,
    )
    write_yolo_predictions(output_dir, records, model_info=model_info)
    write_yolo_html(output_dir, stats, target_analyzer, model_info, perf)
    write_yolo_md(output_dir, stats, target_analyzer, model_info, perf)
    write_metrics(output_dir, stats, records, target_analyzer, wall_sec)

    job_manager.set_finished(
        job.job_id,
        result_path=str(output_dir / "yolo_predictions.json"),
        report_path=str(output_dir / "yolo_report.html"),
    )


def _accumulate_quality(
    record: InferenceRecord,
    summary,
    topic_quality: dict,
    seq_trackers: dict,
) -> None:
    topic = record.topic
    if topic not in topic_quality:
        t_info = next((t for t in summary.image_topics if t.topic == topic), None)
        topic_quality[topic] = TopicQualitySummary(
            topic=topic,
            message_type=t_info.message_type if t_info else "",
            total_frames=t_info.message_count if t_info else 0,
        )
    if topic not in seq_trackers:
        seq_trackers[topic] = TopicSequenceTracker(topic=topic)

    tqs = topic_quality[topic]
    if record.action == "decode_error":
        tqs.add_decode_failure()
    else:
        qr = QualityResult(
            mcap_file=record.mcap_file,
            topic=record.topic,
            frame_seq=record.frame_seq,
            timestamp_ns=record.timestamp_ns,
            quality_score=record.quality_score,
            quality_tags=record.quality_tags,
            penalties=record.quality_penalties,
            is_bad_quality=record.is_bad_quality,
        )
        tqs.add(qr, decode_ms=record.latency_ms.get("decode", 0.0))
        seq_trackers[topic].update(
            timestamp_ns=record.timestamp_ns,
            width=qr.width, height=qr.height,
        )


def launch_worker(job: Job) -> None:
    """Start the job in a background daemon thread."""
    job_manager.set_running(job.job_id)

    def _target():
        try:
            if job.job_type == "quality_scan":
                _run_quality_scan(job)
            elif job.job_type == "yolo_infer":
                _run_yolo_infer(job)
            else:
                job_manager.set_failed(job.job_id, f"Unknown job type: {job.job_type}")
        except Exception as exc:
            logger.exception(f"Job {job.job_id} failed: {exc}")
            job_manager.set_failed(job.job_id, str(exc))

    t = threading.Thread(target=_target, name=f"worker-{job.job_id}", daemon=True)
    t.start()
