#!/usr/bin/env python3
"""
MCAP + YOLO Inference CLI (FR-CLI-002).

Usage:
  python scripts/run_mcap_yolo_inference.py \
    --mcap ./test_data/sample.mcap \
    --topics /camera/front/image/compressed \
    --model ./models/yolov8n.onnx \
    --labels ./models/coco_classes.txt \
    --target-classes person,car,truck,bus \
    --quality-threshold 0.6 \
    --conf-threshold 0.25 \
    --nms-threshold 0.45 \
    --sample-every-n 5 \
    --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.logging import get_logger
from app.mcap_io.reader import read_mcap_summary, scan_mcap_directory
from app.mcap_io.message_types import McapSummary
from app.quality.aggregator import TopicQualitySummary
from app.quality.scoring import QualityResult
from app.quality.sequence_analyzer import TopicSequenceTracker
from app.yolo.onnx_runner import YoloOnnxRunner
from app.yolo.pipeline import PipelineStats, InferenceRecord, run_pipeline
from app.yolo.target_analyzer import TargetAnalyzer
from app.report.json_report import (
    write_mcap_summary,
    write_quality_report,
    write_yolo_predictions,
    write_metrics,
)
from app.report.html_report import write_quality_html, write_yolo_html
from app.report.markdown_report import write_quality_md, write_yolo_md
from app.report.sample_exporter import export_bad_samples, export_detection_samples
from app.quality.duplicate import DuplicateDetector, duplicate_groups_to_dict

logger = get_logger("cli.yolo_inference")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCAP + YOLO inference pipeline (FR-CLI-002)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mcap", type=str, help="Path to a single MCAP file")
    g.add_argument("--mcap-dir", type=str, help="Directory containing MCAP files")

    p.add_argument(
        "--topics", type=str, default=None, help="Comma-separated topic names"
    )
    p.add_argument("--auto-detect-topics", type=str, default="true")

    p.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.onnx",
        help="Path to YOLO ONNX model",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="models/coco_classes.txt",
        help="Path to class labels file",
    )
    p.add_argument(
        "--target-classes",
        type=str,
        default=None,
        help="Comma-separated target classes (e.g. person,car,truck,bus)",
    )
    p.add_argument("--conf-threshold", type=float, default=settings.conf_threshold)
    p.add_argument("--nms-threshold", type=float, default=settings.nms_threshold)
    p.add_argument(
        "--min-box-side",
        type=int,
        default=settings.min_box_side_px,
        help="Drop detections with shorter side below this (pixels); 0=off",
    )
    p.add_argument(
        "--skip-depth-yolo",
        type=str,
        default="true",
        help="Skip YOLO on depth/disparity topics (true/false)",
    )
    p.add_argument(
        "--detection-sample-min-conf",
        type=float,
        default=settings.detection_sample_min_conf,
        help="Min max-object confidence to export detection_samples",
    )
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])

    p.add_argument("--quality-threshold", type=float, default=0.6)
    p.add_argument(
        "--infer-low-quality",
        type=str,
        default="false",
        help="Force YOLO on low-quality frames (true/false)",
    )
    p.add_argument("--sample-every-n", type=int, default=1)
    p.add_argument("--target-fps", type=float, default=0.0)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--end-sec", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--max-bad-samples", type=int, default=200)
    p.add_argument("--max-detection-samples", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="outputs")
    return p.parse_args()


def _resolve_mcap_paths(args: argparse.Namespace) -> list[Path]:
    if args.mcap:
        p = Path(args.mcap)
        if not p.exists():
            logger.error(f"MCAP file not found: {p}")
            sys.exit(1)
        return [p]
    return scan_mcap_directory(args.mcap_dir)


def _resolve_topics(args: argparse.Namespace) -> list[str] | None:
    if args.topics:
        return [t.strip() for t in args.topics.split(",") if t.strip()]
    return None


def _bool_arg(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcap_paths = _resolve_mcap_paths(args)
    topics = _resolve_topics(args)
    infer_low_quality = _bool_arg(args.infer_low_quality)
    skip_depth_yolo = _bool_arg(args.skip_depth_yolo)
    target_classes = (
        [c.strip() for c in args.target_classes.split(",") if c.strip()]
        if args.target_classes
        else None
    )

    logger.info(f"Loading YOLO model: {args.model}")
    runner = YoloOnnxRunner(
        model_path=args.model,
        labels_path=args.labels,
        target_classes=target_classes,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        input_size=args.input_size,
        device=args.device,
        min_box_side_px=args.min_box_side,
    )
    model_info = runner.model_info()

    all_summaries: list[McapSummary] = []
    all_records: list[InferenceRecord] = []
    all_topic_quality: dict[str, TopicQualitySummary] = {}
    all_seq_trackers: dict[str, TopicSequenceTracker] = {}
    target_analyzer = TargetAnalyzer()
    dup_detectors: dict[str, DuplicateDetector] = {}
    sample_images: dict[int, object] = {}
    combined_stats = PipelineStats()
    batch_failures: list[dict[str, str]] = []

    wall_t0 = time.perf_counter()

    for mcap_path in mcap_paths:
        try:
            logger.info(f"Processing {mcap_path.name} ...")
            summary = read_mcap_summary(mcap_path)
            all_summaries.append(summary)

            stats_out: list[PipelineStats] = []
            for record in run_pipeline(
                mcap_path=mcap_path,
                topics=topics,
                runner=runner,
                quality_threshold=args.quality_threshold,
                infer_low_quality=infer_low_quality,
                target_fps=args.target_fps,
                sample_every_n=args.sample_every_n,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                max_frames=args.max_frames,
                skip_depth_topics_for_yolo=skip_depth_yolo,
                stats_out=stats_out,
            ):
                all_records.append(record)
                target_analyzer.update(record)

                topic = record.topic
                if topic not in dup_detectors:
                    dup_detectors[topic] = DuplicateDetector()
                if record.image is not None:
                    dup_detectors[topic].update(
                        record.image, record.frame_seq, record.timestamp_ns
                    )

                if record.image is not None:
                    sample_images[id(record)] = record.image
                    record.image = None

                topic = record.topic
                if topic not in all_topic_quality:
                    t_info = next(
                        (t for t in summary.image_topics if t.topic == topic), None
                    )
                    all_topic_quality[topic] = TopicQualitySummary(
                        topic=topic,
                        message_type=t_info.message_type if t_info else "",
                        total_frames=t_info.message_count if t_info else 0,
                    )
                if topic not in all_seq_trackers:
                    all_seq_trackers[topic] = TopicSequenceTracker(topic=topic)

                tqs = all_topic_quality[topic]
                if record.action == "decode_error":
                    tqs.add_decode_failure()
                elif (
                    record.quality is not None
                    or record.quality_score > 0
                    or record.action
                    in (
                        "quality_only",
                        "skip_inference",
                        "inferred",
                    )
                ):
                    qr = record.quality
                    if qr is None:
                        qr = QualityResult(
                            mcap_file=record.mcap_file,
                            topic=record.topic,
                            frame_seq=record.frame_seq,
                            timestamp_ns=record.timestamp_ns,
                            log_time_ns=record.log_time_ns,
                            publish_time_ns=record.publish_time_ns,
                            ros_stamp_ns=record.ros_stamp_ns,
                            timestamp_source=record.timestamp_source,
                            quality_score=record.quality_score,
                            quality_tags=record.quality_tags,
                            penalties=record.quality_penalties,
                            is_bad_quality=record.is_bad_quality,
                        )
                    tqs.add(qr, decode_ms=record.latency_ms.get("decode", 0.0))
                    all_seq_trackers[topic].update(
                        timestamp_ns=record.timestamp_ns,
                        width=qr.width,
                        height=qr.height,
                    )

            if stats_out:
                s = stats_out[0]
                combined_stats.total_raw_frames += s.total_raw_frames
                combined_stats.skipped_by_sampling += s.skipped_by_sampling
                combined_stats.sampled_frames += s.sampled_frames
                combined_stats.decode_failed += s.decode_failed
                combined_stats.quality_analyzed += s.quality_analyzed
                combined_stats.quality_passed += s.quality_passed
                combined_stats.quality_failed += s.quality_failed
                combined_stats.infer_success += s.infer_success
                combined_stats.infer_failed += s.infer_failed
                combined_stats.skipped_low_quality += s.skipped_low_quality
                combined_stats.skipped_depth_topic += s.skipped_depth_topic
                combined_stats.sampling_mode = s.sampling_mode
                combined_stats.target_fps = s.target_fps
                combined_stats.estimated_source_fps = s.estimated_source_fps
                combined_stats.computed_sample_every_n = s.computed_sample_every_n
                combined_stats.estimated_actual_fps = s.estimated_actual_fps
                combined_stats.max_frames = s.max_frames
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.exception("Failed processing MCAP %s", mcap_path)
            batch_failures.append(
                {"mcap_file": str(mcap_path.resolve()), "error": str(exc)}
            )

    wall_sec = time.perf_counter() - wall_t0

    topic_summaries = list(all_topic_quality.values())
    seq_summaries = [t.finalize() for t in all_seq_trackers.values()]
    dup_results = {t: d.finalize() for t, d in dup_detectors.items()}

    from app.report.json_report import _aggregate_latencies

    perf = {
        "wall_time_sec": round(wall_sec, 3),
        "processed_frames_per_sec": (
            round(combined_stats.sampled_frames / wall_sec, 2) if wall_sec > 0 else 0.0
        ),
        **_aggregate_latencies([r for r in all_records if r.action == "inferred"]),
    }

    logger.info("Writing reports ...")
    write_mcap_summary(output_dir, all_summaries)
    write_quality_report(
        output_dir,
        topic_summaries,
        seq_summaries,
        combined_stats,
        dup_results,
        batch_failures=batch_failures or None,
    )
    write_quality_html(
        output_dir,
        topic_summaries,
        seq_summaries,
        combined_stats,
        dup_results=dup_results,
        batch_failures=batch_failures or None,
    )
    write_quality_md(
        output_dir,
        topic_summaries,
        seq_summaries,
        combined_stats,
        dup_results=dup_results,
    )
    write_yolo_predictions(
        output_dir,
        all_records,
        model_info=model_info,
        target_classes=(
            list(target_classes) if target_classes else runner.class_names[:10]
        ),
    )
    write_yolo_html(
        output_dir,
        pipeline_stats=combined_stats,
        target_analyzer=target_analyzer,
        model_info=model_info,
        perf=perf,
        records=all_records,
    )
    write_yolo_md(
        output_dir,
        pipeline_stats=combined_stats,
        target_analyzer=target_analyzer,
        model_info=model_info,
        perf=perf,
        records=all_records,
    )
    write_metrics(
        output_dir,
        combined_stats,
        all_records,
        target_analyzer=target_analyzer,
        wall_time_sec=wall_sec,
        batch_failures=batch_failures or None,
    )

    if sample_images:
        export_bad_samples(
            output_dir, all_records, sample_images, max_samples=args.max_bad_samples
        )
        export_detection_samples(
            output_dir,
            all_records,
            sample_images,
            max_samples=args.max_detection_samples,
            min_confidence=args.detection_sample_min_conf,
        )

    if batch_failures:
        logger.warning(
            "Completed with %d failed MCAP file(s): %s",
            len(batch_failures),
            ", ".join(f["mcap_file"] for f in batch_failures),
        )
    logger.info(
        f"Done. {combined_stats.sampled_frames} sampled, "
        f"{combined_stats.infer_success} inferred in {wall_sec:.1f}s. "
        f"Reports written to {output_dir}/"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Partial outputs may exist.")
        sys.exit(130)
