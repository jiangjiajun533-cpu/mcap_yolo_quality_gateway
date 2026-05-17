#!/usr/bin/env python3
"""
MCAP Quality Scan CLI (FR-CLI-001).

Usage:
  python scripts/run_mcap_quality_scan.py \
    --mcap ./test_data/sample.mcap \
    --auto-detect-topics true \
    --sample-every-n 5 \
    --quality-threshold 0.6 \
    --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import get_logger
from app.mcap_io.reader import read_mcap_summary, scan_mcap_directory
from app.mcap_io.message_types import McapSummary
from app.quality.aggregator import TopicQualitySummary
from app.quality.sequence_analyzer import TopicSequenceTracker, SequenceSummary
from app.yolo.pipeline import PipelineStats, InferenceRecord, run_pipeline
from app.report.json_report import (
    write_mcap_summary,
    write_quality_report,
    write_metrics,
)
from app.report.html_report import write_quality_html
from app.report.markdown_report import write_quality_md
from app.report.sample_exporter import export_bad_samples
from app.quality.duplicate import DuplicateDetector, duplicate_groups_to_dict

logger = get_logger("cli.quality_scan")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MCAP image quality scan (FR-CLI-001)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mcap", type=str, help="Path to a single MCAP file")
    g.add_argument("--mcap-dir", type=str, help="Directory containing MCAP files")

    p.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topic names (overrides auto-detect)",
    )
    p.add_argument(
        "--auto-detect-topics",
        type=str,
        default="true",
        help="Auto-detect image topics (true/false)",
    )
    p.add_argument("--sample-every-n", type=int, default=1)
    p.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Target FPS for sampling (bonus, 0=disabled)",
    )
    p.add_argument("--quality-threshold", type=float, default=0.6)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--end-sec", type=float, default=0.0)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max sampled frames to process (0=unlimited)",
    )
    p.add_argument("--max-bad-samples", type=int, default=200)
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
    if args.auto_detect_topics.lower() in ("true", "1", "yes"):
        return None
    return None


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcap_paths = _resolve_mcap_paths(args)
    topics = _resolve_topics(args)

    all_summaries: list[McapSummary] = []
    all_records: list[InferenceRecord] = []
    all_topic_quality: dict[str, TopicQualitySummary] = {}
    all_seq_trackers: dict[str, TopicSequenceTracker] = {}
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
                runner=None,
                quality_threshold=args.quality_threshold,
                target_fps=args.target_fps,
                sample_every_n=args.sample_every_n,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                max_frames=args.max_frames,
                stats_out=stats_out,
            ):
                all_records.append(record)

                topic = record.topic
                if topic not in dup_detectors:
                    dup_detectors[topic] = DuplicateDetector()
                if record.image is not None:
                    dup_detectors[topic].update(
                        record.image, record.frame_seq, record.timestamp_ns
                    )

                if record.image is not None:
                    sample_images[id(record)] = record.image
                    record.image = None  # free memory after capturing

                if topic not in all_topic_quality:
                    t_info = next(
                        (t for t in summary.image_topics if t.topic == topic), None
                    )
                    tqs = TopicQualitySummary(
                        topic=topic,
                        message_type=t_info.message_type if t_info else "",
                        total_frames=t_info.message_count if t_info else 0,
                    )
                    all_topic_quality[topic] = tqs

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
                        from app.quality.scoring import QualityResult

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
    write_metrics(
        output_dir,
        combined_stats,
        all_records,
        wall_time_sec=wall_sec,
        batch_failures=batch_failures or None,
    )

    if sample_images:
        export_bad_samples(
            output_dir, all_records, sample_images, max_samples=args.max_bad_samples
        )

    if batch_failures:
        logger.warning(
            "Completed with %d failed MCAP file(s): %s",
            len(batch_failures),
            ", ".join(f["mcap_file"] for f in batch_failures),
        )
    logger.info(
        f"Done. {combined_stats.sampled_frames} frames processed in {wall_sec:.1f}s. "
        f"Reports written to {output_dir}/"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Partial outputs may exist.")
        sys.exit(130)
