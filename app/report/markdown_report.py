"""
Markdown report generators (FR-REPORT-002, FR-REPORT-003).

Outputs:
  quality_report.md — text-based quality overview
  yolo_report.md    — text-based detection summary
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.core.logging import get_logger
from app.quality.aggregator import TopicQualitySummary
from app.quality.duplicate import DuplicateGroup
from app.quality.sequence_analyzer import SequenceSummary
from app.yolo.pipeline import InferenceRecord, PipelineStats
from app.yolo.target_analyzer import TargetAnalyzer

logger = get_logger("report.markdown")


# ── FR-REPORT-002: quality_report.md ──────────────────────────────────────

def write_quality_md(
    output_dir: Path,
    topic_summaries: List[TopicQualitySummary],
    sequence_summaries: Optional[List[SequenceSummary]] = None,
    pipeline_stats: Optional[PipelineStats] = None,
    dup_results: Optional[dict] = None,
) -> Path:
    lines: list[str] = ["# Quality Report\n"]

    if pipeline_stats:
        sd = pipeline_stats.to_dict()
        lines.append("## Sampling Overview\n")
        lines.append(f"| Key | Value |")
        lines.append(f"|-----|-------|")
        for k, v in sd["sampling"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("## Frame Counts\n")
        lines.append(f"| Counter | Value |")
        lines.append(f"|---------|-------|")
        for k, v in sd["frames"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    for ts in topic_summaries:
        d = ts.to_dict()
        lines.append(f"## Topic: `{ts.topic}`\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for k, v in d.items():
            if k == "quality_issue_counts":
                continue
            lines.append(f"| {k} | {v} |")
        lines.append("")

        issues = ts.quality_issue_counts
        if issues:
            lines.append("### Quality Issues\n")
            lines.append("| Issue | Count |")
            lines.append("|-------|-------|")
            for issue, cnt in sorted(issues.items(), key=lambda x: -x[1]):
                lines.append(f"| {issue} | {cnt} |")
            lines.append("")

        worst = ts.top_worst_frames
        if worst:
            lines.append("### Worst Frames (Top 20)\n")
            lines.append("| Seq | Score | Tags |")
            lines.append("|-----|-------|------|")
            for w in worst[:20]:
                tags = ", ".join(w.quality_tags)
                lines.append(f"| {w.frame_seq} | {w.quality_score:.4f} | {tags} |")
            lines.append("")

    if sequence_summaries:
        lines.append("## Sequence Analysis\n")
        for ss in sequence_summaries:
            lines.append(f"### `{ss.topic}`\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Duration (sec) | {ss.duration_sec:.2f} |")
            lines.append(f"| Total Frames | {ss.total_frames} |")
            lines.append(f"| Estimated FPS | {ss.estimated_fps} |")
            lines.append(f"| Avg Interval (ms) | {ss.frame_interval_ms_avg:.2f} |")
            lines.append(f"| P95 Interval (ms) | {ss.frame_interval_ms_p95:.2f} |")
            lines.append(f"| Timestamp Jumps | {ss.timestamp_jump_count} |")
            lines.append(f"| Long Gaps | {ss.long_gap_count} |")
            lines.append(f"| Resolution Changes | {ss.resolution_change_count} |")
            lines.append("")

    if dup_results:
        any_groups = False
        for topic, groups in dup_results.items():
            if groups:
                if not any_groups:
                    lines.append("## Duplicate Frame Groups\n")
                    lines.append("| Topic | Start | End | Duration (s) |")
                    lines.append("|-------|-------|-----|--------------|")
                    any_groups = True
                for g in groups:
                    lines.append(f"| {topic.split('/')[-1]} | {g.start_frame_seq} | {g.end_frame_seq} | {g.duration_sec:.2f} |")
        if any_groups:
            lines.append("")

    lines.append("\n---\n*All quality statistics are based on sampled frames only.*\n")

    out = output_dir / "quality_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out


# ── FR-REPORT-003: yolo_report.md ────────────────────────────────────────

def write_yolo_md(
    output_dir: Path,
    pipeline_stats: Optional[PipelineStats] = None,
    target_analyzer: Optional[TargetAnalyzer] = None,
    model_info: Optional[dict] = None,
    perf: Optional[dict] = None,
) -> Path:
    lines: list[str] = ["# YOLO Detection Report\n"]

    if model_info:
        lines.append("## Model\n")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        for k, v in model_info.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if pipeline_stats:
        sd = pipeline_stats.to_dict()
        lines.append("## Frame Counts\n")
        lines.append("| Counter | Value |")
        lines.append("|---------|-------|")
        for k, v in sd["frames"].items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if target_analyzer:
        ta_data = target_analyzer.finalize()["target_analysis"]
        if ta_data:
            lines.append("## Target Class Statistics\n")
            lines.append(
                "| Class | Detected | Avg Conf | Avg Quality | Normal | Low Quality |"
            )
            lines.append(
                "|-------|----------|----------|-------------|--------|-------------|"
            )
            for ta in ta_data:
                lines.append(
                    f"| {ta['label']} | {ta['detected_count']} | "
                    f"{ta['avg_confidence']:.4f} | {ta['avg_quality_score']:.4f} | "
                    f"{ta['normal_quality_frame_detected_count']} | "
                    f"{ta['low_quality_frame_detected_count']} |"
                )
            lines.append("")

    if perf:
        lines.append("## Performance\n")
        avg = perf.get("avg_latency_ms", {})
        p95 = perf.get("p95_latency_ms", {})
        if avg:
            lines.append("| Stage | Avg (ms) | P95 (ms) |")
            lines.append("|-------|----------|----------|")
            for k in ["decode", "quality", "preprocess", "inference", "postprocess", "total"]:
                a = avg.get(k, "-")
                p = p95.get(k, "-")
                lines.append(f"| {k} | {a} | {p} |")
            lines.append("")
        fps = perf.get("processed_frames_per_sec", 0)
        if fps:
            lines.append(f"**Throughput:** {fps} frames/sec\n")

    lines.append("\n---\n*All statistics are based on sampled frames only.*\n")

    out = output_dir / "yolo_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out
