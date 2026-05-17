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
from typing import Dict

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
        ptr = sd.get("processing_time_range") or {}
        if ptr:
            lines.append("## Processing Time Range (FR-MCAP-003)\n")
            lines.append("| Key | Value |")
            lines.append("|-----|-------|")
            for k, v in ptr.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
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

    # Auto-generated conclusion
    total_proc = sum(ts.processed_frames for ts in topic_summaries)
    total_bad = sum(ts.bad_quality_frames for ts in topic_summaries)
    bad_r = total_bad / total_proc if total_proc else 0
    avg_scores = [ts.avg_quality_score for ts in topic_summaries if ts.avg_quality_score]
    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0
    all_issues: dict[str, int] = {}
    for ts in topic_summaries:
        for k, v in (ts.quality_issue_counts or {}).items():
            all_issues[k] = all_issues.get(k, 0) + v
    top3 = sorted(all_issues.items(), key=lambda x: -x[1])[:3]

    lines.append("## Overall Conclusion\n")
    if bad_r < 0.05:
        verdict = "excellent"
    elif bad_r < 0.15:
        verdict = "good"
    elif bad_r < 0.30:
        verdict = "moderate — review recommended"
    else:
        verdict = "poor — immediate attention needed"
    lines.append(f"Analyzed **{total_proc}** frames across **{len(topic_summaries)}** camera topics. "
                 f"Overall quality is **{verdict}** — {bad_r:.1%} bad frames (avg score {overall_avg:.3f}).")
    if top3:
        issue_str = ", ".join(f"{k} ({v})" for k, v in top3)
        lines.append(f"Top issues: {issue_str}.")
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
    records: Optional[List[InferenceRecord]] = None,
) -> Path:
    lines: list[str] = ["# YOLO Detection Report\n"]

    if model_info:
        lines.append("## Model\n")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        for k, v in model_info.items():
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            lines.append(f"| {k} | {v} |")
        if "device" not in model_info:
            lines.append("| device | CPU |")
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

    # Per-topic target distribution
    if records:
        topic_class_counts: Dict[str, Dict[str, int]] = {}
        all_classes: set = set()
        for r in records:
            if r.action == "inferred" and r.objects:
                t_short = r.topic.split("/")[-1] if "/" in r.topic else r.topic
                if t_short not in topic_class_counts:
                    topic_class_counts[t_short] = {}
                for o in r.objects:
                    lbl = o.label if hasattr(o, "label") else o.get("label", "?")
                    topic_class_counts[t_short][lbl] = topic_class_counts[t_short].get(lbl, 0) + 1
                    all_classes.add(lbl)
        if topic_class_counts:
            cls_list = sorted(all_classes)
            lines.append("## Per-topic Target Distribution\n")
            lines.append("| Topic | " + " | ".join(cls_list) + " |")
            lines.append("|-------" + "|------" * len(cls_list) + "|")
            for t_short, cc in sorted(topic_class_counts.items()):
                vals = " | ".join(str(cc.get(c, 0)) for c in cls_list)
                lines.append(f"| {t_short} | {vals} |")
            lines.append("")

    # Quality-confidence analysis
    if records:
        buckets: Dict[str, list] = {"0.0–0.6": [], "0.6–0.8": [], "0.8–1.0": []}
        for r in records:
            if r.action != "inferred":
                continue
            qs = r.quality_score
            if qs < 0.6:
                buckets["0.0–0.6"].append(r)
            elif qs < 0.8:
                buckets["0.6–0.8"].append(r)
            else:
                buckets["0.8–1.0"].append(r)
        lines.append("## Quality Score vs Detection Confidence\n")
        lines.append("| Quality Bucket | Frames | Avg Objects | Avg Confidence |")
        lines.append("|----------------|--------|-------------|----------------|")
        for bname, recs in buckets.items():
            if not recs:
                lines.append(f"| {bname} | 0 | — | — |")
                continue
            n_obj = [len(r.objects) for r in recs]
            confs = [o.confidence for r in recs for o in r.objects if hasattr(o, "confidence")]
            avg_obj = round(sum(n_obj) / len(n_obj), 2)
            avg_conf = round(sum(confs) / len(confs), 3) if confs else "—"
            lines.append(f"| {bname} | {len(recs)} | {avg_obj} | {avg_conf} |")
        lines.append("")

    # Auto conclusion
    lines.append("## Overall Conclusion\n")
    conclusion_parts = []
    if pipeline_stats:
        sd = pipeline_stats.to_dict()
        fr = sd.get("frames", {})
        sampled = fr.get("sampled_frames", 0)
        inferred = fr.get("infer_success_frames", 0)
        skipped_q = fr.get("skipped_low_quality_frames", 0)
        failed = fr.get("infer_failed_frames", 0)
        conclusion_parts.append(f"Processed **{sampled}** sampled frames; **{inferred}** successfully inferred.")
        if skipped_q:
            conclusion_parts.append(f"{skipped_q} frames skipped due to low quality gating.")
        if failed:
            conclusion_parts.append(f"**{failed}** frames failed inference.")
        else:
            conclusion_parts.append("All inferred frames completed successfully.")
    if target_analyzer:
        ta_data = target_analyzer.finalize().get("target_analysis", [])
        if ta_data:
            total_det = sum(t.get("detected_count", 0) for t in ta_data)
            top_cls = sorted(ta_data, key=lambda t: -t.get("detected_count", 0))[:3]
            cls_str = ", ".join(f"{t.get('label','')} ({t.get('detected_count',0)})" for t in top_cls)
            conclusion_parts.append(f"Total detections: {total_det}. Top classes: {cls_str}.")
    lines.append(" ".join(conclusion_parts) if conclusion_parts else "No inference data available.")
    lines.append("")

    lines.append("\n---\n*All statistics are based on sampled frames only.*\n")

    out = output_dir / "yolo_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {out}")
    return out
