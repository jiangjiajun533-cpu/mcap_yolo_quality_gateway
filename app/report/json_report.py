"""
JSON report generators (FR-REPORT-001, FR-REPORT-002, FR-REPORT-003).

Outputs:
  mcap_summary.json     — MCAP file-level overview
  quality_report.json   — per-topic quality + sequence analysis
  yolo_predictions.json — per-frame inference records
  metrics.json          — pipeline-level sampling + perf stats
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.logging import get_logger
from app.mcap_io.message_types import McapSummary
from app.quality.aggregator import TopicQualitySummary
from app.quality.duplicate import DuplicateGroup, duplicate_groups_to_dict
from app.quality.sequence_analyzer import SequenceSummary, sequence_summary_to_dict
from app.yolo.pipeline import InferenceRecord, PipelineStats
from app.yolo.target_analyzer import TargetAnalyzer

logger = get_logger("report.json")


def _safe_json(obj: Any) -> Any:
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_safe_json)
    logger.info(f"Wrote {path} ({path.stat().st_size} bytes)")


# ── FR-REPORT-001: mcap_summary.json ──────────────────────────────────────

def write_mcap_summary(
    output_dir: Path,
    summaries: List[McapSummary],
) -> Path:
    """Generate ``mcap_summary.json`` from one or more MCAP summaries."""
    data = {
        "files": [
            {
                "mcap_file": s.mcap_file,
                "duration_sec": round(s.duration_sec, 3),
                "topic_count": s.topic_count,
                "image_topics": [t.topic for t in s.image_topics],
                "start_time_ns": s.start_time_ns,
                "end_time_ns": s.end_time_ns,
            }
            for s in summaries
        ]
    }
    out = output_dir / "mcap_summary.json"
    _write_json(out, data)
    return out


# ── FR-REPORT-002: quality_report.json ────────────────────────────────────

def write_quality_report(
    output_dir: Path,
    topic_summaries: List[TopicQualitySummary],
    sequence_summaries: Optional[List[SequenceSummary]] = None,
    pipeline_stats: Optional[PipelineStats] = None,
    duplicate_results: Optional[Dict[str, List[DuplicateGroup]]] = None,
) -> Path:
    """Generate ``quality_report.json``."""
    data: Dict[str, Any] = {
        "topics": [ts.to_dict() for ts in topic_summaries],
    }
    if sequence_summaries:
        data["sequence_analysis"] = [
            sequence_summary_to_dict(ss) for ss in sequence_summaries
        ]
    for ts in topic_summaries:
        worst = ts.top_worst_frames
        if worst:
            topic_key = ts.topic.replace("/", "_").strip("_")
            data.setdefault("worst_frames", {})[topic_key] = [
                {
                    "frame_seq": w.frame_seq,
                    "timestamp_ns": w.timestamp_ns,
                    "quality_score": w.quality_score,
                    "quality_tags": w.quality_tags,
                    "penalties": w.penalties,
                }
                for w in worst[:20]
            ]
    if pipeline_stats:
        data["pipeline_stats"] = pipeline_stats.to_dict()

    if duplicate_results:
        data["duplicate_analysis"] = [
            duplicate_groups_to_dict(topic, groups)
            for topic, groups in duplicate_results.items()
            if groups
        ]

    out = output_dir / "quality_report.json"
    _write_json(out, data)
    return out


# ── FR-REPORT-003: yolo_predictions.json ──────────────────────────────────

def write_yolo_predictions(
    output_dir: Path,
    records: List[InferenceRecord],
    model_info: Optional[dict] = None,
    target_classes: Optional[List[str]] = None,
) -> Path:
    """Generate ``yolo_predictions.json`` — one entry per sampled frame."""
    data = {
        "predictions": [
            r.to_dict(model_info=model_info, target_classes=target_classes)
            for r in records
        ]
    }
    out = output_dir / "yolo_predictions.json"
    _write_json(out, data)
    return out


# ── metrics.json (FR-REPORT-003 + NFR-002 perf) ──────────────────────────

def write_metrics(
    output_dir: Path,
    pipeline_stats: PipelineStats,
    records: List[InferenceRecord],
    target_analyzer: Optional[TargetAnalyzer] = None,
    wall_time_sec: float = 0.0,
) -> Path:
    """Generate ``metrics.json`` with sampling info, latencies, target analysis."""
    inferred = [r for r in records if r.action == "inferred"]
    latencies = _aggregate_latencies(inferred)

    data: Dict[str, Any] = {
        **pipeline_stats.to_dict(),
        "performance": {
            "wall_time_sec": round(wall_time_sec, 3),
            "processed_frames_per_sec": (
                round(pipeline_stats.sampled_frames / wall_time_sec, 2)
                if wall_time_sec > 0 else 0.0
            ),
            **latencies,
        },
    }
    if target_analyzer:
        data["target_analysis"] = target_analyzer.finalize()["target_analysis"]

    out = output_dir / "metrics.json"
    _write_json(out, data)
    return out


def _aggregate_latencies(records: List[InferenceRecord]) -> dict:
    """Compute mean / p95 from inference records that have latency_ms."""
    if not records:
        return {"avg_latency_ms": {}, "p95_latency_ms": {}}

    keys = ["decode", "quality", "preprocess", "inference", "postprocess", "total"]
    vals: Dict[str, list] = {k: [] for k in keys}
    for r in records:
        lat = r.latency_ms
        for k in keys:
            v = lat.get(k)
            if v is not None:
                vals[k].append(v)

    avg = {}
    p95 = {}
    for k in keys:
        if vals[k]:
            avg[k] = round(float(np.mean(vals[k])), 2)
            p95[k] = round(float(np.percentile(vals[k], 95)), 2)
    return {"avg_latency_ms": avg, "p95_latency_ms": p95}
