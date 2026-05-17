"""
Load CLI pipeline outputs for dashboard review (bad / detection samples).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.paths import (
    output_dir_display,
    output_dir_static_rel,
    resolve_output_dir,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline-review"])


def _load_index(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("samples") or [])


def _preview_url(root: Path, static_rel: Optional[str], subdir: str, fname: str) -> Optional[str]:
    if not fname:
        return None
    if static_rel is not None:
        return f"/outputs/{static_rel}/{subdir}/{fname}"
    root_s = quote(str(root.resolve()), safe="")
    return (
        f"/pipeline/sample_image?output_dir={root_s}"
        f"&subdir={quote(subdir, safe='')}&file={quote(fname, safe='')}"
    )


@router.get("/sample_image", include_in_schema=False)
def pipeline_sample_image(
    output_dir: str = Query(...),
    subdir: str = Query(..., description="bad_samples or detection_samples"),
    file: str = Query(...),
):
    """Serve sample JPG when output_dir is outside the /outputs static mount."""
    root = resolve_output_dir(output_dir)
    if subdir not in ("bad_samples", "detection_samples"):
        raise HTTPException(status_code=400, detail="Invalid subdir")
    p = (root / subdir / file).resolve()
    root_res = root.resolve()
    if not str(p).startswith(str(root_res)):
        raise HTTPException(status_code=403, detail="Path escape")
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Image not found: {p}")
    return FileResponse(str(p), media_type="image/jpeg")


@router.get("/review_index")
def pipeline_review_index(
    output_dir: str = Query(..., description="Pipeline output directory"),
):
    """
    Return bad-quality and detection sample lists from a completed CLI run.
    Each entry includes topic, pipeline frame_seq, timestamp_ns, and preview image URL.
    """
    root = resolve_output_dir(output_dir)
    if not root.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Output directory not found: {output_dir} (resolved: {root})",
        )

    static_rel = output_dir_static_rel(root)
    display_dir = output_dir_display(root)
    bad_dir = root / "bad_samples"
    det_dir = root / "detection_samples"

    bad_samples = _load_index(bad_dir / "index.json")
    det_samples = _load_index(det_dir / "index.json")

    def _enrich(samples: List[Dict[str, Any]], subdir: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in samples:
            fname = s.get("file") or ""
            entry = {
                "topic": s.get("topic", ""),
                "frame_seq": s.get("frame_seq", 0),
                "timestamp_ns": s.get("timestamp_ns", 0),
                "quality_score": s.get("quality_score"),
                "quality_tags": s.get("quality_tags") or [],
                "mcap_file": s.get("mcap_file", ""),
                "preview_url": _preview_url(root, static_rel, subdir, fname),
            }
            if s.get("objects"):
                entry["objects"] = s["objects"]
            out.append(entry)
        return out

    quality_report = root / "quality_report.json"
    topics_summary: List[Dict[str, Any]] = []
    if quality_report.exists():
        qr = json.loads(quality_report.read_text(encoding="utf-8"))
        for t in qr.get("topics") or []:
            topics_summary.append(
                {
                    "topic": t.get("topic"),
                    "total_frames": t.get("total_frames"),
                    "processed_frames": t.get("processed_frames"),
                    "bad_quality_frames": t.get("bad_quality_frames"),
                }
            )

    metrics_path = root / "metrics.json"
    metrics: Optional[Dict[str, Any]] = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    predictions_path = root / "yolo_predictions.json"
    predictions: List[Dict[str, Any]] = []
    if predictions_path.exists():
        raw = json.loads(predictions_path.read_text(encoding="utf-8"))
        for p in raw.get("predictions") or []:
            entry: Dict[str, Any] = {
                "mcap_file": p.get("mcap_file", ""),
                "topic": p.get("topic", ""),
                "frame_seq": p.get("frame_seq", 0),
                "raw_frame_idx": p.get("raw_frame_idx"),
                "timestamp_ns": p.get("timestamp_ns", 0),
                "quality_score": p.get("quality_score"),
                "quality_tags": p.get("quality_tags") or [],
                "action": p.get("action", ""),
            }
            if p.get("reason"):
                entry["reason"] = p["reason"]
            if p.get("quality_penalties"):
                entry["quality_penalties"] = p["quality_penalties"]
            if p.get("objects"):
                entry["objects"] = p["objects"]
            if p.get("latency_ms"):
                entry["latency_ms"] = p["latency_ms"]
            if p.get("model"):
                entry["model"] = p["model"]
            if p.get("target_classes"):
                entry["target_classes"] = p["target_classes"]
            predictions.append(entry)

    report_q = quote(display_dir, safe="")
    metrics_json_url = (
        f"/outputs/{static_rel}/metrics.json"
        if static_rel
        else f"/report/metrics?output_dir={report_q}"
    )

    return {
        "output_dir": str(root.resolve()),
        "output_dir_display": display_dir,
        "predictions": predictions,
        "bad_frames": _enrich(bad_samples, "bad_samples"),
        "detection_frames": _enrich(det_samples, "detection_samples"),
        "topics_summary": topics_summary,
        "metrics": metrics,
        "report_urls": {
            "quality_html": f"/report/quality?output_dir={report_q}",
            "yolo_html": f"/report/yolo?output_dir={report_q}",
            "metrics_json": metrics_json_url,
        },
    }
