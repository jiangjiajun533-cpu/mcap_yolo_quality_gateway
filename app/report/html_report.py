"""
Simplified HTML reports (FR-REPORT-002, FR-REPORT-003).

Single-page tables + sample thumbnails. No external chart libraries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, BaseLoader
from markupsafe import Markup

from app.core.config import settings
from app.core.logging import get_logger
from app.quality.aggregator import TopicQualitySummary
from app.quality.duplicate import DuplicateGroup, duplicate_groups_to_dict
from app.quality.sequence_analyzer import SequenceSummary
from app.yolo.pipeline import PipelineStats, InferenceRecord
from app.yolo.target_analyzer import TargetAnalyzer

logger = get_logger("report.html")

_ENV = Environment(loader=BaseLoader(), autoescape=True)

_CSS = """
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f6f8fa; color: #1f2328; line-height: 1.5; }
  a { color: #0969da; }
  h1 { font-size: 1.5rem; margin: 0 0 8px; }
  h2 { font-size: 1.1rem; margin: 24px 0 8px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
  .meta { color: #656d76; font-size: 0.9rem; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d7de; margin-bottom: 16px; font-size: 0.88rem; }
  th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
  th { background: #f6f8fa; font-weight: 600; }
  tr:nth-child(even) { background: #f6f8fa; }
  .bad { color: #cf222e; font-weight: 600; }
  .ok { color: #1a7f37; }
  .gallery { display: flex; flex-wrap: wrap; gap: 12px; }
  .gallery img { max-width: 200px; max-height: 150px; border: 1px solid #d0d7de; border-radius: 4px; }
  .cap { font-size: 0.75rem; color: #656d76; max-width: 200px; margin-top: 4px; }
  .back { display: inline-block; margin-bottom: 16px; }
  .cam-browser {
    margin: 16px 0 24px; padding: 14px 16px 18px;
    background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  }
  .cam-browser-toolbar { margin-bottom: 12px; }
  .cam-browser-toolbar .meta { margin: 0 0 6px; }
  .cam-panels { min-height: 120px; }
  .cam-panel[hidden] { display: none !important; }
  .cam-panel-head { margin-bottom: 10px; font-size: 0.9rem; color: #656d76; }
  .cam-panel-head strong { color: #1f2328; }
  .cam-tabs {
    display: flex; flex-wrap: wrap; gap: 6px; padding: 0 0 10px; margin-bottom: 12px;
    overflow-x: auto; border-bottom: 1px solid #d0d7de;
  }
  .cam-tab {
    flex: 0 0 auto; padding: 8px 12px; border: 1px solid #d0d7de; border-radius: 8px;
    background: #f6f8fa; cursor: pointer; font-size: 0.82rem; white-space: nowrap;
  }
  .cam-tab.active { background: #0969da; color: #fff; border-color: #0969da; }
  .cam-tab .cnt { opacity: 0.9; font-size: 0.72rem; margin-left: 4px; }
</style>
"""

_CAM_GALLERY_TPL = """
{% if gallery and gallery.by_topic %}
<h2>{{ heading }} {{ gallery.title }}</h2>
<div class="cam-browser" id="{{ browser_id }}">
  <div class="cam-browser-toolbar">
    <nav class="cam-tabs" role="tablist" aria-label="Camera views">
    {% for section in gallery.by_topic %}
    <button type="button" class="cam-tab{% if loop.first %} active{% endif %}" role="tab"
      data-cam="{{ section.slug }}" aria-selected="{% if loop.first %}true{% else %}false{% endif %}">
      {{ section.topic_short }}<span class="cnt">({{ section.thumbnails | length }})</span>
    </button>
    {% endfor %}
    </nav>
  </div>
  <div class="cam-panels">
  {% for section in gallery.by_topic %}
  <section class="cam-panel" data-cam="{{ section.slug }}"{% if not loop.first %} hidden{% endif %}>
    <div class="cam-panel-head"><strong>{{ section.topic_short }}</strong> — {{ section.thumbnails | length }} frame(s)</div>
    <div class="gallery">
    {% for s in section.thumbnails %}
    <div>
      <img src="{{ s.src }}" alt="" loading="lazy">
      <div class="cap">{{ s.caption }}</div>
    </div>
    {% endfor %}
    </div>
  </section>
  {% endfor %}
  </div>
</div>
<script>
(function () {
  var root = document.getElementById({{ browser_id | tojson }});
  if (!root) return;
  var panels = root.querySelectorAll(".cam-panel");
  var tabs = root.querySelectorAll(".cam-tab");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var cam = tab.getAttribute("data-cam");
      tabs.forEach(function (t) {
        var on = t === tab;
        t.classList.toggle("active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach(function (p) { p.hidden = p.getAttribute("data-cam") !== cam; });
    });
  });
})();
</script>
{% endif %}
"""

_QUALITY_TPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Quality Report</title>""" + _CSS + """</head><body>
<a class="back" href="/">← Back to Dashboard</a>
<h1>Quality Report</h1>
{% if subtitle %}<p class="meta">{{ subtitle }}</p>{% endif %}

<h2>Per-topic summary</h2>
<table>
<tr><th>Topic</th><th>Processed</th><th>Bad</th><th>Bad %</th><th>Avg score</th><th>Main issues</th></tr>
{% for t in topics %}
<tr>
  <td>{{ t.short }}</td>
  <td>{{ t.processed_frames }}</td>
  <td class="{% if t.bad_quality_frames > 0 %}bad{% endif %}">{{ t.bad_quality_frames }}</td>
  <td>{{ (t.bad_quality_ratio * 100) | round(1) }}%</td>
  <td>{{ t.avg_quality_score }}</td>
  <td>{% for k,v in (t.quality_issue_counts or {}).items() %}{{ k }}({{ v }}) {% endfor %}</td>
</tr>
{% endfor %}
</table>

{% if worst %}
<h2>Worst frames (top {{ worst | length }})</h2>
<table>
<tr><th>#</th><th>Frame</th><th>Score</th><th>Tags</th></tr>
{% for w in worst %}
<tr>
  <td>{{ loop.index }}</td>
  <td>{{ w.frame_seq }}</td>
  <td class="bad">{{ w.quality_score }}</td>
  <td>{{ (w.quality_tags or []) | join(', ') }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

{{ bad_gallery_html }}

{% if dup_groups %}
<h2>Duplicate / near-duplicate frame groups</h2>
<table>
<tr><th>Topic</th><th>Start frame</th><th>End frame</th><th>Duration (s)</th></tr>
{% for g in dup_groups %}
<tr><td>{{ g.topic_short }}</td><td>{{ g.start }}</td><td>{{ g.end }}</td><td>{{ g.dur }}</td></tr>
{% endfor %}
</table>
{% endif %}

</body></html>"""

_YOLO_TPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>YOLO Report</title>""" + _CSS + """</head><body>
<a class="back" href="/">← Back to Dashboard</a>
<h1>YOLO Detection Report</h1>
<p class="meta">Model: {{ model_name }} · {{ model_format }} · {{ input_size }} · {{ backend }}</p>

<h2>Pipeline stats</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Sampled frames</td><td>{{ sampled }}</td></tr>
<tr><td>Inferred</td><td>{{ inferred }}</td></tr>
<tr><td>Skipped (low quality)</td><td>{{ skipped_quality }}</td></tr>
<tr><td>Skipped (depth topic)</td><td>{{ skipped_depth }}</td></tr>
<tr><td>Inference failed</td><td>{{ infer_failed }}</td></tr>
<tr><td>Throughput</td><td>{{ throughput }} frames/s</td></tr>
</table>

{% if targets %}
<h2>Target class statistics</h2>
<table>
<tr><th>Class</th><th>Count</th><th>Avg conf</th><th>Avg quality</th></tr>
{% for t in targets %}
<tr>
  <td>{{ t.label }}</td>
  <td>{{ t.detected_count }}</td>
  <td>{{ t.avg_confidence }}</td>
  <td>{{ t.avg_quality_score }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

{% if perf_rows %}
<h2>Latency (ms)</h2>
<table>
<tr><th>Stage</th><th>Avg</th><th>P95</th></tr>
{% for r in perf_rows %}
<tr><td>{{ r.stage }}</td><td>{{ r.avg }}</td><td>{{ r.p95 }}</td></tr>
{% endfor %}
</table>
{% endif %}

{% if per_topic %}
<h2>Per-topic target distribution</h2>
<table>
<tr><th>Topic</th>{% for cls in per_topic_classes %}<th>{{ cls }}</th>{% endfor %}</tr>
{% for row in per_topic %}
<tr><td>{{ row.topic_short }}</td>{% for c in row.counts %}<td>{{ c }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% endif %}

{% if qc_summary %}
<h2>Quality score vs detection confidence</h2>
<table>
<tr><th>Quality bucket</th><th>Frames inferred</th><th>Avg objects</th><th>Avg confidence</th></tr>
{% for b in qc_summary %}
<tr><td>{{ b.bucket }}</td><td>{{ b.frames }}</td><td>{{ b.avg_obj }}</td><td>{{ b.avg_conf }}</td></tr>
{% endfor %}
</table>
{% endif %}

{{ det_gallery_html }}

</body></html>"""


def _topic_short(topic: str) -> str:
    if "realsense_head" in topic:
        return "Head Depth" if "depth" in topic else "Head RGB"
    if "realsense_up" in topic:
        return "Up Depth" if "depth" in topic else "Up RGB"
    if "right_wrist" in topic:
        return "Right Wrist"
    if "left_wrist" in topic:
        return "Left Wrist"
    parts = topic.split("/")
    return parts[-1] if parts else topic


def _output_web_prefix(output_dir: Path) -> str:
    """URL prefix under /outputs/ for static assets in this run."""
    p = output_dir.as_posix().replace("\\", "/").lstrip("./")
    if p.startswith("outputs/"):
        p = p[len("outputs/") :]
    return f"/outputs/{p}"


def _gallery_title(shown: int, exported_total: int) -> str:
    """Dynamic heading suffix from index.json counts (not hard-coded totals)."""
    if exported_total <= 0:
        return ""
    if shown >= exported_total:
        return f"({exported_total} exported)"
    return f"(showing {shown} of {exported_total} exported)"


_CAMERA_ORDER = [
    "Head RGB", "Head Depth", "Up RGB", "Up Depth", "Left Wrist", "Right Wrist",
]


def _empty_gallery() -> dict:
    return {"thumbnails": [], "shown": 0, "exported_total": 0, "title": "", "by_topic": []}


def _sample_caption(entry: dict, *, is_bad: bool) -> str:
    """Human-readable label: MCAP raw index + per-topic sampled index (not fused)."""
    parts: List[str] = []
    raw = entry.get("raw_frame_idx")
    if raw is not None:
        parts.append(f"raw #{raw}")
    if entry.get("frame_seq") is not None:
        parts.append(f"sampled #{entry['frame_seq']}")
    if is_bad:
        if entry.get("quality_score") is not None:
            parts.append(f"score={entry['quality_score']}")
        tags = entry.get("quality_tags") or []
        if tags:
            parts.append(", ".join(str(t) for t in tags))
    else:
        labels = [o.get("label", "") for o in (entry.get("objects") or []) if o.get("label")]
        if labels:
            parts.append(" ".join(labels[:4]))
    return " · ".join(parts)


def _cam_slug(topic_short: str, index: int = 0) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic_short.lower()).strip("-")
    return slug or f"cam-{index}"


def _group_gallery_by_topic(thumbnails: List[dict]) -> List[dict]:
    buckets: Dict[str, List[dict]] = {}
    for t in thumbnails:
        key = t.get("topic_short") or "Other"
        buckets.setdefault(key, []).append(t)

    def _sort_key(name: str) -> tuple:
        try:
            return (0, _CAMERA_ORDER.index(name))
        except ValueError:
            return (1, name)

    sorted_names = sorted(buckets.keys(), key=_sort_key)
    return [
        {
            "topic_short": name,
            "slug": _cam_slug(name, i),
            "thumbnails": sorted(
                buckets[name],
                key=lambda t: (t.get("frame_seq") is None, t.get("frame_seq", 0)),
            ),
        }
        for i, name in enumerate(sorted_names)
    ]


def _load_sampling_hint(output_dir: Path) -> str:
    """Explain decimation + caption numbering for the gallery toolbar."""
    mp = output_dir / "metrics.json"
    n = fps = None
    if mp.exists():
        sm = json.loads(mp.read_text(encoding="utf-8")).get("sampling") or {}
        n = sm.get("computed_sample_every_n")
        fps = sm.get("estimated_actual_fps")
    parts: List[str] = []
    if n:
        parts.append(f"流水线按约每 {n} 条原始消息取 1 帧")
    if fps:
        parts.append(f"≈{float(fps):.1f} FPS")
    lead = " · ".join(parts) if parts else "流水线对 MCAP 做了抽帧"
    return (
        f"{lead}。"
        "图中 sampled # 是该摄像机已采样序号（连续递增）；"
        "raw # 是 MCAP 原始消息序号（间隔更大）。"
        "导出检测样例时按时间均匀抽取，避免只显示连续片段。"
    )


def _format_model_line(model_info: Optional[dict]) -> str:
    if not model_info:
        return ""
    name = model_info.get("name", "yolov8n")
    fmt = model_info.get("format", "onnx")
    sz = model_info.get("input_size")
    size = "×".join(str(x) for x in sz) if sz else "640×640"
    backend = model_info.get("backend", "onnxruntime")
    return f"Model: {name} · {fmt} · {size} · {backend}"


def _render_cam_gallery_browser(
    gallery: dict,
    *,
    browser_id: str,
    heading: str,
    info_line: str = "",
    sampling_hint: str = "",
) -> Markup:
    if not gallery.get("by_topic"):
        return Markup("")
    return Markup(
        _ENV.from_string(_CAM_GALLERY_TPL).render(
            gallery=gallery,
            browser_id=browser_id,
            heading=heading,
            info_line=info_line,
            sampling_hint=sampling_hint,
        )
    )


def _load_report_subtitle(
    output_dir: Path,
    pipeline_stats: Optional[PipelineStats],
    topic_count: int,
) -> str:
    """Build header line from mcap_summary.json + metrics (no N/A placeholders)."""
    parts: List[str] = []
    summ_path = output_dir / "mcap_summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        files = summ.get("files") or []
        if files:
            name = files[0].get("mcap_file")
            if name:
                parts.append(str(name))
            dur = files[0].get("duration_sec")
            if dur is not None:
                parts.append(f"{float(dur):.1f}s")
    if topic_count:
        parts.append(f"{topic_count} topics")
    sampled: Optional[int] = None
    if pipeline_stats is not None:
        sampled = pipeline_stats.sampled_frames
    if sampled is None:
        mp = output_dir / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            sampled = (m.get("frames") or {}).get("sampled_frames")
    if sampled is not None:
        parts.append(f"{sampled} sampled frames")
    return " · ".join(parts)


def _load_sample_gallery(
    output_dir: Path,
    subdir: str,
    limit: Optional[int] = None,
    *,
    full: bool = False,
) -> dict:
    """
    Load exported sample thumbnails for HTML embedding.
    ``exported_total`` comes from index.json; ``shown`` is min(total, preview limit)
    unless ``full=True`` (load every exported sample, grouped per camera).
    """
    if limit is None:
        limit = 0 if full else settings.html_gallery_preview_limit
    idx = output_dir / subdir / "index.json"
    if not idx.exists():
        return _empty_gallery()

    prefix = _output_web_prefix(output_dir)
    data = json.loads(idx.read_text(encoding="utf-8"))
    all_samples: List[dict] = list(data.get("samples") or [])
    exported_total = len(all_samples)
    preview = all_samples[:limit] if limit > 0 else all_samples

    thumbnails: List[dict] = []
    for s in preview:
        fname = s.get("file") or ""
        if not fname:
            continue
        topic = s.get("topic", "")
        is_bad = subdir == "bad_samples"
        entry: dict = {
            "src": f"{prefix}/{subdir}/{fname}",
            "topic": topic,
            "topic_short": _topic_short(topic),
            "frame_seq": s.get("frame_seq"),
            "raw_frame_idx": s.get("raw_frame_idx"),
        }
        if is_bad:
            entry["quality_score"] = s.get("quality_score")
            entry["quality_tags"] = s.get("quality_tags") or []
        else:
            entry["objects"] = s.get("objects") or []
        entry["caption"] = _sample_caption(entry, is_bad=is_bad)
        thumbnails.append(entry)

    shown = len(thumbnails)
    return {
        "thumbnails": thumbnails,
        "shown": shown,
        "exported_total": exported_total,
        "title": _gallery_title(shown, exported_total),
        "by_topic": _group_gallery_by_topic(thumbnails),
    }


def _load_bad_samples(
    output_dir: Path,
    limit: Optional[int] = None,
    *,
    full: bool = False,
) -> dict:
    return _load_sample_gallery(output_dir, "bad_samples", limit=limit, full=full)


def _load_det_samples(
    output_dir: Path,
    limit: Optional[int] = None,
    *,
    full: bool = False,
) -> dict:
    return _load_sample_gallery(
        output_dir, "detection_samples", limit=limit, full=full
    )


def _flatten_worst(output_dir: Path, limit: int = 20) -> List[dict]:
    qr_path = output_dir / "quality_report.json"
    if not qr_path.exists():
        return []
    qr = json.loads(qr_path.read_text(encoding="utf-8"))
    rows: List[dict] = []
    for frames in (qr.get("worst_frames") or {}).values():
        if isinstance(frames, list):
            rows.extend(frames)
    rows.sort(key=lambda x: x.get("quality_score", 1.0))
    return rows[:limit]


def write_quality_html(
    output_dir: Path,
    topic_summaries: List[Any],
    sequence_summaries: Optional[List[SequenceSummary]] = None,
    pipeline_stats: Optional[PipelineStats] = None,
    worst_frames: Optional[List[dict]] = None,
    bad_sample_files: Optional[List[dict]] = None,
    all_scores: Optional[List[float]] = None,
    dup_results: Optional[Dict[str, List[DuplicateGroup]]] = None,
) -> Path:
    """Generate simplified quality_report.html."""
    topics = []
    for ts in topic_summaries:
        d = ts.to_dict() if hasattr(ts, "to_dict") else dict(ts)
        d["short"] = _topic_short(d.get("topic", ""))
        topics.append(d)

    topic_count = len(topics)
    subtitle = _load_report_subtitle(output_dir, pipeline_stats, topic_count)

    worst = worst_frames if worst_frames else _flatten_worst(output_dir)
    bad_gallery = _load_bad_samples(output_dir, full=True)
    bad_gallery_html = _render_cam_gallery_browser(
        bad_gallery,
        browser_id="bad-cam-gallery",
        heading="Bad sample images",
        sampling_hint=_load_sampling_hint(output_dir),
    )

    # Duplicate groups for template
    dup_groups: List[dict] = []
    if dup_results:
        for topic, groups in dup_results.items():
            for g in groups:
                dup_groups.append({
                    "topic_short": _topic_short(topic),
                    "start": g.start_frame_seq, "end": g.end_frame_seq,
                    "dur": round(g.duration_sec, 2),
                })

    html = _ENV.from_string(_QUALITY_TPL).render(
        subtitle=subtitle,
        topics=topics,
        worst=worst,
        bad_gallery_html=bad_gallery_html,
        dup_groups=dup_groups,
    )
    out = output_dir / "quality_report.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {out} ({out.stat().st_size} bytes)")
    return out


def write_yolo_html(
    output_dir: Path,
    pipeline_stats: Optional[PipelineStats] = None,
    target_analyzer: Optional[TargetAnalyzer] = None,
    model_info: Optional[dict] = None,
    perf: Optional[dict] = None,
    target_classes: Optional[List[str]] = None,
    per_topic_detections: Optional[dict] = None,
    detection_sample_files: Optional[List[dict]] = None,
    quality_confidence_pairs: Optional[List[dict]] = None,
    records: Optional[List[InferenceRecord]] = None,
) -> Path:
    """Generate simplified yolo_report.html."""
    model_name = "yolov8n"
    model_format = "onnx"
    input_size = "640×640"
    backend = "onnxruntime"
    if model_info:
        model_name = model_info.get("name", model_name)
        model_format = model_info.get("format", model_format)
        sz = model_info.get("input_size")
        if sz:
            input_size = "×".join(str(x) for x in sz)
        backend = model_info.get("backend", backend)

    sampled = inferred = skipped_quality = skipped_depth = infer_failed = 0
    throughput = "—"
    if pipeline_stats:
        sd = pipeline_stats.to_dict()
        fr = sd.get("frames", {})
        sampled = fr.get("sampled_frames", 0)
        inferred = fr.get("infer_success_frames", 0)
        skipped_quality = fr.get("skipped_low_quality_frames", 0)
        skipped_depth = fr.get("skipped_depth_topic_frames", 0)
        infer_failed = fr.get("infer_failed_frames", 0)
    if perf:
        throughput = perf.get("processed_frames_per_sec", throughput)

    targets: List[dict] = []
    if target_analyzer:
        targets = target_analyzer.finalize().get("target_analysis", [])
    elif (output_dir / "metrics.json").exists():
        m = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        targets = m.get("target_analysis") or []

    perf_rows: List[dict] = []
    if perf:
        avg = perf.get("avg_latency_ms") or {}
        p95 = perf.get("p95_latency_ms") or {}
        for stage in ("decode", "quality", "preprocess", "inference", "postprocess", "total"):
            if stage in avg:
                perf_rows.append({
                    "stage": stage,
                    "avg": avg[stage],
                    "p95": p95.get(stage, "—"),
                })

    det_gallery = _load_det_samples(output_dir, full=True)
    model_info_dict = model_info or {}
    if not model_info_dict and (output_dir / "metrics.json").exists():
        m = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        model_info_dict = m.get("model") or {}
    det_gallery_html = _render_cam_gallery_browser(
        det_gallery,
        browser_id="det-cam-gallery",
        heading="Detection samples",
        info_line=_format_model_line(model_info_dict),
        sampling_hint=_load_sampling_hint(output_dir),
    )

    # Per-topic target distribution
    per_topic: List[dict] = []
    per_topic_classes: List[str] = []
    if records:
        topic_class_counts: Dict[str, Dict[str, int]] = {}
        all_classes_set: set = set()
        for r in records:
            if r.action == "inferred" and r.objects:
                t = _topic_short(r.topic)
                if t not in topic_class_counts:
                    topic_class_counts[t] = {}
                for o in r.objects:
                    lbl = o.label if hasattr(o, "label") else o.get("label", "?")
                    topic_class_counts[t][lbl] = topic_class_counts[t].get(lbl, 0) + 1
                    all_classes_set.add(lbl)
        per_topic_classes = sorted(all_classes_set)
        for t_short, cc in sorted(topic_class_counts.items()):
            per_topic.append({
                "topic_short": t_short,
                "counts": [cc.get(c, 0) for c in per_topic_classes],
            })

    # Quality-confidence buckets
    qc_summary: List[dict] = []
    if records:
        buckets: Dict[str, List] = {"0.0–0.6": [], "0.6–0.8": [], "0.8–1.0": []}
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
        for bname, recs in buckets.items():
            if not recs:
                qc_summary.append({"bucket": bname, "frames": 0, "avg_obj": "—", "avg_conf": "—"})
                continue
            n_obj = [len(r.objects) for r in recs]
            confs = [o.confidence for r in recs for o in r.objects if hasattr(o, "confidence")]
            qc_summary.append({
                "bucket": bname,
                "frames": len(recs),
                "avg_obj": round(sum(n_obj) / len(n_obj), 2),
                "avg_conf": round(sum(confs) / len(confs), 3) if confs else "—",
            })

    avg_infer_ms = "—"
    p95_infer_ms = "—"
    if perf:
        avg_infer_ms = perf.get("avg_latency_ms", {}).get("inference", "—")
        p95_infer_ms = perf.get("p95_latency_ms", {}).get("inference", "—")

    html = _ENV.from_string(_YOLO_TPL).render(
        model_name=model_name,
        model_format=model_format,
        input_size=input_size,
        backend=backend,
        sampled=sampled,
        inferred=inferred,
        skipped_quality=skipped_quality,
        skipped_depth=skipped_depth,
        infer_failed=infer_failed,
        throughput=throughput,
        targets=targets,
        perf_rows=perf_rows,
        det_gallery_html=det_gallery_html,
        per_topic=per_topic,
        per_topic_classes=per_topic_classes,
        qc_summary=qc_summary,
        avg_infer_ms=avg_infer_ms,
        p95_infer_ms=p95_infer_ms,
    )
    out = output_dir / "yolo_report.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {out} ({out.stat().st_size} bytes)")
    return out


def regenerate_html_reports(output_dir: Path) -> None:
    """Regenerate HTML only from existing JSON in output_dir (no MCAP re-run)."""
    qr_path = output_dir / "quality_report.json"
    if qr_path.exists():
        qr = json.loads(qr_path.read_text(encoding="utf-8"))
        topics = list(qr.get("topics") or [])
        stats = PipelineStats()
        stats.mcap_file = "sample.mcap"
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            fr = m.get("frames") or {}
            stats.sampled_frames = fr.get("sampled_frames", 0)
            stats.quality_failed_frames = fr.get("quality_failed_frames", 0)
        write_quality_html(output_dir, topics, pipeline_stats=stats)

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        stats = PipelineStats()
        fr = m.get("frames") or {}
        for key in (
            "sampled_frames", "infer_success_frames", "skipped_low_quality_frames",
            "skipped_depth_topic_frames", "infer_failed_frames",
        ):
            if key in fr:
                setattr(stats, key, fr[key])
        perf = m.get("performance") or {}
        model_info = {"name": "yolov8n", "format": "onnx", "input_size": [640, 640], "backend": "onnxruntime"}
        write_yolo_html(output_dir, pipeline_stats=stats, model_info=model_info, perf=perf)
