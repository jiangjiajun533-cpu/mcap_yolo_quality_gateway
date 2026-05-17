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
  body { font-family: system-ui, sans-serif; margin: 0; padding: 20px 24px 32px; background: #f6f8fa; color: #1f2328; line-height: 1.5; }
  .page { max-width: 960px; margin: 0 auto; }
  a { color: #0969da; }
  h1 { font-size: 1.35rem; margin: 8px 0 10px; font-weight: 600; }
  h2 { font-size: 1rem; margin: 28px 0 10px; border-bottom: 1px solid #d0d7de; padding-bottom: 5px; font-weight: 600; }
  h2:first-of-type { margin-top: 16px; }
  .meta { color: #656d76; font-size: 0.85rem; margin-bottom: 12px; line-height: 1.45; }
  .conclusion-box {
    background: #fff; border: 1px solid #d0d7de; border-left: 4px solid #0969da;
    padding: 12px 16px; margin: 8px 0 20px; font-size: 0.88rem; line-height: 1.6; border-radius: 0 6px 6px 0;
  }
  .trend-chart-wrap {
    max-width: 680px; margin: 10px 0 20px; padding: 14px 16px 10px;
    background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  }
  .trend-chart-wrap .meta { margin: 0 0 10px; font-size: 0.8rem; }
  .trend-chart-wrap svg { display: block; width: 100%; max-height: 220px; height: auto; }
  .issue-chart-wrap { max-width: 560px; margin: 8px 0 16px; }
  .dup-section { margin-top: 24px; }
  .dup-section table { font-size: 0.84rem; max-width: 640px; }
  table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d0d7de; margin-bottom: 16px; font-size: 0.86rem; }
  th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
  th { background: #f6f8fa; font-weight: 600; }
  tr:nth-child(even) { background: #f6f8fa; }
  .bad { color: #cf222e; font-weight: 600; }
  .ok { color: #1a7f37; }
  .gallery { display: flex; flex-wrap: wrap; gap: 10px; }
  .gallery img { max-width: 168px; max-height: 126px; border: 1px solid #d0d7de; border-radius: 4px; }
  .cap { font-size: 0.72rem; color: #656d76; max-width: 168px; margin-top: 4px; line-height: 1.35; }
  .back { display: inline-block; margin-bottom: 4px; font-size: 0.88rem; }
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
<div class="page">
<a class="back" href="/">← Back to Dashboard</a>
<h1>Quality Report</h1>
{% if subtitle %}<p class="meta">{{ subtitle }}</p>{% endif %}

{% if batch_failures %}
<h2>批量 MCAP 处理失败</h2>
<p class="meta">目录模式下个别文件异常已跳过；其余文件结果已合并。</p>
<table>
<tr><th>MCAP 路径</th><th>原因</th></tr>
{% for f in batch_failures %}
<tr><td style="font-size:12px;word-break:break-all">{{ f.mcap_file }}</td><td class="bad" style="font-size:12px">{{ f.error }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>Per-topic summary</h2>
<table>
<tr><th>Topic</th><th>Duration</th><th>FPS</th><th>分辨率变化</th><th>时间戳跳变</th><th>Processed</th><th>Bad</th><th>Bad %</th><th>Avg score</th><th>Main issues</th></tr>
{% for t in topics %}
<tr>
  <td>{{ t.short }}</td>
  <td>{{ t.duration_sec if t.duration_sec is not none else '—' }}</td>
  <td>{{ t.estimated_fps if t.estimated_fps is not none else '—' }}</td>
  <td>{{ t.resolution_change_count if t.resolution_change_count is not none else '—' }}</td>
  <td>{{ t.timestamp_jump_count if t.timestamp_jump_count is not none else '—' }}</td>
  <td>{{ t.processed_frames }}</td>
  <td class="{% if t.bad_quality_frames > 0 %}bad{% endif %}">{{ t.bad_quality_frames }}</td>
  <td>{{ (t.bad_quality_ratio * 100) | round(1) }}%</td>
  <td>{{ t.avg_quality_score }}</td>
  <td>{% for k,v in (t.quality_issue_counts or {}).items() %}{{ k }}({{ v }}) {% endfor %}</td>
</tr>
{% endfor %}
</table>

{% if issue_chart %}
<h2>Quality Issue Distribution</h2>
<div class="issue-chart-wrap">
{% for item in issue_chart %}
<div style="display:flex;align-items:center;margin:4px 0">
  <span style="width:160px;font-size:13px;text-align:right;padding-right:10px;white-space:nowrap">{{ item.label }}</span>
  <div style="flex:1;background:#eee;border-radius:3px;height:22px;position:relative">
    <div style="width:{{ item.pct }}%;background:#e74c3c;height:100%;border-radius:3px;min-width:2px"></div>
  </div>
  <span style="width:50px;font-size:13px;padding-left:8px">{{ item.count }}</span>
</div>
{% endfor %}
</div>
{% endif %}

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

{% if trend_svg %}
<h2>多相机质量趋势</h2>
<div class="trend-chart-wrap">
<p class="meta">各相机采样帧质量分走势（0–1）；折线按 topic 分组。</p>
{{ trend_svg | safe }}
</div>
{% endif %}

{% if conclusion %}
<h2>总体结论</h2>
<div class="conclusion-box">{{ conclusion | safe }}</div>
{% endif %}

{% if dup_groups %}
<div class="dup-section">
<h2>重复 / 近重复帧组</h2>
<p class="meta">加分项：感知哈希检测到的连续近重复片段。</p>
<table>
<tr><th>Topic</th><th>Start frame</th><th>End frame</th><th>Duration (s)</th></tr>
{% for g in dup_groups %}
<tr><td>{{ g.topic_short }}</td><td>{{ g.start }}</td><td>{{ g.end }}</td><td>{{ g.dur }}</td></tr>
{% endfor %}
</table>
</div>
{% endif %}

</div>
</body></html>"""

_YOLO_TPL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>YOLO Report</title>""" + _CSS + """</head><body>
<div class="page">
<a class="back" href="/">← Back to Dashboard</a>
<h1>YOLO Detection Report</h1>
<p class="meta">模型：{{ model_name }} · 格式：{{ model_format }} · 输入：{{ input_size }} · 推理后端：{{ backend }} · 计算设备：{{ device }}</p>
{% if target_classes_str %}<p class="meta">目标类别：{{ target_classes_str }}</p>{% endif %}

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

{% if yolo_conclusion %}
<h2>总体结论</h2>
<div class="conclusion-box">{{ yolo_conclusion | safe }}</div>
{% endif %}

</div>
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
    if pipeline_stats is not None:
        ptr = pipeline_stats.to_dict().get("processing_time_range") or {}
        cs, ce = ptr.get("clip_start_sec"), ptr.get("clip_end_sec")
        if cs or ce:
            end_label = f"{ce}s" if ce and ce > 0 else "end"
            parts.append(f"clip {cs}s–{end_label}")
        pd = ptr.get("processed_duration_sec")
        if pd is not None:
            parts.append(f"processed span {pd}s")
    mp = output_dir / "metrics.json"
    if mp.exists():
        mi = (json.loads(mp.read_text(encoding="utf-8")).get("model") or {})
        dev = mi.get("device")
        backend = mi.get("backend")
        if dev or backend:
            parts.append(f"推理：{backend or 'onnxruntime'} / {dev or 'cpu'}")
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


def _generate_quality_conclusion(topics: List[dict]) -> str:
    """自动生成质量报告总体结论（中文）。"""
    total_processed = sum(d.get("processed_frames", 0) for d in topics)
    total_bad = sum(d.get("bad_quality_frames", 0) for d in topics)
    bad_ratio = total_bad / total_processed if total_processed else 0
    avg_scores = [d.get("avg_quality_score", 0) for d in topics if d.get("avg_quality_score")]
    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0

    all_issues: Dict[str, int] = {}
    for d in topics:
        for k, v in (d.get("quality_issue_counts") or {}).items():
            all_issues[k] = all_issues.get(k, 0) + v
    top_issues = sorted(all_issues.items(), key=lambda x: -x[1])[:3]

    parts = []
    parts.append(f"共分析 <b>{len(topics)}</b> 个摄像机 Topic，合计处理 <b>{total_processed}</b> 帧。")
    if bad_ratio < 0.05:
        parts.append(f"整体图像质量<b>优秀</b>——仅 {bad_ratio:.1%} 的帧被标记为低质量（平均质量分 {overall_avg:.3f}）。")
    elif bad_ratio < 0.15:
        parts.append(f"整体图像质量<b>良好</b>——{bad_ratio:.1%} 的帧被标记为低质量（平均质量分 {overall_avg:.3f}）。")
    elif bad_ratio < 0.30:
        parts.append(f"整体图像质量<b>中等</b>——{bad_ratio:.1%} 的帧被标记为低质量（平均质量分 {overall_avg:.3f}），建议进一步排查。")
    else:
        parts.append(f"整体图像质量<b>较差</b>——{bad_ratio:.1%} 的帧被标记为低质量（平均质量分 {overall_avg:.3f}），需立即关注数据采集条件。")

    if top_issues:
        issue_str = "、".join(f"{k}（{v} 帧）" for k, v in top_issues)
        parts.append(f"主要质量问题：{issue_str}。")

    worst_topic = max(topics, key=lambda d: d.get("bad_quality_ratio", 0)) if topics else None
    if worst_topic and worst_topic.get("bad_quality_ratio", 0) > 0.1:
        cam = worst_topic.get("short", worst_topic.get("topic", ""))
        parts.append(f"质量最差摄像机：<code>{cam}</code>（{worst_topic['bad_quality_ratio']:.1%} 帧低质量）。")

    return "".join(f"<p style='margin:4px 0'>{p}</p>" for p in parts)


_TREND_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def _generate_quality_trend_svg(output_dir: Path) -> str:
    """Generate inline SVG showing quality score timeline per camera."""
    pred_path = output_dir / "yolo_predictions.json"
    if not pred_path.exists():
        return ""
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    preds = data.get("predictions") or data if isinstance(data, list) else data.get("predictions", [])
    if not preds:
        return ""

    by_topic: Dict[str, List[tuple]] = {}
    for p in preds:
        topic = p.get("topic", "")
        qs = p.get("quality_score")
        seq = p.get("frame_seq", 0)
        if qs is not None:
            short = _topic_short(topic)
            by_topic.setdefault(short, []).append((seq, qs))
    if not by_topic:
        return ""

    for v in by_topic.values():
        v.sort(key=lambda x: x[0])

    W, H = 680, 240
    PAD_L, PAD_R, PAD_T, PAD_B = 48, 20, 22, 52
    chart_w = W - PAD_L - PAD_R
    chart_h = H - PAD_T - PAD_B

    all_seqs = [s for pts in by_topic.values() for s, _ in pts]
    min_seq, max_seq = min(all_seqs), max(all_seqs)
    seq_range = max_seq - min_seq if max_seq > min_seq else 1

    def sx(s):
        return PAD_L + (s - min_seq) / seq_range * chart_w

    def sy(q):
        return PAD_T + (1.0 - max(0.0, min(1.0, q))) * chart_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'class="trend-svg" style="font-family:system-ui,sans-serif;font-size:10px">'
    ]
    # 绘图区背景
    lines.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{chart_w}" height="{chart_h}" '
        f'fill="#fff" stroke="#d0d7de" rx="4"/>'
    )

    # Y 轴网格与刻度
    for y_val in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = sy(y_val)
        lines.append(
            f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{PAD_L + chart_w:.1f}" y2="{yy:.1f}" '
            f'stroke="#eaeef2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{PAD_L - 8}" y="{yy + 4:.1f}" text-anchor="end" fill="#57606a" '
            f'font-size="10">{y_val:.1f}</text>'
        )

    # X 轴刻度（约 5 个）
    x_ticks: List[int] = []
    if seq_range <= 4:
        x_ticks = list(range(int(min_seq), int(max_seq) + 1))
    else:
        step = max(1, int(seq_range // 4))
        x_ticks = [int(min_seq)]
        v = int(min_seq)
        while v < max_seq:
            v += step
            if v < max_seq:
                x_ticks.append(v)
        x_ticks.append(int(max_seq))
    for xv in x_ticks:
        xx = sx(xv)
        lines.append(
            f'<line x1="{xx:.1f}" y1="{PAD_T}" x2="{xx:.1f}" y2="{PAD_T + chart_h:.1f}" '
            f'stroke="#eaeef2" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{xx:.1f}" y="{PAD_T + chart_h + 16}" text-anchor="middle" '
            f'fill="#57606a" font-size="10">{xv}</text>'
        )

    # 坐标轴
    lines.append(
        f'<line x1="{PAD_L}" y1="{PAD_T + chart_h:.1f}" x2="{PAD_L + chart_w:.1f}" '
        f'y2="{PAD_T + chart_h:.1f}" stroke="#8c959f" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + chart_h:.1f}" '
        f'stroke="#8c959f" stroke-width="1.2"/>'
    )

    # 各相机折线
    for i, (topic_short, pts) in enumerate(sorted(by_topic.items())):
        color = _TREND_COLORS[i % len(_TREND_COLORS)]
        n = len(pts)
        if n > 120:
            step = max(1, n // 120)
            pts = pts[::step]
        points = " ".join(f"{sx(s):.1f},{sy(q):.1f}" for s, q in pts)
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>'
        )

    # 图例
    legend_y = H - 18
    lx = PAD_L
    for i, topic_short in enumerate(sorted(by_topic.keys())):
        color = _TREND_COLORS[i % len(_TREND_COLORS)]
        lines.append(f'<rect x="{lx}" y="{legend_y - 9}" width="14" height="3" fill="{color}" rx="1"/>')
        lines.append(
            f'<text x="{lx + 18}" y="{legend_y}" fill="#1f2328" font-size="10">{topic_short}</text>'
        )
        lx += len(topic_short) * 6.5 + 28

    # 轴标题（中文）
    lines.append(
        f'<text x="{PAD_L + chart_w / 2:.1f}" y="{H - 4}" text-anchor="middle" '
        f'fill="#1f2328" font-size="11" font-weight="600">帧序号（sampled #）</text>'
    )
    lines.append(
        f'<text x="14" y="{PAD_T + chart_h / 2:.1f}" text-anchor="middle" fill="#1f2328" '
        f'font-size="11" font-weight="600" transform="rotate(-90,14,{PAD_T + chart_h / 2:.1f})">'
        f'质量分</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


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
    batch_failures: Optional[List[dict]] = None,
) -> Path:
    """Generate simplified quality_report.html."""
    seq_map: Dict[str, SequenceSummary] = {}
    if sequence_summaries:
        for ss in sequence_summaries:
            seq_map[ss.topic] = ss

    topics = []
    for ts in topic_summaries:
        d = ts.to_dict() if hasattr(ts, "to_dict") else dict(ts)
        d["short"] = _topic_short(d.get("topic", ""))
        ss = seq_map.get(d.get("topic", ""))
        d["duration_sec"] = f"{ss.duration_sec:.1f}s" if ss else None
        d["estimated_fps"] = f"{ss.estimated_fps:.1f}" if ss and ss.estimated_fps else None
        if ss:
            d["resolution_change_count"] = ss.resolution_change_count
            d["timestamp_jump_count"] = ss.timestamp_jump_count
        else:
            d["resolution_change_count"] = None
            d["timestamp_jump_count"] = None
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

    issue_counts: Dict[str, int] = {}
    for d in topics:
        for k, v in (d.get("quality_issue_counts") or {}).items():
            issue_counts[k] = issue_counts.get(k, 0) + v
    max_count = max(issue_counts.values()) if issue_counts else 1
    issue_chart = [
        {"label": k, "count": v, "pct": round(v / max_count * 100, 1)}
        for k, v in sorted(issue_counts.items(), key=lambda x: -x[1])
    ]

    conclusion = _generate_quality_conclusion(topics) if topics else ""
    trend_svg = _generate_quality_trend_svg(output_dir)

    html = _ENV.from_string(_QUALITY_TPL).render(
        subtitle=subtitle,
        topics=topics,
        worst=worst,
        bad_gallery_html=bad_gallery_html,
        dup_groups=dup_groups,
        issue_chart=issue_chart,
        conclusion=conclusion,
        trend_svg=trend_svg,
        batch_failures=batch_failures,
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

    device = "CPU"
    target_classes_str = ""
    if model_info:
        device = model_info.get("device", "CPU")
        tc = model_info.get("target_classes") or target_classes or []
        target_classes_str = ", ".join(tc) if tc else ""
    elif target_classes:
        target_classes_str = ", ".join(target_classes)
    if not target_classes_str and (output_dir / "metrics.json").exists():
        m = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        mi = m.get("model") or {}
        device = mi.get("device", device)
        tc = mi.get("target_classes") or []
        target_classes_str = ", ".join(tc) if tc else ""

    # YOLO 总体结论（中文），在 device/backend 确定后生成
    yolo_conclusion = ""
    if sampled > 0 or inferred > 0:
        _backend_label = backend if backend else "onnxruntime"
        _device_label = "GPU（TensorRT）" if "tensorrt" in _backend_label.lower() else (
            "GPU（CUDA）" if (device or "").lower() == "gpu" else "CPU"
        )
        parts_zh = []
        parts_zh.append(
            f"<p style='margin:4px 0'>共采样 <b>{sampled}</b> 帧，"
            f"推理后端：<b>{_backend_label}</b>，计算设备：<b>{_device_label}</b>。"
            f"成功完成推理 <b>{inferred}</b> 帧。</p>"
        )
        if skipped_quality > 0:
            parts_zh.append(
                f"<p style='margin:4px 0'>因质量门控跳过 <b>{skipped_quality}</b> 帧"
                f"（质量分低于阈值，未执行 YOLO 推理）。</p>"
            )
        if targets:
            total_det = sum(t.get("detected_count", 0) for t in targets)
            top_cls = sorted(targets, key=lambda t: -t.get("detected_count", 0))[:3]
            cls_str = "、".join(
                f"{t.get('label','')}（{t.get('detected_count',0)} 次）" for t in top_cls
            )
            parts_zh.append(
                f"<p style='margin:4px 0'>累计检测目标 <b>{total_det}</b> 个，Top 类别：{cls_str}。</p>"
            )
        if infer_failed > 0:
            parts_zh.append(
                f"<p style='margin:4px 0;color:#c0392b'>⚠ 推理失败 <b>{infer_failed}</b> 帧，"
                f"请检查模型兼容性。</p>"
            )
        else:
            parts_zh.append("<p style='margin:4px 0'>所有已推理帧均成功完成，未出现推理错误。</p>")
        yolo_conclusion = "".join(parts_zh)

    html = _ENV.from_string(_YOLO_TPL).render(
        model_name=model_name,
        model_format=model_format,
        input_size=input_size,
        backend=backend,
        device=device,
        target_classes_str=target_classes_str,
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
        yolo_conclusion=yolo_conclusion,
    )
    out = output_dir / "yolo_report.html"
    out.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {out} ({out.stat().st_size} bytes)")
    return out


_METRICS_FRAME_TO_STATS = {
    "sampled_frames": "sampled_frames",
    "infer_success_frames": "infer_success",
    "skipped_low_quality_frames": "skipped_low_quality",
    "skipped_depth_topic_frames": "skipped_depth_topic",
    "infer_failed_frames": "infer_failed",
    "quality_failed_frames": "quality_failed",
}


def _stats_from_metrics(metrics: dict) -> PipelineStats:
    stats = PipelineStats()
    fr = metrics.get("frames") or {}
    for json_key, attr in _METRICS_FRAME_TO_STATS.items():
        if json_key in fr:
            setattr(stats, attr, fr[json_key])
    sm = metrics.get("sampling") or {}
    if "mode" in sm:
        stats.sampling_mode = sm["mode"]
    if "computed_sample_every_n" in sm:
        stats.computed_sample_every_n = sm["computed_sample_every_n"]
    return stats


def regenerate_html_reports(output_dir: Path) -> None:
    """Regenerate HTML only from existing JSON in output_dir (no MCAP re-run)."""
    from app.report.sample_exporter import rebuild_detection_index

    rebuild_detection_index(output_dir)

    qr_path = output_dir / "quality_report.json"
    if qr_path.exists():
        qr = json.loads(qr_path.read_text(encoding="utf-8"))
        topics = list(qr.get("topics") or [])
        stats = PipelineStats()
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            stats = _stats_from_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))
        seq_summaries: Optional[List[SequenceSummary]] = None
        raw_seq = qr.get("sequence_analysis") or []
        if raw_seq:
            seq_summaries = []
            for sd in raw_seq:
                ss = SequenceSummary(topic=sd.get("topic", ""))
                ss.duration_sec = sd.get("duration_sec", 0.0)
                ss.estimated_fps = sd.get("estimated_fps", 0.0)
                ss.total_frames = sd.get("total_frames", 0)
                ss.resolution_change_count = int(sd.get("resolution_change_count", 0) or 0)
                ss.timestamp_jump_count = int(sd.get("timestamp_jump_count", 0) or 0)
                seq_summaries.append(ss)
        bf = qr.get("batch_failures") or []
        write_quality_html(
            output_dir, topics, sequence_summaries=seq_summaries,
            pipeline_stats=stats, batch_failures=bf,
        )

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        stats = _stats_from_metrics(m)
        perf = m.get("performance") or {}
        model_info = m.get("model") or {
            "name": "yolov8n", "format": "onnx", "input_size": [640, 640], "backend": "onnxruntime",
        }
        targets = m.get("target_analysis") or []
        write_yolo_html(
            output_dir,
            pipeline_stats=stats,
            model_info=model_info,
            perf=perf,
        )
