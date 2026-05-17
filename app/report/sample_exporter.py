"""
Export bad-quality and detection sample images (FR-REPORT-004, FR-REPORT-005).

Outputs:
  bad_samples/
    <topic>_<seq>_<tag>.jpg
    index.json
  detection_samples/
    <topic>_<seq>_<classes>.jpg
    index.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.core.logging import get_logger
from app.yolo.pipeline import InferenceRecord
from app.yolo.postprocess import Detection
from app.yolo.visualizer import draw_detections

logger = get_logger("report.sample_exporter")


def _topic_short(topic: str) -> str:
    """Human-readable camera label (aligned with html_report._topic_short)."""
    if "realsense_head" in topic:
        return "head_depth" if "depth" in topic else "head_rgb"
    if "realsense_up" in topic:
        return "up_depth" if "depth" in topic else "up_rgb"
    if "right_wrist" in topic:
        return "right_wrist"
    if "left_wrist" in topic:
        return "left_wrist"
    parts = topic.strip("/").split("/")
    # /camera/front/image/compressed → front
    if len(parts) >= 2 and parts[0] == "camera":
        return parts[1]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "cam")


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def _pick_spread_records(
    records: List[InferenceRecord],
    max_samples: int,
) -> List[InferenceRecord]:
    """
    Pick up to ``max_samples`` with even spacing per topic (by frame_seq).

    Avoids exporting only consecutive MCAP-order slices for one camera.
    """
    if len(records) <= max_samples:
        return sorted(records, key=lambda r: (r.topic, r.frame_seq))

    by_topic: Dict[str, List[InferenceRecord]] = {}
    for r in records:
        by_topic.setdefault(r.topic, []).append(r)
    for lst in by_topic.values():
        lst.sort(key=lambda r: r.frame_seq)

    topics = sorted(by_topic.keys())
    base = max(1, max_samples // len(topics))
    picked: List[InferenceRecord] = []
    picked_ids: set[int] = set()

    for topic in topics:
        lst = by_topic[topic]
        if len(lst) <= base:
            for r in lst:
                if id(r) not in picked_ids:
                    picked.append(r)
                    picked_ids.add(id(r))
        else:
            step = len(lst) / base
            for i in range(base):
                r = lst[min(int(i * step + step / 2), len(lst) - 1)]
                if id(r) not in picked_ids:
                    picked.append(r)
                    picked_ids.add(id(r))

    if len(picked) < max_samples:
        rest = sorted(
            [r for r in records if id(r) not in picked_ids],
            key=lambda r: (r.topic, r.frame_seq),
        )
        for r in rest:
            if len(picked) >= max_samples:
                break
            picked.append(r)
            picked_ids.add(id(r))

    picked.sort(key=lambda r: (r.topic, r.frame_seq))
    return picked[:max_samples]


# ── FR-REPORT-004: bad_samples/ ──────────────────────────────────────────


def export_bad_samples(
    output_dir: Path,
    records: List[InferenceRecord],
    images: dict[int, np.ndarray],
    max_samples: int = 200,
) -> Path:
    """
    Export the worst quality frames as JPEG thumbnails.

    ``images`` maps ``id(record)`` → BGR ndarray for records
    that were flagged as bad quality.
    Returns the path to the bad_samples directory.
    """
    bad_dir = output_dir / "bad_samples"
    bad_dir.mkdir(parents=True, exist_ok=True)

    bad_records = [r for r in records if r.is_bad_quality and id(r) in images]
    bad_records.sort(key=lambda r: r.quality_score)
    bad_records = bad_records[:max_samples]

    index_entries: list[dict] = []
    for r in bad_records:
        img = images.get(id(r))
        if img is None:
            continue

        tag = r.quality_tags[0] if r.quality_tags else "bad"
        short_topic = _safe_filename(_topic_short(r.topic))
        fname = f"{short_topic}_{r.frame_seq:06d}_{_safe_filename(tag)}.jpg"
        fpath = bad_dir / fname

        thumb = _make_thumbnail(img, max_side=640)
        cv2.imwrite(str(fpath), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        index_entries.append(
            {
                "file": fname,
                "mcap_file": r.mcap_file,
                "topic": r.topic,
                "frame_seq": r.frame_seq,
                "raw_frame_idx": r.raw_frame_idx,
                "timestamp_ns": r.timestamp_ns,
                "quality_score": round(r.quality_score, 4),
                "quality_tags": r.quality_tags,
            }
        )

    _write_index(bad_dir / "index.json", index_entries)
    logger.info(f"Exported {len(index_entries)} bad samples to {bad_dir}")
    return bad_dir


# ── FR-REPORT-005: detection_samples/ ─────────────────────────────────────


def export_detection_samples(
    output_dir: Path,
    records: List[InferenceRecord],
    images: dict[int, np.ndarray],
    max_samples: int = 200,
    min_confidence: float = 0.0,
) -> Path:
    """
    Export frames with YOLO detections, drawn with bounding boxes.

    ``images`` maps ``id(record)`` → BGR ndarray for inferred records.
    Returns the path to the detection_samples directory.
    """
    det_dir = output_dir / "detection_samples"
    det_dir.mkdir(parents=True, exist_ok=True)

    det_records = [
        r
        for r in records
        if r.action == "inferred"
        and r.objects
        and id(r) in images
        and max(d.confidence for d in r.objects) >= min_confidence
    ]
    det_records = _pick_spread_records(det_records, max_samples)

    index_entries: list[dict] = []
    for r in det_records:
        img = images.get(id(r))
        if img is None:
            continue

        labels = sorted({d.label for d in r.objects})
        label_str = "_".join(labels[:3])
        short_topic = _safe_filename(_topic_short(r.topic))
        fname = f"{short_topic}_{r.frame_seq:06d}_{_safe_filename(label_str)}.jpg"
        fpath = det_dir / fname

        annotated = draw_detections(img, r.objects)
        thumb = _make_thumbnail(annotated, max_side=960)
        cv2.imwrite(str(fpath), thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])

        index_entries.append(
            {
                "file": fname,
                "mcap_file": r.mcap_file,
                "topic": r.topic,
                "frame_seq": r.frame_seq,
                "raw_frame_idx": r.raw_frame_idx,
                "timestamp_ns": r.timestamp_ns,
                "quality_score": round(r.quality_score, 4),
                "objects": [d.to_dict() for d in r.objects],
            }
        )

    _write_index(det_dir / "index.json", index_entries)
    logger.info(f"Exported {len(index_entries)} detection samples to {det_dir}")
    return det_dir


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_thumbnail(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _write_index(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": entries}, f, indent=2, ensure_ascii=False)


def _detection_fname(topic: str, frame_seq: int, objects: list) -> str:
    labels = sorted({d["label"] if isinstance(d, dict) else d.label for d in objects})
    label_str = "_".join(labels[:3])
    short_topic = _safe_filename(_topic_short(topic))
    return f"{short_topic}_{frame_seq:06d}_{_safe_filename(label_str)}.jpg"


def rebuild_detection_index(output_dir: Path) -> int:
    """
    Rebuild detection_samples/index.json from yolo_predictions.json + on-disk JPGs.

    Use when index.json was cleared but exported images remain.
    """
    det_dir = output_dir / "detection_samples"
    if not det_dir.is_dir():
        return 0

    entries: list[dict] = []
    pred_path = output_dir / "yolo_predictions.json"
    if pred_path.exists():
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        preds = data.get("predictions") or []
        for p in preds:
            if p.get("action") != "inferred" or not p.get("objects"):
                continue
            fname = _detection_fname(
                p["topic"], int(p.get("frame_seq", 0)), p["objects"]
            )
            if not (det_dir / fname).is_file():
                continue
            entries.append(
                {
                    "file": fname,
                    "mcap_file": p.get("mcap_file", ""),
                    "topic": p.get("topic", ""),
                    "frame_seq": p.get("frame_seq"),
                    "raw_frame_idx": p.get("raw_frame_idx"),
                    "timestamp_ns": p.get("timestamp_ns"),
                    "quality_score": p.get("quality_score"),
                    "objects": p.get("objects") or [],
                }
            )

    if not entries:
        pat = re.compile(r"^(.+)_(\d{6})_(.+)\.jpg$", re.IGNORECASE)
        for fpath in sorted(det_dir.glob("*.jpg")):
            m = pat.match(fpath.name)
            if not m:
                continue
            short_topic, seq_s, _labels = m.groups()
            entries.append(
                {
                    "file": fpath.name,
                    "mcap_file": "sample.mcap",
                    "topic": short_topic,
                    "frame_seq": int(seq_s),
                    "raw_frame_idx": None,
                    "timestamp_ns": None,
                    "quality_score": None,
                    "objects": [],
                }
            )

    _write_index(det_dir / "index.json", entries)
    logger.info(f"Rebuilt detection index: {len(entries)} entries")
    return len(entries)
