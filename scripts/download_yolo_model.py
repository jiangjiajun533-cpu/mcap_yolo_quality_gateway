#!/usr/bin/env python3
"""
Download YOLOv8n weights and produce ONNX + COCO labels (FR-YOLO-002).

Ultralytics assets releases host ``yolov8n.pt`` but not pre-built ``yolov8n.onnx``.
This script downloads the official ``.pt`` and exports ONNX when ``ultralytics`` is installed.

Usage:
  python scripts/download_yolo_model.py [--output-dir models]

Manual export (if ultralytics is not installed):
  pip install ultralytics
  yolo export model=models/yolov8n.pt format=onnx imgsz=640 opset=12 simplify=True
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Official Ultralytics release assets (verified: .pt returns HTTP 200; .onnx is not published)
YOLOV8N_PT_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
)
RELEASE_PAGE = "https://github.com/ultralytics/assets/releases"

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"Saved to {dest} ({dest.stat().st_size} bytes)")


def _export_onnx(pt_path: Path, onnx_path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run:\n"
            "  pip install ultralytics\n"
            f"  yolo export model={pt_path} format=onnx imgsz=640 opset=12 simplify=True\n"
            f"Then move the exported file to {onnx_path}"
        ) from exc

    print(f"Exporting ONNX from {pt_path} ...")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        if onnx_path.exists():
            onnx_path.unlink()
        exported_path.replace(onnx_path)
    print(f"ONNX ready: {onnx_path} ({onnx_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLOv8n and export ONNX")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "yolov8n.pt"
    onnx_path = out_dir / "yolov8n.onnx"
    labels_path = out_dir / "coco_classes.txt"

    if not pt_path.exists():
        _download(YOLOV8N_PT_URL, pt_path)
    else:
        print(f"Weights already exist: {pt_path} ({pt_path.stat().st_size} bytes)")

    if not onnx_path.exists():
        try:
            _export_onnx(pt_path, onnx_path)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            print(f"\nRelease index: {RELEASE_PAGE}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"ONNX already exists: {onnx_path} ({onnx_path.stat().st_size} bytes)")

    labels_path.write_text("\n".join(COCO_CLASSES) + "\n", encoding="utf-8")
    print(f"Wrote {len(COCO_CLASSES)} class labels to {labels_path}")


if __name__ == "__main__":
    main()
