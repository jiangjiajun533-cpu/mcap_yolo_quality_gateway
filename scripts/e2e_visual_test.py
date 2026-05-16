#!/usr/bin/env python3
"""
End-to-end smoke test: CLI pipeline → reports → Dashboard APIs → draw_frame.

Usage (from project root):
  python scripts/e2e_visual_test.py
  python scripts/e2e_visual_test.py --api-base http://127.0.0.1:8000
  python scripts/e2e_visual_test.py --skip-cli   # only test API if outputs exist
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MCAP = PROJECT_ROOT / "test_data" / "sample.mcap"
MODEL = PROJECT_ROOT / "models" / "yolov8n.onnx"
LABELS = PROJECT_ROOT / "models" / "coco_classes.txt"
OUTPUT = PROJECT_ROOT / "outputs" / "e2e_test"


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def http_get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def http_post_json(url: str, body: dict) -> tuple[int, bytes]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def run_cli(output_dir: Path) -> None:
    step("1. CLI: YOLO inference pipeline")
    if not MCAP.is_file():
        fail(f"MCAP missing: {MCAP}")
    if not MODEL.is_file():
        fail(f"ONNX model missing: {MODEL} (run scripts/download_yolo_model.py)")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_mcap_yolo_inference.py"),
        "--mcap", str(MCAP),
        "--auto-detect-topics", "true",
        "--model", str(MODEL),
        "--labels", str(LABELS),
        "--target-classes", "person,car,truck,bus",
        "--sample-every-n", "5",
        "--quality-threshold", "0.6",
        "--max-frames", "50",
        "--output-dir", str(output_dir),
    ]
    print("  ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        fail("CLI pipeline exited non-zero")

    required = [
        "yolo_predictions.json",
        "quality_report.json",
        "metrics.json",
        "yolo_report.html",
    ]
    for name in required:
        p = output_dir / name
        if not p.is_file():
            fail(f"Missing output: {p}")
    ok(f"CLI outputs in {output_dir}")


def verify_reports(output_dir: Path) -> dict:
    step("2. Reports: JSON structure")
    preds = json.loads((output_dir / "yolo_predictions.json").read_text(encoding="utf-8"))
    pred_list = preds.get("predictions") or []
    if not pred_list:
        fail("yolo_predictions.json has no predictions")
    inferred = [p for p in pred_list if p.get("action") == "inferred" and p.get("objects")]
    if not inferred:
        fail("No inferred frames with objects — cannot test draw_frame")
    sample = inferred[0]
    ok(f"{len(pred_list)} predictions, {len(inferred)} with detections")

    qr = json.loads((output_dir / "quality_report.json").read_text(encoding="utf-8"))
    if "topics" not in qr:
        fail("quality_report.json missing topics")
    if qr.get("duplicate_analysis") is not None:
        ok("duplicate_analysis present in quality_report.json")
    else:
        print("  [WARN] duplicate_analysis absent (no dup groups in sample?)")

    return sample


def test_api(api_base: str, output_dir: Path, sample: dict) -> None:
    step("3. API: health + pipeline review")
    try:
        status, _ = http_get(f"{api_base}/health")
    except urllib.error.URLError as e:
        fail(f"API not reachable at {api_base}: {e}")
    if status != 200:
        fail(f"/health returned {status}")
    ok("/health")

    rel = output_dir.relative_to(PROJECT_ROOT).as_posix()
    status, body = http_get(f"{api_base}/pipeline/review_index?output_dir={rel}")
    if status != 200:
        fail(f"/pipeline/review_index returned {status}")
    data = json.loads(body.decode())
    if not data.get("predictions"):
        fail("review_index returned empty predictions")
    ok(f"review_index: {len(data['predictions'])} frames")

    step("4. API: frame decode (GET /mcap/frame)")
    mcap_rel = "test_data/sample.mcap"
    topic = sample["topic"]
    ts = sample["timestamp_ns"]
    raw_idx = sample.get("raw_frame_idx", 0)
    url = (
        f"{api_base}/mcap/frame?mcap_path={urllib.parse.quote(mcap_rel)}"
        f"&topic={urllib.parse.quote(topic)}"
        f"&timestamp_ns={ts}&raw_frame_idx={raw_idx}"
    )
    status, img = http_get(url)
    if status != 200 or len(img) < 1000:
        fail(f"/mcap/frame returned {status}, {len(img)} bytes")
    ok(f"/mcap/frame JPEG {len(img)} bytes")

    step("5. API: draw_frame with pipeline boxes (POST /mcap/draw_frame)")
    body = {
        "mcap_path": mcap_rel,
        "topic": topic,
        "timestamp_ns": ts,
        "raw_frame_idx": raw_idx,
        "frame_seq": sample.get("frame_seq", 0),
        "objects": sample["objects"],
    }
    status, annotated = http_post_json(f"{api_base}/mcap/draw_frame", body)
    if status != 200 or len(annotated) < 1000:
        fail(f"/mcap/draw_frame returned {status}, {len(annotated)} bytes")
    ok(f"/mcap/draw_frame JPEG {len(annotated)} bytes")

    step("6. Static: metrics.json via /outputs")
    metrics_url = f"{api_base}/outputs/{rel.replace('outputs/', '')}/metrics.json"
    status, mbody = http_get(metrics_url)
    if status != 200:
        fail(f"metrics.json static URL returned {status}: {metrics_url}")
    ok("metrics.json accessible")


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E pipeline + visualization test")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="Running uvicorn base URL")
    parser.add_argument("--output-dir", default=str(OUTPUT), help="CLI output directory")
    parser.add_argument("--skip-cli", action="store_true", help="Skip CLI run, use existing outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    if not args.skip_cli:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_cli(output_dir)
    elif not (output_dir / "yolo_predictions.json").is_file():
        fail(f"--skip-cli but no predictions at {output_dir}")

    sample = verify_reports(output_dir)
    test_api(args.api_base.rstrip("/"), output_dir, sample)

    print("\n=== ALL PASSED ===")
    print(f"Dashboard: {args.api_base.rstrip('/')}/")
    print(f"  MCAP path: test_data/sample.mcap")
    print(f"  Results:   {output_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print("  Then: Load Results → Head RGB → Detect tab → Show Boxes")


if __name__ == "__main__":
    main()
