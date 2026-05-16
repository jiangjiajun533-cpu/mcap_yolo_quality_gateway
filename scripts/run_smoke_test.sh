#!/usr/bin/env bash
# Smoke test: generate a test MCAP, run quality scan, optionally run YOLO.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Step 1: Generate test MCAP ==="
python scripts/generate_test_mcap.py --output test_data/sample.mcap --frames 50 --fps 30

echo ""
echo "=== Step 2: Quality scan ==="
python scripts/run_mcap_quality_scan.py \
  --mcap test_data/sample.mcap \
  --auto-detect-topics true \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --output-dir outputs

echo ""
echo "=== Step 3: Check quality outputs ==="
for f in outputs/mcap_summary.json outputs/quality_report.json outputs/quality_report.html outputs/quality_report.md outputs/metrics.json; do
  if [ -f "$f" ]; then
    echo "  OK  $f ($(wc -c < "$f") bytes)"
  else
    echo "  MISSING  $f"
    exit 1
  fi
done

if [ -f "models/yolov8n.onnx" ]; then
  echo ""
  echo "=== Step 4: YOLO inference ==="
  python scripts/run_mcap_yolo_inference.py \
    --mcap test_data/sample.mcap \
    --model models/yolov8n.onnx \
    --labels models/coco_classes.txt \
    --target-classes person,car,truck,bus \
    --sample-every-n 5 \
    --quality-threshold 0.6 \
    --output-dir outputs

  echo ""
  echo "=== Step 5: Check YOLO outputs ==="
  for f in outputs/yolo_predictions.json outputs/yolo_report.html outputs/yolo_report.md; do
    if [ -f "$f" ]; then
      echo "  OK  $f ($(wc -c < "$f") bytes)"
    else
      echo "  MISSING  $f"
      exit 1
    fi
  done
else
  echo ""
  echo "=== Step 4: YOLO skipped (no model at models/yolov8n.onnx) ==="
  echo "  Run: python scripts/download_yolo_model.py"
fi

echo ""
echo "=== Smoke test PASSED ==="
