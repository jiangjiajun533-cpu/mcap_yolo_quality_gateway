#!/usr/bin/env bash
# Ubuntu end-to-end verification: unit tests + Docker smoke + JSON field checks.
# Usage (from repo root on Ubuntu):
#   chmod +x scripts/ubuntu_verify.sh
#   ./scripts/ubuntu_verify.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] pytest ==="
python3 -m pytest tests/ -q

echo "=== [2/4] release port 8088 (if occupied) ==="
if command -v fuser >/dev/null 2>&1; then
  fuser -k 8088/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  pid="$(lsof -ti:8088 2>/dev/null || true)"
  if [ -n "${pid}" ]; then kill "${pid}" 2>/dev/null || true; fi
fi
sleep 1

echo "=== [3/4] docker compose smoke (profile test) ==="
docker compose --profile test up --build --abort-on-container-exit

SMOKE_DIR="${ROOT}/outputs/smoke_test"
echo "=== [4/4] validate smoke outputs ==="
for f in \
  "${SMOKE_DIR}/yolo_predictions.json" \
  "${SMOKE_DIR}/quality_report.json" \
  "${SMOKE_DIR}/metrics.json" \
  "${SMOKE_DIR}/yolo_report.html"
do
  test -f "${f}" || { echo "Missing: ${f}"; exit 1; }
done

python3 - <<'PY'
import json
from pathlib import Path

root = Path("outputs/smoke_test")
pred = json.loads((root / "yolo_predictions.json").read_text(encoding="utf-8"))
rows = pred.get("predictions") or []
assert rows, "no predictions"
r0 = rows[0]
for key in ("log_time_ns", "ros_stamp_ns", "timestamp_source", "timestamp_ns"):
    assert key in r0, f"missing {key} in yolo_predictions[0]"

qr = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
worst = qr.get("worst_frames") or {}
if worst:
    w0 = next(iter(worst.values()))[0]
    assert "log_time_ns" in w0 and "brightness_mean" in w0, "worst_frames missing FR fields"

print("OK: smoke outputs and FR-IMG-003 timestamp fields present")
PY

echo ""
echo "All checks passed. Optional: start API with"
echo "  docker compose up --build"
echo "  open http://127.0.0.1:8088/docs"
