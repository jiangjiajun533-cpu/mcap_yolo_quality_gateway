#!/usr/bin/env python3
"""
Check whether this machine can run GPU inference (CUDA / TensorRT).

Does NOT install CUDA or drivers — only reports status and next steps.
Usage (from repo root):
  python scripts/verify_gpu_env.py
  python scripts/verify_gpu_env.py --model models/yolov8n.onnx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_nvidia_smi() -> bool:
    _section("NVIDIA driver")
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("FAIL: nvidia-smi not found (no NVIDIA driver or not in PATH)")
        return False
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=15, check=False
        )
        line = next(
            (ln for ln in out.stdout.splitlines() if "Driver Version" in ln),
            out.stdout.splitlines()[2] if out.stdout else "",
        )
        print("OK:", line.strip() or "nvidia-smi succeeded")
        return True
    except Exception as exc:
        print(f"FAIL: nvidia-smi error: {exc}")
        return False


def check_onnxruntime(model: Path | None) -> bool:
    _section("ONNX Runtime (CUDA EP)")
    try:
        import onnxruntime as ort
    except ImportError:
        print("FAIL: onnxruntime not installed. Run: pip install -r requirements.txt")
        return False

    print(f"onnxruntime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print("Available providers:", providers)

    has_cuda = "CUDAExecutionProvider" in providers
    if not has_cuda:
        print(
            "WARN: CUDAExecutionProvider not listed. "
            "Install: pip install onnxruntime-gpu  (and CUDA 12.x + cuDNN 9.x on PATH)"
        )
        return False

    if model is None or not model.is_file():
        print(
            "SKIP: model load test (pass --model models/yolov8n.onnx to test session)"
        )
        return True

    try:
        sess = ort.InferenceSession(
            str(model),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = sess.get_providers()
        print(f"Model: {model}")
        print("Session providers:", active)
        if active and active[0] == "CUDAExecutionProvider":
            print("OK: CUDA EP is active for this model")
            return True
        print("WARN: Session did not use CUDA first (may have fallen back to CPU)")
        return False
    except Exception as exc:
        print(f"FAIL: could not create CUDA session: {exc}")
        print(
            "Typical fix (Windows): install CUDA Toolkit 12.x, cuDNN 9.x, "
            "add CUDA\\v12.x\\bin to PATH, reopen terminal."
        )
        return False


def check_tensorrt() -> bool:
    _section("TensorRT (bonus backend)")
    from app.yolo.trt_runner import TRT_AVAILABLE

    if not TRT_AVAILABLE:
        print(
            "SKIP: tensorrt / pycuda not installed. "
            "TRT is optional; build .engine locally (see README §26.1)."
        )
        return False
    print("OK: tensorrt and pycuda importable")
    print("Build engine example:")
    print(
        "  trtexec --onnx=models/yolov8n.onnx --saveEngine=models/yolov8n.engine --fp16"
    )
    print("Run inference:")
    print(
        "  python scripts/run_mcap_yolo_inference.py "
        "--backend tensorrt --model models/yolov8n.engine ..."
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GPU / TRT environment")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n.onnx",
        help="ONNX path for CUDA session test (optional)",
    )
    args = parser.parse_args()
    model = Path(args.model) if args.model else None

    print("GPU environment check (nothing will be installed automatically)")
    ok_drv = check_nvidia_smi()
    ok_cuda = check_onnxruntime(model)
    ok_trt = check_tensorrt()

    _section("Summary")
    print(f"  Driver (nvidia-smi):     {'OK' if ok_drv else 'FAIL'}")
    print(f"  ONNX CUDA EP:            {'OK' if ok_cuda else 'FAIL/SKIP'}")
    print(f"  TensorRT (optional):     {'OK' if ok_trt else 'not installed'}")
    print()
    print("For this project you do NOT upload CUDA/cuDNN/TRT engines to Git.")
    print(
        "README documents how reviewers install deps and run --device gpu / --backend tensorrt."
    )
    if ok_drv and ok_cuda:
        print("\nReady: python scripts/run_mcap_yolo_inference.py ... --device gpu")
        sys.exit(0)
    if ok_drv:
        print("\nCPU fallback works: use --device cpu (assignment default).")
    sys.exit(1 if ok_drv and not ok_cuda else 0)


if __name__ == "__main__":
    main()
