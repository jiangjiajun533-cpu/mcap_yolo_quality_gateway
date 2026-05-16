#!/usr/bin/env python3
"""
Generate a synthetic test MCAP file with CompressedImage messages.

Usage:
  python scripts/generate_test_mcap.py [--output test_data/sample.mcap] [--frames 100]

The generated file contains a single topic '/camera/front/image/compressed'
with JPEG-encoded synthetic images (colour gradients + noise) so the pipeline
can be exercised without real sensor data.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TOPICS = [
    "/camera/front/image/compressed",
    "/camera/left/image/compressed",
]
SCHEMA_NAME = "sensor_msgs/CompressedImage"
MESSAGE_ENCODING = "ros1"
WIDTH, HEIGHT = 640, 480
FPS = 30

_COMPRESSED_IMAGE_MSG = """\
std_msgs/Header header
string format
uint8[] data

================================================================================
MSG: std_msgs/Header
uint32 seq
time stamp
string frame_id
"""


def _make_synthetic_image(seq: int, variant: int = 0) -> np.ndarray:
    """Generate a BGR image with a moving gradient and some noise.
    
    variant shifts colour channels to differentiate cameras.
    Occasionally injects quality issues (dark, bright, blurry) for testing.
    """
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    offset = (seq * 5 + variant * 100) % WIDTH
    for c in range(3):
        grad = np.linspace(0, 255, WIDTH, dtype=np.uint8)
        grad = np.roll(grad, offset + c * 80)
        img[:, :, c] = grad[np.newaxis, :]
    noise = np.random.randint(0, 20, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Inject quality issues for ~15% of frames
    if seq % 7 == 0:
        img = (img * 0.15).astype(np.uint8)  # too dark
    elif seq % 11 == 0:
        img = cv2.GaussianBlur(img, (31, 31), 10)  # blurry
    elif seq % 13 == 0:
        img[:] = 240  # near-solid bright

    return img


def _encode_ros1_compressed_image(img: np.ndarray, stamp_ns: int, frame_id: str = "front_camera") -> bytes:
    """
    Build a minimal ros1 serialised sensor_msgs/CompressedImage.

    Layout:
      header.seq (uint32) | header.stamp.sec (uint32) | header.stamp.nsec (uint32)
      header.frame_id (uint32 len + utf8 bytes)
      format (uint32 len + utf8 bytes)
      data (uint32 len + raw bytes)
    """
    _, jpeg_buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = jpeg_buf.tobytes()

    sec = int(stamp_ns // 1_000_000_000)
    nsec = int(stamp_ns % 1_000_000_000)
    seq_num = 0

    frame_id_bytes = frame_id.encode("utf-8")
    fmt_str = "jpeg"
    fmt_bytes = fmt_str.encode("utf-8")

    buf = bytearray()
    buf += struct.pack("<I", seq_num)
    buf += struct.pack("<II", sec, nsec)
    buf += struct.pack("<I", len(frame_id_bytes)) + frame_id_bytes
    buf += struct.pack("<I", len(fmt_bytes)) + fmt_bytes
    buf += struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes
    return bytes(buf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a test MCAP file")
    parser.add_argument("--output", type=str, default="test_data/sample.mcap")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mcap.writer import Writer
    except ImportError:
        print("ERROR: mcap package is required. Install with: pip install mcap", file=sys.stderr)
        sys.exit(1)

    interval_ns = int(1e9 / args.fps)
    base_time_ns = 1_710_000_000_000_000_000

    print(f"Generating {args.frames} frames x {len(TOPICS)} topics at {args.fps}fps to {out_path} ...")
    t0 = time.perf_counter()

    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name=SCHEMA_NAME,
            encoding="ros1msg",
            data=_COMPRESSED_IMAGE_MSG.encode("utf-8"),
        )

        channel_ids = {}
        for topic in TOPICS:
            channel_ids[topic] = writer.register_channel(
                topic=topic,
                message_encoding=MESSAGE_ENCODING,
                schema_id=schema_id,
            )

        for i in range(args.frames):
            stamp_ns = base_time_ns + i * interval_ns
            for ti, topic in enumerate(TOPICS):
                img = _make_synthetic_image(i, variant=ti)
                frame_id = topic.split("/")[2] + "_camera"
                msg_data = _encode_ros1_compressed_image(img, stamp_ns, frame_id=frame_id)

                writer.add_message(
                    channel_id=channel_ids[topic],
                    log_time=stamp_ns,
                    data=msg_data,
                    publish_time=stamp_ns,
                )

        writer.finish()

    elapsed = time.perf_counter() - t0
    size_kb = out_path.stat().st_size / 1024
    print(f"Done. {args.frames} frames, {size_kb:.0f} KB, {elapsed:.2f}s")


if __name__ == "__main__":
    main()
