"""Tests for ROS image decoder (FR-IMG-001, FR-IMG-002)."""
import struct

import cv2
import numpy as np
import pytest

from app.mcap_io.ros_image_decoder import decode_ros_message
from app.core.errors import DecodeError, UnsupportedEncodingError


def _make_compressed_ros_msg(img: np.ndarray, fmt: str = "jpeg") -> object:
    """Create a minimal mock ROS CompressedImage-like object."""
    if fmt == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elif fmt == "png":
        _, buf = cv2.imencode(".png", img)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    class MockStamp:
        sec = 1710000000
        nanosec = 123456789

    class MockHeader:
        stamp = MockStamp()
        frame_id = "test_camera"

    class MockMsg:
        header = MockHeader()
        format = fmt
        data = buf.tobytes()

    return MockMsg()


class TestDecodeCompressed:
    def test_jpeg_decode(self):
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        ros_msg = _make_compressed_ros_msg(img, "jpeg")
        frame = decode_ros_message(
            ros_msg=ros_msg,
            schema_name="sensor_msgs/CompressedImage",
            mcap_file="test.mcap",
            topic="/cam/compressed",
            frame_seq=0,
            log_time_ns=1_000_000_000,
            publish_time_ns=1_000_000_000,
        )
        assert frame.image is not None
        assert frame.image.shape[0] == 480
        assert frame.image.shape[1] == 640
        assert frame.encoding == "jpeg"

    def test_png_decode(self):
        img = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        ros_msg = _make_compressed_ros_msg(img, "png")
        frame = decode_ros_message(
            ros_msg=ros_msg,
            schema_name="sensor_msgs/msg/CompressedImage",
            mcap_file="test.mcap",
            topic="/cam/compressed",
            frame_seq=1,
            log_time_ns=2_000_000_000,
            publish_time_ns=2_000_000_000,
        )
        assert frame.image is not None
        assert frame.width == 320
        assert frame.height == 240

    def test_empty_data_does_not_crash(self):
        class MockStamp:
            sec = 0
            nanosec = 0
        class MockHeader:
            stamp = MockStamp()
            frame_id = ""
        class MockMsg:
            header = MockHeader()
            format = "jpeg"
            data = b""

        with pytest.raises(DecodeError):
            decode_ros_message(
                ros_msg=MockMsg(),
                schema_name="sensor_msgs/CompressedImage",
                mcap_file="test.mcap",
                topic="/cam",
                frame_seq=0,
                log_time_ns=0,
                publish_time_ns=0,
            )

    def test_corrupted_data_does_not_crash(self):
        class MockStamp:
            sec = 0
            nanosec = 0
        class MockHeader:
            stamp = MockStamp()
            frame_id = ""
        class MockMsg:
            header = MockHeader()
            format = "jpeg"
            data = b"not_a_real_jpeg_file_content"

        with pytest.raises(DecodeError):
            decode_ros_message(
                ros_msg=MockMsg(),
                schema_name="sensor_msgs/CompressedImage",
                mcap_file="test.mcap",
                topic="/cam",
                frame_seq=0,
                log_time_ns=0,
                publish_time_ns=0,
            )
