"""Tests for YOLO post-processing (FR-YOLO-004)."""
import numpy as np
import pytest

from app.yolo.postprocess import (
    Detection,
    _cxcywh_to_xyxy,
    _decode_yolov8,
    postprocess,
)
from app.yolo.preprocess import LetterboxMeta


class TestCxcywhToXyxy:
    def test_basic_conversion(self):
        boxes = np.array([[100, 100, 50, 50]], dtype=np.float32)
        xyxy = _cxcywh_to_xyxy(boxes)
        np.testing.assert_array_almost_equal(xyxy, [[75, 75, 125, 125]])

    def test_zero_size(self):
        boxes = np.array([[50, 50, 0, 0]], dtype=np.float32)
        xyxy = _cxcywh_to_xyxy(boxes)
        np.testing.assert_array_almost_equal(xyxy, [[50, 50, 50, 50]])


class TestDecodeYolov8:
    def test_shape(self):
        output = np.random.rand(1, 84, 10).astype(np.float32)
        boxes, scores, class_ids = _decode_yolov8(output)
        assert boxes.shape == (10, 4)
        assert scores.shape == (10,)
        assert class_ids.shape == (10,)


class TestPostprocess:
    def _make_meta(self):
        return LetterboxMeta(
            orig_w=640, orig_h=480, input_w=640, input_h=640,
            scale=1.0, pad_left=0.0, pad_top=80.0,
        )

    def test_empty_output(self):
        meta = self._make_meta()
        result = postprocess(
            np.array([]).reshape(1, 84, 0).astype(np.float32),
            meta, ["person"] * 80,
        )
        assert result == []

    def test_no_high_confidence(self):
        output = np.zeros((1, 84, 10), dtype=np.float32)
        meta = self._make_meta()
        result = postprocess(output, meta, ["person"] * 80, conf_threshold=0.5)
        assert result == []

    def test_detection_to_dict(self):
        det = Detection(label="car", class_id=2, confidence=0.85,
                        x1=100, y1=50, x2=300, y2=250)
        d = det.to_dict()
        assert d["label"] == "car"
        assert d["class_id"] == 2
        assert d["confidence"] == 0.85
        assert d["bbox"] == {"x1": 100, "y1": 50, "x2": 300, "y2": 250}

    def test_synthetic_detection(self):
        """Create a synthetic YOLOv8 output with one strong detection."""
        n_detections = 8400
        output = np.zeros((1, 84, n_detections), dtype=np.float32)
        output[0, 0, 0] = 320   # cx
        output[0, 1, 0] = 320   # cy
        output[0, 2, 0] = 100   # w
        output[0, 3, 0] = 100   # h
        output[0, 4, 0] = 0.95  # class 0 (person) score

        meta = LetterboxMeta(
            orig_w=640, orig_h=640, input_w=640, input_h=640,
            scale=1.0, pad_left=0.0, pad_top=0.0,
        )
        names = ["person"] + [f"cls_{i}" for i in range(1, 80)]
        result = postprocess(output, meta, names, conf_threshold=0.5)
        assert len(result) >= 1
        assert result[0].label == "person"
        assert result[0].confidence >= 0.5

    def test_min_box_side_filter(self):
        output = np.zeros((1, 84, 10), dtype=np.float32)
        output[0, 0, 0] = 320
        output[0, 1, 0] = 320
        output[0, 2, 0] = 10
        output[0, 3, 0] = 10
        output[0, 4, 0] = 0.95

        meta = LetterboxMeta(
            orig_w=640, orig_h=640, input_w=640, input_h=640,
            scale=1.0, pad_left=0.0, pad_top=0.0,
        )
        names = ["person"] + [f"cls_{i}" for i in range(1, 80)]
        result = postprocess(
            output, meta, names,
            conf_threshold=0.5,
            min_box_side_px=32,
        )
        assert result == []
