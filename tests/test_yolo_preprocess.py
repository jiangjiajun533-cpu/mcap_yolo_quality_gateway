"""Tests for YOLO preprocessing (FR-YOLO-003)."""

import numpy as np
import pytest

from app.yolo.preprocess import letterbox, preprocess, unscale_coords, LetterboxMeta


class TestLetterbox:
    def test_square_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, meta = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert meta.scale == 1.0
        assert meta.pad_left == 0.0
        assert meta.pad_top == 0.0

    def test_wide_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, meta = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert meta.scale == 1.0
        assert meta.pad_top > 0
        assert meta.pad_left == 0.0

    def test_tall_image(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        padded, meta = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert meta.pad_left > 0
        assert meta.pad_top == 0.0

    def test_small_image(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, meta = letterbox(img, 640)
        assert padded.shape == (640, 640, 3)
        assert meta.scale > 1.0

    def test_grayscale_promoted(self):
        img = np.zeros((480, 640), dtype=np.uint8)
        padded, meta = letterbox(img, 640)
        assert padded.ndim == 3
        assert padded.shape[2] == 3


class TestPreprocess:
    def test_output_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, meta = preprocess(img, 640)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_values_normalized(self):
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        tensor, _ = preprocess(img, 640)
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0


class TestUnscaleCoords:
    def test_identity(self):
        meta = LetterboxMeta(
            orig_w=640,
            orig_h=640,
            input_w=640,
            input_h=640,
            scale=1.0,
            pad_left=0.0,
            pad_top=0.0,
        )
        x1, y1, x2, y2 = unscale_coords(100, 200, 300, 400, meta)
        assert (x1, y1, x2, y2) == (100, 200, 300, 400)

    def test_with_padding(self):
        meta = LetterboxMeta(
            orig_w=640,
            orig_h=480,
            input_w=640,
            input_h=640,
            scale=1.0,
            pad_left=0.0,
            pad_top=80.0,
        )
        x1, y1, x2, y2 = unscale_coords(100, 180, 300, 380, meta)
        assert x1 == 100
        assert y1 == 100
        assert x2 == 300
        assert y2 == 300

    def test_clips_negative(self):
        meta = LetterboxMeta(
            orig_w=640,
            orig_h=480,
            input_w=640,
            input_h=640,
            scale=1.0,
            pad_left=0.0,
            pad_top=0.0,
        )
        x1, y1, x2, y2 = unscale_coords(-10, -20, 100, 200, meta)
        assert x1 == 0
        assert y1 == 0

    def test_clips_overflow(self):
        meta = LetterboxMeta(
            orig_w=640,
            orig_h=480,
            input_w=640,
            input_h=640,
            scale=1.0,
            pad_left=0.0,
            pad_top=0.0,
        )
        x1, y1, x2, y2 = unscale_coords(600, 400, 700, 500, meta)
        assert x2 == 640
        assert y2 == 480
