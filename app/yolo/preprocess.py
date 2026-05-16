"""
YOLO preprocessing: letterbox resize + normalize + HWC→CHW (FR-YOLO-003).

Key contract: returns both the model input tensor AND the LetterboxMeta
needed to map bbox coordinates back to the original image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LetterboxMeta:
    """
    Scale and padding values used to invert the letterbox transform.

    original (W, H)  →  padded (input_w, input_h)
    pixel coordinates in model space  →  original image space:

        x_orig = (x_model - pad_left) / scale
        y_orig = (y_model - pad_top)  / scale
    """
    orig_w: int
    orig_h: int
    input_w: int
    input_h: int
    scale: float        # same scale applied to both axes
    pad_left: float     # horizontal padding (pixels, model space)
    pad_top: float      # vertical padding  (pixels, model space)


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    Resize image with preserved aspect ratio + symmetric grey padding.

    Returns:
        padded_img: uint8 BGR, shape (target_size, target_size, 3)
        meta:       LetterboxMeta for coordinate inversion
    """
    orig_h, orig_w = img.shape[:2]

    # Handle grayscale: promote to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Scale to fit target_size keeping aspect ratio
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Symmetric padding
    pad_left = (target_size - new_w) / 2
    pad_top  = (target_size - new_h) / 2
    pad_right  = target_size - new_w - int(pad_left)
    pad_bottom = target_size - new_h - int(pad_top)

    padded = cv2.copyMakeBorder(
        resized,
        int(pad_top), pad_bottom,
        int(pad_left), pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    meta = LetterboxMeta(
        orig_w=orig_w,
        orig_h=orig_h,
        input_w=target_size,
        input_h=target_size,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return padded, meta


def preprocess(
    img: np.ndarray,
    target_size: int = 640,
) -> Tuple[np.ndarray, LetterboxMeta]:
    """
    Full YOLO preprocessing pipeline (FR-YOLO-003):
      1. letterbox resize
      2. BGR → RGB
      3. uint8 → float32 / 255 normalisation
      4. HWC → CHW
      5. add batch dimension  → (1, 3, H, W)

    Returns:
        tensor: np.float32, shape (1, 3, target_size, target_size)
        meta:   LetterboxMeta
    """
    padded, meta = letterbox(img, target_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)                     # HWC → CHW
    tensor = chw.astype(np.float32) / 255.0           # normalise
    tensor = np.expand_dims(tensor, axis=0)           # add batch dim
    return tensor, meta


def unscale_coords(
    x1: float, y1: float, x2: float, y2: float,
    meta: LetterboxMeta,
) -> Tuple[int, int, int, int]:
    """
    Map bbox from model-input coordinates back to original image pixels.
    Clips to valid range.
    """
    x1 = (x1 - meta.pad_left) / meta.scale
    y1 = (y1 - meta.pad_top)  / meta.scale
    x2 = (x2 - meta.pad_left) / meta.scale
    y2 = (y2 - meta.pad_top)  / meta.scale

    # Clip to original image bounds
    x1 = max(0, min(int(round(x1)), meta.orig_w - 1))
    y1 = max(0, min(int(round(y1)), meta.orig_h - 1))
    x2 = max(0, min(int(round(x2)), meta.orig_w))
    y2 = max(0, min(int(round(y2)), meta.orig_h))

    return x1, y1, x2, y2
