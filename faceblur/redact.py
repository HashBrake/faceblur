"""Mask building and pixel destruction.

Rectangles, not ellipses. An ellipse inscribed in the box leaves the corners, and
hairlines and ears live in the corners.

Every mode except `solid` downsamples the frame first. The downsample throws
information away, so the original pixels cannot be recovered from the output. A
plain Gaussian blur is a linear filter and can be partly inverted, so this file
never ships a mode that only blurs.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def build_mask(shape: tuple[int, int], boxes: Sequence[Sequence[float]],
               pad: float) -> np.ndarray:
    """Union of padded rectangles, as a uint8 mask the size of the frame."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh, *_ in boxes:
        px, py = bw * pad, bh * pad
        x0, y0 = int(max(0, x - px)), int(max(0, y - py))
        x1, y1 = int(min(w, x + bw + px)), int(min(h, y + bh + py))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def cover_image(frame: np.ndarray, mode: str, strength: int) -> np.ndarray:
    """The image the masked pixels are replaced from."""
    h, w = frame.shape[:2]
    if mode == "solid":
        return np.zeros_like(frame)

    k = max(1, int(round(min(h, w) / strength)))
    small = cv2.resize(frame, (max(1, w // k), max(1, h // k)),
                       interpolation=cv2.INTER_AREA)
    if mode == "pixelate":
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    cover = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(cover, (0, 0), k)


def redact(frame: np.ndarray, boxes: Sequence[Sequence[float]], pad: float,
           mode: str, strength: int) -> np.ndarray:
    """Return the frame with every box area destroyed. Pixels outside stay exact."""
    if not len(boxes):
        return frame
    mask = build_mask(frame.shape[:2], boxes, pad)
    if not mask.any():
        return frame

    if mode == "solid":
        out = frame.copy()
        out[mask > 0] = 0
        return out

    cover = cover_image(frame, mode, strength)
    m3 = cv2.merge([mask] * 3) > 0
    return np.where(m3, cover, frame)
