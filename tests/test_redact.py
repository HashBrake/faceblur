"""Mask building and pixel destruction."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.redact import build_mask, cover_image, redact

BOX = [100.0, 100.0, 80.0, 80.0, 0.9]


def busy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Random pixels. Any smoothing changes them a lot, so the test can see it."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def mask_bool(frame, boxes, pad):
    return build_mask(frame.shape[:2], boxes, pad) > 0


@pytest.mark.parametrize("mode", ["blur", "pixelate", "solid"])
def test_pixels_change_inside_and_stay_exact_outside(mode):
    """Phase 1 verify: mean absolute difference above 20 inside, exactly 0 outside."""
    frame = busy_frame()
    out = redact(frame, [BOX], pad=0.30, mode=mode, strength=28)
    inside = mask_bool(frame, [BOX], 0.30)

    diff = np.abs(out.astype(np.int16) - frame.astype(np.int16))
    assert diff[inside].mean() > 20, f"{mode} changed too little inside the mask"
    assert diff[~inside].sum() == 0, f"{mode} changed pixels outside the mask"


def test_blur_mode_destroys_information():
    """The cover comes from a downsample, so fine detail cannot come back."""
    frame = busy_frame()
    out = redact(frame, [BOX], pad=0.30, mode="blur", strength=28)
    inside = mask_bool(frame, [BOX], 0.30)
    # Random pixels have a high standard deviation. A downsampled copy is flat.
    assert out[inside].std() < frame[inside].std() / 2


def test_no_boxes_returns_the_frame_untouched():
    frame = busy_frame()
    out = redact(frame, [], pad=0.30, mode="blur", strength=28)
    assert out is frame


def test_mask_is_padded_by_the_right_amount():
    mask = build_mask((480, 640), [[100.0, 100.0, 80.0, 80.0, 0.9]], pad=0.25)
    ys, xs = np.where(mask > 0)
    # 25 percent of 80 is 20 pixels on each side, so the box spans 80 to 200.
    # A filled cv2 rectangle covers its end pixel too, which makes the mask one
    # pixel wider than the range. Wider is the safe direction for this tool.
    assert xs.min() == 80 and ys.min() == 80
    assert xs.max() == 200 and ys.max() == 200


def test_mask_is_a_rectangle_not_an_ellipse():
    """Hairlines and ears live in the corners, so the corners must be covered."""
    mask = build_mask((480, 640), [BOX], pad=0.0)
    assert mask[100, 100] > 0
    assert mask[179, 179] > 0


def test_mask_clips_to_the_frame():
    mask = build_mask((100, 100), [[-40.0, -40.0, 60.0, 60.0, 0.9]], pad=0.5)
    assert mask[0, 0] > 0
    assert mask.shape == (100, 100)


def test_a_box_fully_outside_the_frame_marks_nothing():
    mask = build_mask((100, 100), [[500.0, 500.0, 20.0, 20.0, 0.9]], pad=0.0)
    assert not mask.any()


def test_solid_cover_is_black():
    assert cover_image(busy_frame(), "solid", 28).max() == 0


def test_two_boxes_both_get_covered():
    frame = busy_frame()
    boxes = [BOX, [400.0, 300.0, 60.0, 60.0, 0.8]]
    out = redact(frame, boxes, pad=0.30, mode="pixelate", strength=28)
    inside = mask_bool(frame, boxes, 0.30)
    diff = np.abs(out.astype(np.int16) - frame.astype(np.int16))
    assert diff[inside].mean() > 20
    assert diff[~inside].sum() == 0
