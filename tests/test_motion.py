"""Camera shift between frames."""
from __future__ import annotations

import numpy as np
import pytest

from faceblur.motion import downscale, estimate_shift


def textured(seed=1, size=(650, 800)):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size[0] + 200, size[1] + 200, 3), dtype=np.uint8)
    # Smooth a little so the texture has structure at the downscaled size.
    import cv2
    return cv2.GaussianBlur(base, (0, 0), 3)


def test_a_still_camera_gives_no_shift():
    big = textured()
    a = big[100:750, 100:900]
    dx, dy = estimate_shift(downscale(a), downscale(a), 800)
    assert abs(dx) < 1 and abs(dy) < 1


@pytest.mark.parametrize("sx, sy", [(24, 0), (0, -16), (-40, 30)])
def test_a_pan_is_measured_in_source_pixels(sx, sy):
    big = textured()
    a = big[100:750, 100:900]
    # The picture content moves by (sx, sy): what was at x is now at x + sx.
    b = big[100 - sy:750 - sy, 100 - sx:900 - sx]
    dx, dy = estimate_shift(downscale(a), downscale(b), 800)
    assert dx == pytest.approx(sx, abs=3)
    assert dy == pytest.approx(sy, abs=3)


def test_no_previous_frame_means_still():
    a = textured()[100:750, 100:900]
    assert estimate_shift(None, downscale(a), 800) == (0.0, 0.0)


def test_a_cut_between_unrelated_pictures_is_not_a_shift():
    a = textured(1)[100:750, 100:900]
    b = textured(2)[100:750, 100:900]
    dx, dy = estimate_shift(downscale(a), downscale(b), 800)
    assert abs(dx) < 4 and abs(dy) < 4


def test_downscale_keeps_the_aspect_and_long_side():
    small = downscale(np.zeros((1300, 1600, 3), np.uint8))
    assert small.shape == (325, 400)
    assert small.dtype == np.float32
