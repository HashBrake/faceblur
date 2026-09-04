"""Ellipse masks and pixel destruction."""
from __future__ import annotations

import numpy as np
import pytest

from faceblur.detect import Detection
from faceblur.redact import build_alpha, build_mask, ellipse_for, redact
from faceblur.settings import Settings

S = Settings()
LM = ((120.0, 130.0), (160.0, 130.0), (140.0, 150.0), (125.0, 168.0), (155.0, 168.0))
FACE = Detection(100.0, 100.0, 80.0, 90.0, 0.9, LM)
NO_LM = Detection(100.0, 100.0, 80.0, 90.0, 0.9, None)


def busy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", ["blur", "pixelate", "solid"])
def test_pixels_change_inside_and_stay_exact_outside(mode):
    frame = busy_frame()
    s = S.with_changes(mode=mode)
    out, alpha = redact(frame, [FACE], s)
    inside = alpha >= 0.5
    outside = alpha == 0
    diff = np.abs(out.astype(np.int16) - frame.astype(np.int16))
    assert diff[inside].mean() > 20
    assert diff[outside].sum() == 0


def test_the_mask_is_an_ellipse_not_a_rectangle():
    """The corners of the box hold hair, background and hands. Leave them."""
    mask = build_mask((480, 640), [FACE], S.with_changes(feather=0))
    assert mask[145, 140] > 0           # centre
    assert mask[101, 101] == 0          # top left corner of the box
    assert mask[189, 179] == 0          # bottom right corner of the box


def test_the_mask_covers_the_landmarks():
    mask = build_mask((480, 640), [FACE], S)
    for x, y in LM:
        assert mask[int(y), int(x)] > 0


def test_the_ellipse_follows_the_eye_line():
    tilted = Detection(100.0, 100.0, 80.0, 90.0, 0.9,
                       ((120.0, 120.0), (160.0, 140.0), (140.0, 150.0),
                        (125.0, 168.0), (155.0, 168.0)))
    e = ellipse_for(tilted, S)
    assert e.angle == pytest.approx(26.6, abs=1.0)


def test_a_box_without_landmarks_gets_a_padded_upright_ellipse():
    e = ellipse_for(NO_LM, S.with_changes(pad=0.10))
    assert e.angle == 0.0
    assert e.ax == pytest.approx(0.5 * 80 * 1.10)
    assert e.ay == pytest.approx(0.5 * 90 * 1.10)


def test_the_mask_is_far_smaller_than_the_old_padded_rectangle():
    mask = build_mask((480, 640), [FACE], S)
    old_rect = (80 * 1.6) * (90 * 1.6)
    assert (mask > 0).sum() < 0.55 * old_rect


def test_feather_softens_only_the_edge():
    alpha = build_alpha((480, 640), [FACE], S.with_changes(feather=6))
    assert alpha[145, 140] == pytest.approx(1.0, abs=0.02)
    assert 0 < alpha.mean() < 0.05
    assert ((alpha > 0) & (alpha < 1)).any()


def test_no_faces_returns_the_frame_untouched():
    frame = busy_frame()
    out, alpha = redact(frame, [], S)
    assert out is frame
    assert alpha.max() == 0


def test_destruction_scales_with_face_size():
    """A small face is destroyed as thoroughly as a large one."""
    frame = busy_frame()
    small = Detection(300.0, 300.0, 24.0, 26.0, 0.9, None)
    out, alpha = redact(frame, [small], S.with_changes(mode="pixelate"))
    inside = alpha >= 0.5
    assert out[inside].std() < frame[inside].std() / 2


def test_a_face_at_the_frame_edge_does_not_crash():
    frame = busy_frame()
    edge = Detection(-20.0, -10.0, 60.0, 60.0, 0.9, None)
    out, alpha = redact(frame, [edge], S)
    assert out.shape == frame.shape
    assert alpha[0, 0] > 0
