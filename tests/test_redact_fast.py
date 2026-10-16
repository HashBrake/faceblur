"""The write pass's mask work on the box around the regions (T14).

`redact_measured` builds the alpha, the zone cut and the shares on a crop of
the frame instead of the whole 1920x1080, and its answer has to be the old
answer exactly: the same output pixels and the same numbers in the record.
The old code is kept below, copied verbatim from `faceblur/redact.py` and
`faceblur/batch.py` before the change, as `_reference_*`, so these tests keep
comparing against what the pipeline used to do rather than against itself.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from faceblur import redact as R
from faceblur.classes import ELLIPSE
from faceblur.detect import Detection
from faceblur.redact import (FaceEllipse, QuadRegion, _cover_region, ellipse_for,
                             grow_quad, region_for, shape_of)
from faceblur.settings import Settings
from faceblur.zone import ZoneFrame

S = Settings(mask=("face", "screen"))
HD = (1080, 1920)
ODD = (481, 643)


# ------------------------------------------------- the old code, verbatim

def _reference_build_alpha(shape, dets, settings):
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    f = settings.feather
    for det in dets:
        if shape_of(det) == ELLIPSE:
            e = ellipse_for(det, settings)
            cv2.ellipse(canvas, (int(round(e.cx)), int(round(e.cy))),
                        (int(round(e.ax + f)), int(round(e.ay + f))),
                        e.angle, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            quad = grow_quad(det.corners(), f)
            cv2.fillConvexPoly(canvas, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
    if f > 0 and canvas.any():
        canvas = cv2.GaussianBlur(canvas, (0, 0), f / 2)
    return canvas.astype(np.float32) / 255.0


def _reference_alpha_with_zone(shape, dets, settings, zone=None):
    from faceblur.zone import REMEMBERED_PROTECTS, alpha_for

    if not zone or not dets:
        return _reference_build_alpha(shape, dets, settings)
    reach = tuple(zone.live) + tuple(zone.remembered)
    on_faces = tuple(zone.hands) if settings.zone_face_needs_hand else tuple(zone.live)
    if not reach and not on_faces:
        return _reference_build_alpha(shape, dets, settings)

    def cut(mask, polygons):
        if not polygons:
            return mask
        return mask * (1.0 - alpha_for(shape, polygons, settings.feather))

    by_reach = [d for d in dets if d.kind in REMEMBERED_PROTECTS]
    by_hand = [d for d in dets if d.kind not in REMEMBERED_PROTECTS]
    if not by_reach:
        return cut(_reference_build_alpha(shape, dets, settings), on_faces)
    if not by_hand:
        return cut(_reference_build_alpha(shape, dets, settings), reach)
    return np.maximum(cut(_reference_build_alpha(shape, by_hand, settings), on_faces),
                      cut(_reference_build_alpha(shape, by_reach, settings), reach))


def _reference_masked_shares(shape, dets, settings, alpha=None, zone=None):
    kinds = {det.kind for det in dets}
    if not kinds:
        return {}
    if len(kinds) == 1 and alpha is not None:
        return {kinds.pop(): float((alpha >= 0.5).mean())}
    return {name: float((_reference_alpha_with_zone(shape, [d for d in dets if d.kind == name],
                                                    settings, zone) >= 0.5).mean())
            for name in sorted(kinds)}


def _reference_redact(frame, dets, settings, zone=None):
    if not dets:
        return frame, np.zeros(frame.shape[:2], np.float32)
    alpha = _reference_alpha_with_zone(frame.shape[:2], dets, settings, zone)
    if not alpha.any():
        return frame, alpha
    H, W = frame.shape[:2]
    regions = [region_for(det, settings) for det in dets]
    margin = settings.feather * 2 + 2
    x0 = max(0, min(e.bounds(margin)[0] for e in regions))
    y0 = max(0, min(e.bounds(margin)[1] for e in regions))
    x1 = min(W, max(e.bounds(margin)[2] for e in regions))
    y1 = min(H, max(e.bounds(margin)[3] for e in regions))
    if x1 <= x0 or y1 <= y0:
        return frame, alpha
    out = frame.copy()
    roi = frame[y0:y1, x0:x1]
    cover = roi.copy()
    for e in regions:
        shifted = (FaceEllipse(e.cx - x0, e.cy - y0, e.ax, e.ay, e.angle)
                   if isinstance(e, FaceEllipse)
                   else QuadRegion(tuple((px - x0, py - y0) for px, py in e.corners)))
        _cover_region(roi, cover, shifted, settings.mode, settings.strength, settings.feather)
    a = alpha[y0:y1, x0:x1, None]
    blended = roi.astype(np.float32) * (1.0 - a) + cover.astype(np.float32) * a
    out[y0:y1, x0:x1] = np.where(a > 0, np.clip(blended + 0.5, 0, 255).astype(np.uint8), roi)
    return out, alpha


def _reference_frame(frame, d, settings, z):
    """The per frame body of `batch.encode_job` before the change."""
    frame_out, alpha = _reference_redact(frame, d, settings, zone=z)
    masked = float((alpha >= 0.5).mean()) if d else 0.0
    if d and z:
        plain = _reference_build_alpha(frame.shape[:2], d, settings) >= 0.5
        prevented = float((plain & ~(alpha >= 0.5)).mean())
    else:
        prevented = 0.0
    shares = (_reference_masked_shares(frame.shape[:2], d, settings, alpha, z)
              if d else {})
    return frame_out, alpha, masked, shares, prevented


# ---------------------------------------------------------------- helpers

def busy(shape, seed=7):
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def face(x, y, size, tilt=0.0):
    """A face box with landmarks in it, the eye line tilted by `tilt` px."""
    lm = ((x + 0.3 * size, y + 0.35 * size - tilt), (x + 0.7 * size, y + 0.35 * size + tilt),
          (x + 0.5 * size, y + 0.55 * size), (x + 0.35 * size, y + 0.75 * size),
          (x + 0.65 * size, y + 0.75 * size))
    return Detection(x, y, size, size, 0.9, lm)


def bare_face(x, y, w, h):
    return Detection(x, y, w, h, 0.9, None)


def text(cx, cy, w, h, angle_deg=0.0):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    quad = tuple((cx + c * dx - s * dy, cy + s * dx + c * dy)
                 for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)))
    return Detection(cx - w / 2, cy - h / 2, w, h, 0.9, None, "text", True, 0.9, "text", quad)


def screen(x, y, w, h):
    return Detection(x, y, w, h, 0.9, None, "tv", True, 0.9, "screen")


def square(cx, cy, half):
    return ((cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half))


def assert_same(frame, dets, settings, zone=None):
    """Old against new: the output frame, the alpha, and every number."""
    ref_out, ref_alpha, ref_masked, ref_shares, ref_prev = _reference_frame(
        frame, dets, settings, zone)
    out, masked, shares, prev = R.redact_measured(frame, dets, settings, zone)
    assert np.array_equal(out, ref_out)
    assert masked == pytest.approx(ref_masked, abs=1e-9)
    assert prev == pytest.approx(ref_prev, abs=1e-9)
    assert set(shares) == set(ref_shares)
    for name in ref_shares:
        assert shares[name] == pytest.approx(ref_shares[name], abs=1e-9)
    # The public signatures still give the old answers.
    pub_out, pub_alpha = R.redact(frame, dets, settings, zone=zone)
    assert np.array_equal(pub_out, ref_out)
    assert np.array_equal(pub_alpha, ref_alpha)
    if dets:
        shape = frame.shape[:2]
        assert np.array_equal(R.build_alpha(shape, dets, settings),
                              _reference_build_alpha(shape, dets, settings))
        assert np.array_equal(R.alpha_with_zone(shape, dets, settings, zone),
                              _reference_alpha_with_zone(shape, dets, settings, zone))
        assert R.masked_shares(shape, dets, settings, pub_alpha, zone) == \
            _reference_masked_shares(shape, dets, settings, ref_alpha, zone)
    return ref_alpha


SHAPES = pytest.mark.parametrize("shape", [HD, ODD], ids=["1080p", "odd"])


# ------------------------------------------------------------------ cases

@SHAPES
def test_one_interior_region(shape):
    alpha = assert_same(busy(shape), [face(200, 150, 90, tilt=8)], S)
    assert (alpha >= 0.5).any(), "the case has to mask something"


@SHAPES
def test_regions_at_every_edge_and_corner_partly_outside_the_frame(shape):
    h, w = shape
    frame = busy(shape, 3)
    edges = [face(-30, -25, 80), face(w - 50, -30, 90, tilt=5), bare_face(-40, h - 45, 90, 80),
             face(w - 45, h - 40, 85), bare_face(w / 2, -35, 70, 60),
             face(-35, h / 2, 70, tilt=-6), text(w - 10, h / 2, 120, 30, 20),
             screen(w / 3, h - 30, 140, 90)]
    for det in edges:
        assert_same(frame, [det], S)
    assert_same(frame, edges, S)


def test_a_region_wholly_outside_the_frame():
    frame = busy(ODD)
    assert_same(frame, [face(-500, -400, 60)], S)
    assert_same(frame, [face(-500, -400, 60), face(100, 100, 60)], S)


@SHAPES
def test_two_overlapping_regions_of_different_kinds(shape):
    frame = busy(shape, 11)
    dets = [face(150, 120, 120, tilt=10), text(230, 200, 220, 40, 15)]
    alpha = assert_same(frame, dets, S)
    assert (alpha >= 0.5).any()
    assert_same(frame, [face(150, 120, 120), screen(200, 150, 160, 120)], S)


@SHAPES
def test_a_zone_polygon_cutting_through_a_region(shape):
    frame = busy(shape, 5)
    f = face(200, 150, 160, tilt=4)
    sc = screen(320, 260, 140, 110)
    t = text(150, 360, 200, 36, -10)
    # A hand over the lower half of the face, its reach over the screen and
    # far beyond the box around the regions, and a remembered polygon.
    hand = square(280, 330, 70)
    live = square(330, 330, 260)
    z = ZoneFrame(live=(live,), remembered=(square(150, 360, 40),), hands=(hand,))
    ref = assert_same(frame, [f], S, z)
    plain = _reference_build_alpha(shape, [f], S)
    assert ((plain >= 0.5) & ~(ref >= 0.5)).any(), "the zone has to take something back"
    assert (ref >= 0.5).any(), "and leave something"
    assert_same(frame, [f, sc], S, z)
    assert_same(frame, [f, sc, t], S, z)
    assert_same(frame, [sc, t], S, z)
    assert_same(frame, [f, sc], S.with_changes(zone_face_needs_hand=False), z)
    # Live only, remembered only, hands only.
    assert_same(frame, [f, t], S, ZoneFrame(live=(live,)))
    assert_same(frame, [f, t], S, ZoneFrame(remembered=(square(260, 260, 120),)))
    assert_same(frame, [f, t], S, ZoneFrame(hands=(hand,)))


def test_a_zone_polygon_off_the_frame_edge():
    h, w = HD
    frame = busy(HD, 9)
    f = face(w - 120, h - 140, 150)
    z = ZoneFrame(live=(square(w, h, 300),), hands=(square(w - 20, h, 90),))
    assert_same(frame, [f, screen(w - 200, h - 80, 150, 120)], S, z)


@SHAPES
def test_no_regions(shape):
    frame = busy(shape)
    assert_same(frame, [], S)
    assert_same(frame, [], S, ZoneFrame(live=(square(100, 100, 50),)))


@pytest.mark.parametrize("mode", ["blur", "pixelate", "solid"])
@pytest.mark.parametrize("changes", [{}, {"feather": 0.0}, {"feather": 2.5}, {"feather": 11.0},
                                     {"pad": 0.4, "feather": 3.0}],
                         ids=["default", "f0", "f2.5", "f11", "pad0.4"])
def test_every_mode_and_a_non_default_feather_and_pad(mode, changes):
    s = S.with_changes(mode=mode, **changes)
    h, w = ODD
    frame = busy(ODD, 13)
    dets = [face(100, 90, 110, tilt=6), bare_face(w - 60, h - 50, 90, 80),
            text(300, 250, 180, 34, 25)]
    z = ZoneFrame(live=(square(330, 260, 70),), hands=(square(150, 160, 40),))
    assert_same(frame, dets, s)
    assert_same(frame, dets, s, z)


def test_random_frames_at_production_size():
    """Two to four regions of 60 to 200 px anywhere, edges included, with
    zones of hand size crossing them, many times over."""
    rng = np.random.default_rng(2026)
    h, w = HD
    frame = busy(HD, 1)
    for _ in range(25):
        dets = []
        for _ in range(int(rng.integers(2, 5))):
            size = float(rng.uniform(60, 200))
            x, y = float(rng.uniform(-size / 2, w - size / 2)), float(rng.uniform(-size / 2, h - size / 2))
            pick = rng.integers(0, 4)
            if pick == 0:
                dets.append(face(x, y, size, tilt=float(rng.uniform(-15, 15))))
            elif pick == 1:
                dets.append(bare_face(x, y, size, size * float(rng.uniform(0.8, 1.3))))
            elif pick == 2:
                dets.append(text(x, y, size * 2, size / 4, float(rng.uniform(-40, 40))))
            else:
                dets.append(screen(x, y, size, size * 0.7))
        anchor = dets[0]
        z = ZoneFrame(live=(square(anchor.cx + float(rng.uniform(-80, 80)),
                                   anchor.cy + float(rng.uniform(-80, 80)),
                                   float(rng.uniform(100, 400))),),
                      remembered=(square(float(rng.uniform(0, w)), float(rng.uniform(0, h)),
                                         float(rng.uniform(50, 200))),),
                      hands=(square(anchor.cx + float(rng.uniform(-60, 60)),
                                    anchor.cy + float(rng.uniform(-60, 60)),
                                    float(rng.uniform(30, 90))),))
        s = S.with_changes(feather=float(rng.choice([0.0, 3.0, 6.0, 9.5])))
        assert_same(frame, dets, s, z)


# ------------------------------------------------- what the change promises

@SHAPES
def test_the_alpha_is_zero_outside_the_crop(shape):
    """What makes a count over the box divided by the frame's pixel count the
    full frame's share."""
    h, w = shape
    dets = [face(200, 150, 160, tilt=4), text(w - 30, h - 20, 200, 40, 30),
            bare_face(-40, 100, 90, 90)]
    z = ZoneFrame(live=(square(250, 250, 200),), hands=(square(260, 260, 60),))
    for s in (S, S.with_changes(feather=11.0), S.with_changes(feather=0.0)):
        x0, y0, x1, y1 = R._crop(shape, dets, s)
        for alpha in (_reference_build_alpha(shape, dets, s),
                      _reference_alpha_with_zone(shape, dets, s, z)):
            assert alpha.any()
            outside = alpha.copy()
            outside[y0:y1, x0:x1] = 0
            assert not outside.any()


def test_a_frame_with_no_regions_does_no_mask_work(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("mask work on a frame with no regions")

    monkeypatch.setattr(R, "_alpha_in", boom)
    monkeypatch.setattr(R, "_zone_in", boom)
    frame = busy(ODD)
    out, masked, shares, prevented = R.redact_measured(
        frame, [], S, ZoneFrame(live=(square(100, 100, 50),)))
    assert out is frame
    assert (masked, shares, prevented) == (0.0, {}, 0.0)


def test_the_plain_alpha_and_the_zone_are_each_drawn_once(monkeypatch):
    """The share the zone took back reuses the plain alpha the cut was made
    from, where the old write pass drew it a second time."""
    calls = {"alpha": 0, "zone": 0}
    alpha_in, zone_in = R._alpha_in, R._zone_in

    def count_alpha(*a, **k):
        calls["alpha"] += 1
        return alpha_in(*a, **k)

    def count_zone(*a, **k):
        calls["zone"] += 1
        return zone_in(*a, **k)

    monkeypatch.setattr(R, "_alpha_in", count_alpha)
    monkeypatch.setattr(R, "_zone_in", count_zone)
    frame = busy(HD)
    z = ZoneFrame(live=(square(300, 300, 150),), hands=(square(300, 300, 60),))
    _, _, _, prevented = R.redact_measured(frame, [face(200, 200, 150), face(400, 300, 90)], S, z)
    assert prevented > 0
    assert calls == {"alpha": 1, "zone": 1}


def test_the_crop_arrays_are_small(monkeypatch):
    """No full frame alpha: every canvas drawn is the box's size."""
    sizes = []
    blur = R.cv2.GaussianBlur

    def spy(src, *a, **k):
        sizes.append(src.shape)
        return blur(src, *a, **k)

    monkeypatch.setattr(R.cv2, "GaussianBlur", spy)
    frame = busy(HD)
    z = ZoneFrame(live=(square(300, 300, 120),), hands=(square(300, 300, 60),))
    R.redact_measured(frame, [face(250, 250, 120)], S, z)
    assert sizes
    for shape in sizes:
        assert shape[0] * shape[1] < 0.1 * HD[0] * HD[1]


def test_the_zone_over_the_box_is_the_zone_over_the_frame():
    """Box by box against `zone.alpha_for` on the whole frame, for hand sized
    polygons at any angle that cross the box's edges. Drawn cut at the box,
    OpenCV clips each anti aliased edge first and lands it a level away: on
    20000 such cases a third differed, up to 165 px inside the canvas."""
    from faceblur.zone import alpha_for

    rng = np.random.default_rng(17)
    h, w = ODD
    for _ in range(150):
        c = rng.uniform(-100, 700, 2)
        half = rng.uniform(20, 400, 2)
        turn = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        poly = [tuple(p) for p in (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half) @ rot.T + c]
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        box = (x0, y0, int(rng.integers(x0 + 50, w + 1)), int(rng.integers(y0 + 50, h + 1)))
        feather = float(rng.choice([0.0, 3.0, 6.0]))
        whole = alpha_for(ODD, [poly], feather)
        assert np.array_equal(R._zone_in(ODD, box, [poly], feather),
                              whole[box[1]:box[3], box[0]:box[2]])
