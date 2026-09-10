"""The kinds of sensitive thing, the switches that choose them, and the mask
shape each one gets.

The scope grew on 2026-09-10 from faces to faces, personal text and screens,
each one a switch of its own. These tests hold the two promises that change
makes: a switch that is off masks nothing of that kind, and a switch that has
no detector behind it cannot be turned on by anybody, on any surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceblur import classes
from faceblur.classes import ELLIPSE, FACE, POLYGON, SCREEN, TEXT, UnknownKind
from faceblur.detect import Detection
from faceblur.redact import (QuadRegion, build_alpha, ellipse_for, grow_quad,
                             redact, region_for, shape_of)
from faceblur.settings import Settings, SettingsError


def face(x=100.0, y=100.0, size=60.0) -> Detection:
    return Detection(x, y, size, size, 0.9)


def text(x=100.0, y=100.0, w=160.0, h=24.0, quad=None) -> Detection:
    return Detection(x, y, w, h, 0.9, kind="text", quad=quad)


# ------------------------------------------------------------- the registry

def test_every_kind_is_named_once_and_has_a_shape():
    names = [k.name for k in classes.KINDS]
    assert names == sorted(set(names), key=names.index), "a kind is listed twice"
    for k in classes.KINDS:
        assert k.shape in (ELLIPSE, POLYGON)
        assert k.label and k.summary


def test_only_faces_are_ready_in_this_build():
    """When text or screens land, this test changes with them, on purpose:
    nothing else in the codebase should have to."""
    assert [k.name for k in classes.available()] == ["face"]
    assert FACE.ready
    assert not TEXT.ready and not SCREEN.ready


def test_a_face_is_hidden_by_an_ellipse_and_the_others_by_their_own_corners():
    assert FACE.shape == ELLIPSE
    assert TEXT.shape == POLYGON and SCREEN.shape == POLYGON


def test_an_unknown_kind_is_refused_by_name():
    with pytest.raises(UnknownKind) as exc:
        classes.kind("plate")
    assert "plate" in str(exc.value)
    assert "face" in str(exc.value), "the message has to say what it does know"


# ------------------------------------------------------------- parsing --mask

def test_the_flag_reads_a_list_and_puts_it_in_listing_order():
    assert classes.parse("face") == ("face",)
    assert classes.parse(" face , face ") == ("face",)


def test_the_flag_refuses_a_kind_with_no_detector_behind_it():
    """A switch that masks nothing is worse than no switch, so it is an error
    rather than a warning, on every surface."""
    with pytest.raises(UnknownKind) as exc:
        classes.parse("face,text")
    assert "cannot mask text yet" in str(exc.value)


def test_the_flag_refuses_an_empty_list():
    with pytest.raises(UnknownKind):
        classes.parse("  ,, ")


# ----------------------------------------------------------------- settings

def test_the_default_run_masks_faces():
    assert Settings().mask == ("face",)
    assert Settings().wants("face")
    assert not Settings().wants("text")


def test_settings_refuse_a_kind_this_build_cannot_do():
    for bad in ((), ("plate",), ("text",), ("face", "screen")):
        with pytest.raises(SettingsError):
            Settings(mask=bad)


def test_the_record_says_what_the_run_was_asked_to_mask():
    d = Settings().to_dict()
    assert d["mask"] == ["face"], "a list, so a record read back from JSON matches"


def test_a_run_that_does_not_want_faces_loads_no_face_detector():
    """Not a filter after the fact: the models are never built, so a run for
    text alone will not pay for face detection."""
    from faceblur.detect import DetectorBank

    bank = DetectorBank(Settings(device="cpu"))
    assert bank.backends, "faces wanted, so there are detectors"

    quiet = Settings(device="cpu")
    object.__setattr__(quiet, "mask", ("screen",))     # past validate, on purpose
    assert DetectorBank(quiet).backends == []


# -------------------------------------------------------------- mask shapes

def test_a_face_gets_an_ellipse_and_text_gets_its_quad():
    s = Settings()
    assert shape_of(face()) == ELLIPSE
    assert shape_of(text()) == POLYGON
    assert region_for(face(), s) == ellipse_for(face(), s)
    assert isinstance(region_for(text(), s), QuadRegion)


def test_an_unknown_kind_is_masked_as_a_polygon_not_skipped():
    """Belt and braces: a box whose kind this build does not recognise is
    still hidden, by the box itself, rather than quietly let through."""
    assert shape_of(Detection(0, 0, 10, 10, 0.9, kind="plate")) == POLYGON


def test_a_box_with_no_corners_falls_back_to_its_rectangle():
    d = text(quad=None)
    assert d.corners() == ((100.0, 100.0), (260.0, 100.0),
                           (260.0, 124.0), (100.0, 124.0))


def test_a_rotated_quad_is_masked_where_it_lies_not_where_its_bounds_are():
    """A line of text on a page held at an angle. Masking its upright bounds
    would destroy the page around it."""
    diamond = ((50.0, 0.0), (100.0, 50.0), (50.0, 100.0), (0.0, 50.0))
    alpha = build_alpha((100, 100), [text(0, 0, 100, 100, quad=diamond)],
                        Settings(feather=0))
    assert alpha[50, 50] == pytest.approx(1.0)      # the middle is masked
    assert alpha[2, 2] == 0.0                       # the corners are not
    assert 0.4 < float(alpha.mean()) < 0.6          # about half, as a diamond is


def test_growing_a_quad_pushes_every_corner_out():
    grown = grow_quad(((0, 0), (100, 0), (100, 20), (0, 20)), 5.0)
    assert grown[:, 0].min() < 0 and grown[:, 0].max() > 100
    assert grown[:, 1].min() < 0 and grown[:, 1].max() > 20


def test_growing_a_degenerate_quad_does_not_explode():
    flat = ((10, 10), (10, 10), (10, 10), (10, 10))
    assert np.allclose(grow_quad(flat, 5.0), np.asarray(flat, np.float64))


def test_blocks_are_sized_by_the_short_side_of_a_line_of_text():
    """A 400 px line 20 px tall. Sizing the blocks by its length would give
    one block over the whole line and destroy nothing across it."""
    line = QuadRegion(((0, 0), (400, 0), (400, 20), (0, 20)))
    assert line.block_scale() == pytest.approx(10.0)
    tall = QuadRegion(((0, 0), (20, 0), (20, 400), (0, 400)))
    assert tall.block_scale() == pytest.approx(10.0)


# ------------------------------------------------------------- end to end

def test_text_is_destroyed_inside_its_quad_and_nowhere_else():
    frame = np.random.default_rng(0).integers(0, 255, (200, 400, 3), dtype=np.uint8)
    out, alpha = redact(frame, [text(200, 50, 160, 24)], Settings())
    changed = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2) > 0
    assert changed[50:74, 200:360].all(), "the whole line has to go"
    assert not changed[:40, :40].any(), "and nothing else may"


def test_the_same_box_is_masked_differently_by_kind():
    """The one comparison that says what the shape choice buys. The corner of
    a face box is hair and background and survives; the corner of a text quad
    is the first character and does not."""
    box = dict(x=20.0, y=20.0, w=200.0, h=200.0)
    as_face = Detection(score=0.9, **box)
    as_text = Detection(score=0.9, kind="text", **box)
    s = Settings()
    face_alpha = build_alpha((300, 300), [as_face], s)
    text_alpha = build_alpha((300, 300), [as_text], s)
    assert face_alpha[25, 25] == 0.0
    assert text_alpha[25, 25] == pytest.approx(1.0)
    assert face_alpha[120, 120] == pytest.approx(1.0)   # both cover the middle
    assert text_alpha[120, 120] == pytest.approx(1.0)
    assert float(face_alpha.sum()) < float(text_alpha.sum())


def test_a_face_and_a_line_of_text_in_one_frame_are_both_destroyed():
    frame = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    out, _ = redact(frame, [face(40, 40, 80), text(200, 200, 150, 20)], Settings())
    changed = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2) > 0
    assert changed[80, 80], "the face"
    assert changed[201, 201], "the text"
    assert not changed[280, 20], "and nothing between them"
