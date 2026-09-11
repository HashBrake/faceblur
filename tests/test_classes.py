"""The kinds of sensitive thing, the switches that choose them, and the mask
shape each one gets.

The scope grew on 2026-09-10 from faces to faces, personal text and screens,
each one a switch of its own. These tests hold the two promises that change
makes: a switch that is off masks nothing of that kind, and a switch that has
no detector behind it cannot be turned on by anybody, on any surface.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from faceblur import classes
from faceblur.classes import ELLIPSE, FACE, POLYGON, SCREEN, TEXT, UnknownKind
from faceblur.detect import Detection
from faceblur.redact import (QuadRegion, build_alpha, ellipse_for, grow_quad,
                             masked_shares, redact, region_for, shape_of)
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


def test_which_kinds_this_build_can_actually_do():
    """This test changes as each kind lands, on purpose: nothing else in the
    codebase should have to, and a reader wants one place that says it."""
    assert [k.name for k in classes.available()] == ["face", "screen"]
    assert FACE.ready and SCREEN.ready
    assert not TEXT.ready


def test_every_ready_kind_is_checked_in_the_finished_copy():
    """The second of the four conditions a kind has to meet before it may be
    called ready: a miss of that kind, found in the copy, has to be counted
    and able to hold the copy back. Without this the switch says the tool
    masks a thing and nothing anywhere tests whether it did.

    This test fails on purpose when a kind is marked ready before its coverage
    in `faceblur/verify.py` lands. Make the check cover it; do not add the
    name here to make the test pass.
    """
    from faceblur.verify import KINDS_CHECKED

    for kind in classes.available():
        assert kind.name in KINDS_CHECKED, (
            f"{kind.name} is marked ready and faceblur/verify.py does not look "
            f"for it in the finished copy")


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

def test_the_default_run_masks_faces_and_nothing_else():
    """Screens are built but off by default. Widening what a tool destroys
    is the owner's decision on the day, not a side effect of an upgrade."""
    assert Settings().mask == ("face",)
    assert Settings().wants("face")
    assert not Settings().wants("screen")
    assert not Settings().wants("text")


def test_screens_can_be_asked_for_on_their_own_or_beside_faces():
    assert Settings(mask=("screen",)).wants("screen")
    both = Settings(mask=("face", "screen"))
    assert both.wants("face") and both.wants("screen")
    assert classes.parse("screen,face") == ("face", "screen"), "listing order"


def test_settings_refuse_a_kind_this_build_cannot_do():
    for bad in ((), ("plate",), ("text",), ("face", "text")):
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


# ------------------------------------------- how much of the frame each kind took
#
# The audit record's masked_mean, masked_p95, masked_max and frames_over_budget
# are the union of every kind's mask, and they keep that meaning. The trouble
# once a second kind can mask beside faces is that a screen mask is large by
# design: on the table tennis file the screen class destroys up to 13 percent
# of a frame where the face class averages under one. Read only the union and
# the face budget stops meaning anything.


def screen(x=100.0, y=100.0, w=300.0, h=200.0) -> Detection:
    return Detection(x, y, w, h, 0.9, None, "tv", True, 0.9, "screen")


SHAPE = (600, 800)


def test_a_frame_with_one_kind_needs_no_second_pass_to_be_split():
    """The union alpha is that kind's mask when it is the only kind there,
    which is most frames, so nothing extra is computed for them."""
    s = Settings()
    dets = [face()]
    alpha = build_alpha(SHAPE, dets, s)
    shares = masked_shares(SHAPE, dets, s, alpha)
    assert list(shares) == ["face"]
    assert shares["face"] == pytest.approx(float((alpha >= 0.5).mean()))


def test_two_kinds_in_one_frame_are_measured_apart():
    s = Settings(mask=("face", "screen"))
    dets = [face(), screen()]
    union = build_alpha(SHAPE, dets, s)
    shares = masked_shares(SHAPE, dets, s, union)
    assert sorted(shares) == ["face", "screen"]
    assert shares["screen"] > shares["face"] * 4, "the screen here is much the larger"
    union_share = float((union >= 0.5).mean())
    # The parts cannot each be the whole, and together they cannot be less
    # than it: they may overlap, so the sum is the union or more.
    assert shares["face"] < union_share
    assert shares["screen"] < union_share
    assert shares["face"] + shares["screen"] >= union_share - 1e-9


def test_a_frame_with_nothing_on_it_splits_into_nothing():
    assert masked_shares(SHAPE, [], Settings()) == {}


def test_the_face_share_does_not_move_when_a_screen_is_added_beside_it():
    """This is the whole point of the split. A face that was 0.4 percent of
    the frame is still 0.4 percent of it when a television is masked too."""
    s = Settings(mask=("face", "screen"))
    alone = masked_shares(SHAPE, [face()], s)
    beside = masked_shares(SHAPE, [face(), screen()], s)
    assert beside["face"] == pytest.approx(alone["face"])


def test_the_record_splits_the_mask_by_kind_and_keeps_the_union(tmp_path, video_with_face):
    """End to end: the union numbers keep their names and their meaning, and
    the per kind ones are new beside them."""
    from faceblur.batch import run_video, serial_submit
    from faceblur.pipeline import STATUS_DONE

    out = tmp_path / "split.mp4"
    record = run_video(video_with_face, out,
                       Settings(device="cpu", copy_clean=False, chunk_seconds=0,
                                encode_seconds=0),
                       serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.masked_mean > 0, "the face clip is masked"
    assert list(record.masked_mean_by_kind) == ["face"]
    assert record.masked_mean_by_kind["face"] == pytest.approx(record.masked_mean, abs=1e-4)
    assert record.masked_max_by_kind["face"] == pytest.approx(record.masked_max, abs=1e-4)
    assert record.frames_over_budget_by_kind["face"] == record.frames_over_budget


def imported_names(path) -> set[str]:
    """Every module and symbol a file imports, by reading its syntax.

    `ast`, not a substring search over the source. A test that greps for a
    word fails on the word appearing in a comment or a docstring, which makes
    it impossible to write about the thing the test is guarding, and it passes
    on `getattr(mod, "tor" + "ch")`, which is the case it should catch. The
    walk covers imports inside functions, which is where this codebase puts
    the expensive ones.
    """
    import ast

    names: set[str] = set()
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module)
                names.add(module.split(".")[0])
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
                names.add(alias.name)
    return names


def test_the_sweep_measures_faces_and_never_sees_a_screen():
    """`eval/measure.py` builds its own per frame list from the raw face cache
    and the tracker. It never runs the screen detector, so the sweep gates in
    `eval/sweep.py` cannot be diluted by a screen mask however many kinds a
    run is asked for.

    This is here so that a later change which starts feeding screens into the
    harness has to face the question rather than quietly move every gate.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "eval"
    names = imported_names(root / "measure.py") | imported_names(root / "sweep.py")
    for forbidden in ("faceblur.screens", "ScreenDetector", "runs_in", "hold"):
        assert forbidden not in names, (
            f"the evaluation harness imports {forbidden}. The sweep gates are "
            f"face numbers and stop meaning what they say if a screen mask reaches "
            f"them; split them by kind first, as the audit record does.")
    from eval import sweep

    assert set(sweep.GATES) == {"off_face_mean", "off_face_max",
                                "off_face_detections_mean", "hand_damage_wearer"}
