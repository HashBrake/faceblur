"""The text detector: geometry, thresholds, and the promise that it is off.

Package T1 builds the detector and measures it. Nothing masks text: the kind
is not ready, the pipeline never calls this module, and the tests at the
bottom are there to keep that true until package T3 decides otherwise on the
evidence in report 21.

The model itself is not tested here. What it finds on real footage is
measured by `eval/text.py` and audited by eye in
`docs/audits/text_quads_2026-09-12.csv`; what this file covers is the
geometry and the bookkeeping around it, which is where a quiet mistake would
live.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceblur.classes import POLYGON, TEXT
from faceblur.settings import Settings, SettingsError
from faceblur.text import (STRIDE, box_score, is_page, letterbox, merged,
                           quads_from, unclipped)

SHAPE = (600, 800)


def frame(h=600, w=800):
    return np.zeros((h, w, 3), np.uint8)


def prob_with(rect, shape=(300, 400), value=1.0):
    """A probability map with one solid rectangle in it."""
    prob = np.zeros(shape, np.float32)
    x, y, w, h = rect
    prob[y:y + h, x:x + w] = value
    return prob


# ------------------------------------------------------------------ geometry

def test_the_letterbox_keeps_the_aspect_and_pads_to_the_stride():
    canvas, scale = letterbox(frame(600, 800), 960)
    assert scale == pytest.approx(960 / 800)
    assert canvas.shape[0] % STRIDE == 0 and canvas.shape[1] % STRIDE == 0
    assert canvas.shape[1] >= 960 and canvas.shape[0] >= int(600 * scale)


def test_a_tall_frame_is_scaled_by_its_long_side_too():
    canvas, scale = letterbox(frame(800, 600), 960)
    assert scale == pytest.approx(960 / 800)
    assert canvas.shape[0] >= 960


def test_the_padding_is_below_and_to_the_right_so_coordinates_survive():
    """A point in the canvas divided by the scale is a point in the frame,
    which is only true if the picture starts at the top left."""
    img = frame(600, 800)
    img[0, 0] = (255, 255, 255)
    canvas, _ = letterbox(img, 960)
    assert tuple(canvas[0, 0]) == (255, 255, 255)
    assert tuple(canvas[-1, -1]) == (0, 0, 0)


def test_the_unclip_grows_a_rectangle_by_the_published_rule():
    """DB offsets the polygon by area times the ratio over the perimeter, and
    for a rectangle that is exactly both sides longer by twice the distance."""
    rect = ((100.0, 100.0), (40.0, 10.0), 0.0)
    (cx, cy), (w, h), _ = unclipped(rect, 1.6)
    distance = (40 * 10 * 1.6) / (2 * (40 + 10))
    assert (cx, cy) == (100.0, 100.0), "the unclip moves nothing"
    assert w == pytest.approx(40 + 2 * distance)
    assert h == pytest.approx(10 + 2 * distance)


def test_an_empty_rectangle_is_left_alone_rather_than_dividing_by_zero():
    assert unclipped(((1.0, 1.0), (0.0, 0.0), 0.0), 1.6)[1] == (0.0, 0.0)


def test_the_box_score_is_the_mean_inside_the_quad_and_not_the_peak():
    """A wall with one bright pixel is not a line of text."""
    prob = np.zeros((100, 100), np.float32)
    prob[50, 50] = 1.0
    quad = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)
    assert box_score(prob, quad) < 0.01
    prob[10:90, 10:90] = 1.0
    assert box_score(prob, quad) > 0.9


def test_a_quad_off_the_edge_of_the_map_does_not_raise():
    prob = np.zeros((50, 50), np.float32)
    quad = np.array([[-40, -40], [-10, -40], [-10, -10], [-40, -10]], np.float32)
    assert box_score(prob, quad) == 0.0


# ------------------------------------------------------------- what survives

def test_a_solid_patch_becomes_one_quad_in_frame_coordinates():
    settings = Settings()
    prob = prob_with((40, 30, 120, 20))
    found = quads_from(prob, 0.5, SHAPE, settings)
    assert len(found) == 1
    det = found[0]
    assert det.kind == "text" and det.source == "ppocr"
    assert det.quad is not None and len(det.quad) == 4
    # The map was built at half scale, so the box is twice as far out.
    assert det.x < 80 + 1 and det.x + det.w > 2 * 160 - 1


def test_a_faint_patch_is_dropped_by_the_box_threshold():
    settings = Settings()
    assert quads_from(prob_with((40, 30, 120, 20), value=0.55), 0.5, SHAPE, settings) == []
    assert quads_from(prob_with((40, 30, 120, 20), value=0.95), 0.5, SHAPE, settings)


def test_a_quad_thinner_than_the_floor_is_dropped():
    settings = Settings(text_min_px=200)
    assert quads_from(prob_with((40, 30, 120, 20)), 0.5, SHAPE, settings) == []


def test_a_quad_is_clipped_to_the_frame():
    settings = Settings()
    found = quads_from(prob_with((0, 0, 200, 40)), 0.5, SHAPE, settings)
    assert found
    for x, y in found[0].quad:
        assert 0 <= x <= SHAPE[1] and 0 <= y <= SHAPE[0]


def test_a_page_is_kept_and_named():
    settings = Settings()
    small = quads_from(prob_with((40, 30, 60, 20)), 0.5, SHAPE, settings)[0]
    assert not is_page(small, SHAPE, settings)
    big = quads_from(prob_with((0, 0, 400, 300)), 0.5, SHAPE, settings)[0]
    assert is_page(big, SHAPE, settings), "a quad over text_max_frac is a page"


def test_two_scales_that_found_the_same_line_are_merged_to_one():
    settings = Settings()
    a = quads_from(prob_with((40, 30, 120, 20)), 0.5, SHAPE, settings)
    b = quads_from(prob_with((41, 31, 120, 20)), 0.5, SHAPE, settings)
    assert len(merged(a + b, settings)) == 1


def test_two_lines_far_apart_are_both_kept():
    settings = Settings()
    a = quads_from(prob_with((10, 10, 60, 16)), 0.5, SHAPE, settings)
    b = quads_from(prob_with((200, 200, 60, 16)), 0.5, SHAPE, settings)
    assert len(merged(a + b, settings)) == 2


def test_merging_nothing_gives_nothing():
    assert merged([], Settings()) == []


# ------------------------------------------------------------- over the time

def test_a_line_seen_once_is_dropped_and_a_line_seen_twice_is_held():
    """The same rule screens have, for the same reason: a detector that fires
    on a tile pattern fires once, and a sign stays a sign."""
    from faceblur.text import hold

    settings = Settings()
    found = quads_from(prob_with((40, 30, 120, 20)), 0.5, SHAPE, settings)
    once = hold({5: found}, settings, 40)
    assert once == {}
    twice = hold({5: found, 6: found}, settings, 40)
    assert twice, "two sightings is a line"
    assert max(twice) >= 6 + settings.text_tail - 1, "and it carries a tail"


def test_the_tail_does_not_run_past_the_end_of_the_video():
    from faceblur.text import hold

    settings = Settings()
    found = quads_from(prob_with((40, 30, 120, 20)), 0.5, SHAPE, settings)
    held = hold({8: found, 9: found}, settings, 10)
    assert max(held) < 10


# ------------------------------------------------------------ the settings

def test_every_text_setting_is_validated_and_recorded():
    recorded = Settings().to_dict()
    for name in ("text_sizes", "text_thresh", "text_box_thresh", "text_unclip",
                 "text_min_px", "text_max_frac", "text_nms", "text_min_run",
                 "text_gap", "text_tail", "text_link_iou"):
        assert name in recorded, f"{name} is not in the audit record"
    for bad in ({"text_thresh": 1.5}, {"text_box_thresh": 0.0}, {"text_unclip": 0.0},
                {"text_min_px": 0}, {"text_sizes": (32,)}, {"text_max_frac": 2.0},
                {"text_min_run": 0}, {"text_gap": -1}):
        with pytest.raises(SettingsError):
            Settings(**bad)


def test_the_scales_are_the_ones_the_measurement_chose():
    """1920 and 2560, not the 960 and 1920 the build plan proposed. At 960 the
    detector finds almost nothing on this footage and what it finds is mostly
    one huge false quad a frame. Report 21.2."""
    assert Settings().text_sizes == (1920, 2560)


# --------------------------------------------------- and it is still not on

def test_text_is_not_ready_and_the_pipeline_never_calls_it():
    """T1 measures. Nothing masks text until T3 decides on the evidence, and
    90 percent of what this detector finds on the sample footage is not text
    at all."""
    import pathlib

    assert TEXT.ready is False
    assert TEXT.shape == POLYGON
    root = pathlib.Path(__file__).resolve().parents[1] / "faceblur"
    for name in ("batch.py", "pipeline.py", "redact.py", "verify.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "text import" not in source and "from .text" not in source, (
            f"faceblur/{name} imports the text detector. Nothing in the pipeline "
            f"may call it until package T3 sets TEXT.ready on the evidence")


def test_the_audit_tables_come_from_the_committed_labels():
    """Report 21.3's numbers are printed from the CSV by a command, the rule
    report 17.7 was rewritten to keep."""
    from eval.text import AUDIT_CSV, audit_tables

    assert AUDIT_CSV.is_file(), "the by eye labels are not committed"
    text = audit_tables()
    assert "124 quads from 4 files" in text
    assert "| none | 111 | 90% |" in text
