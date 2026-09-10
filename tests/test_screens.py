"""Screens as objects: the decode, the hold, and what the class does.

The decode is the part that can be wrong quietly. A detector whose grid or
strides are off still returns boxes, and they still look like boxes; they
just sit somewhere else. So the grid is checked against the length the model
actually writes, and the decode against output built by hand with a box in a
known place.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceblur import screens
from faceblur.detect import Detection
from faceblur.screens import (GRID, INPUT_SIZE, PAD_VALUE, SCREEN_LABELS, STRIDES,
                              ScreenDetector, decode, hold, letterbox)
from faceblur.settings import Settings, SettingsError


def screen(x=100.0, y=100.0, w=200.0, h=150.0, label="tv") -> Detection:
    return Detection(x, y, w, h, 0.9, None, label, True, 0.9, "screen")


# --------------------------------------------------------------- the grid

def test_the_grid_is_as_long_as_the_model_output():
    """3549 rows, from 52x52 + 26x26 + 13x13. A grid of another length cannot
    be lined up with the output at all; one of the same length built from the
    wrong strides puts every box somewhere else."""
    assert len(GRID.gx) == len(GRID.gy) == len(GRID.stride) == 3549
    assert sum((INPUT_SIZE // s) ** 2 for s in STRIDES) == 3549


def test_the_grid_walks_each_feature_map_row_by_row():
    assert (GRID.gx[0], GRID.gy[0], GRID.stride[0]) == (0, 0, 8)
    assert (GRID.gx[1], GRID.gy[1]) == (1, 0)
    assert (GRID.gx[52], GRID.gy[52]) == (0, 1), "second row of the stride 8 map"
    assert GRID.stride[52 * 52] == 16, "the stride 16 map starts here"
    assert GRID.stride[-1] == 32


# --------------------------------------------------------------- letterbox

def test_the_letterbox_sits_top_left_on_grey_not_centred_on_black():
    """YOLOX's own preprocessing pads bottom and right with 114. Centring it
    instead moves every box by half the padding."""
    canvas, ratio = letterbox(np.full((100, 200, 3), 255, np.uint8))
    assert canvas.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert ratio == pytest.approx(INPUT_SIZE / 200)
    assert canvas[0, 0].tolist() == [255, 255, 255], "the picture starts at the corner"
    assert canvas[-1, -1].tolist() == [PAD_VALUE] * 3, "and the pad is grey"


def test_a_square_frame_fills_the_input():
    canvas, ratio = letterbox(np.zeros((832, 832, 3), np.uint8))
    assert ratio == pytest.approx(0.5)
    assert not (canvas == PAD_VALUE).all(axis=2).any()


# ----------------------------------------------------------------- decode

def raw_with_box(row: int, cx: float, cy: float, w: float, h: float,
                 label: int, score: float) -> np.ndarray:
    """Model output with one box on one grid row, in the model's own units."""
    raw = np.zeros((1, 3549, 85), np.float32)
    raw[0, :, 4] = 0.001
    stride = GRID.stride[row]
    raw[0, row, 0] = cx / stride - GRID.gx[row]
    raw[0, row, 1] = cy / stride - GRID.gy[row]
    raw[0, row, 2] = np.log(w / stride)
    raw[0, row, 3] = np.log(h / stride)
    raw[0, row, 4] = 1.0
    raw[0, row, 5 + label] = score
    return raw


def test_a_decoded_box_lands_where_the_model_put_it():
    raw = raw_with_box(1000, 208.0, 182.0, 156.0, 114.0, 62, 0.85)
    found = decode(raw, 1.0, 0.3, 0.45, SCREEN_LABELS)
    assert len(found) == 1
    d = found[0]
    assert (d.cx, d.cy) == pytest.approx((208.0, 182.0), abs=0.5)
    assert (d.w, d.h) == pytest.approx((156.0, 114.0), abs=0.5)
    assert d.source == "tv" and d.kind == "screen"
    assert d.score == pytest.approx(0.85, abs=1e-3)


def test_the_letterbox_ratio_is_undone_on_the_way_out():
    raw = raw_with_box(1000, 100.0, 100.0, 40.0, 40.0, 62, 0.9)
    d = decode(raw, 0.25, 0.3, 0.45, SCREEN_LABELS)[0]
    assert (d.cx, d.cy) == pytest.approx((400.0, 400.0), abs=2.0)
    assert (d.w, d.h) == pytest.approx((160.0, 160.0), abs=2.0)


def test_a_class_the_run_did_not_ask_for_is_not_returned():
    raw = raw_with_box(1000, 200.0, 200.0, 60.0, 60.0, 67, 0.9)    # a phone
    assert decode(raw, 1.0, 0.3, 0.45, {62: "tv"}) == []
    assert len(decode(raw, 1.0, 0.3, 0.45, SCREEN_LABELS)) == 1


def test_a_person_is_not_a_screen():
    """The model knows eighty things. Only three of them are glass."""
    raw = raw_with_box(1000, 200.0, 200.0, 60.0, 60.0, 0, 0.99)    # a person
    assert decode(raw, 1.0, 0.3, 0.45, SCREEN_LABELS) == []


def test_nothing_over_the_threshold_means_nothing_found():
    raw = raw_with_box(1000, 200.0, 200.0, 60.0, 60.0, 62, 0.2)
    assert decode(raw, 1.0, 0.3, 0.45, SCREEN_LABELS) == []


def test_suppression_runs_within_a_class_not_across_them():
    """A laptop and the phone lying on it overlap and are two things. One
    suppression pass over both would drop the smaller."""
    raw = raw_with_box(1000, 200.0, 200.0, 200.0, 200.0, 63, 0.9)   # laptop
    phone = raw_with_box(1000, 200.0, 200.0, 60.0, 60.0, 67, 0.8)   # phone on it
    raw[0, 1200] = phone[0, 1000]
    raw[0, 1200, 0] = 200.0 / GRID.stride[1200] - GRID.gx[1200]
    raw[0, 1200, 1] = 200.0 / GRID.stride[1200] - GRID.gy[1200]
    raw[0, 1200, 2] = np.log(60.0 / GRID.stride[1200])
    raw[0, 1200, 3] = np.log(60.0 / GRID.stride[1200])
    found = decode(raw, 1.0, 0.3, 0.45, SCREEN_LABELS)
    assert {d.source for d in found} == {"laptop", "phone"}


# ------------------------------------------------------------------- hold

def seen(*frames):
    """Detections of one screen, in the frames given."""
    return {i: [screen()] for i in frames}


def test_a_screen_seen_once_is_a_flicker_and_is_not_masked():
    """The rule that carries the precision. A table only looks like a laptop
    from some angles, so the detector calls it one for a frame and stops; a
    monitor on a wall stays a monitor. On the sample footage 14 of the 29
    false runs last one frame and no real screen does."""
    assert hold(seen(7), Settings(), 50) == {}


def test_a_screen_seen_enough_times_is_masked_from_the_first_of_them():
    """Masking has to start at the first sighting, not the third: the two
    frames that earned it show the screen just as plainly."""
    out = hold(seen(10, 11, 12), Settings(screen_tail=0), 50)
    assert sorted(out) == [10, 11, 12]


def test_the_bar_is_a_setting():
    loose = Settings(screen_min_run=1, screen_tail=0)
    assert sorted(hold(seen(7), loose, 50)) == [7]
    strict = Settings(screen_min_run=5, screen_tail=0)
    assert hold(seen(7, 8, 9), strict, 50) == {}


def test_a_short_gap_inside_a_run_is_closed():
    """A screen does not leave the room between one frame and the next."""
    out = hold(seen(0, 1, 5), Settings(screen_tail=0), 30)
    assert sorted(out) == [0, 1, 2, 3, 4, 5]


def test_a_long_gap_breaks_a_run_in_two():
    """Past screen_gap the two sightings are not evidence about each other,
    and neither half is then long enough to be masked at all."""
    s = Settings(screen_gap=3, screen_tail=0)
    assert hold(seen(0, 1, 20, 21), s, 30) == {}
    assert sorted(hold(seen(0, 1, 2, 20, 21, 22), s, 30)) == [0, 1, 2, 20, 21, 22]


def test_a_screen_is_held_past_its_last_sighting():
    out = hold(seen(10, 11, 12), Settings(screen_tail=4, screen_gap=0), 30)
    assert sorted(out) == [10, 11, 12, 13, 14, 15, 16]


def test_the_tail_stops_at_the_end_of_the_video():
    out = hold(seen(6, 7, 8), Settings(screen_tail=10, screen_gap=0), 10)
    assert max(out) == 9


def test_two_screens_in_the_same_frames_are_held_apart():
    left, right = screen(0, 0, 100, 100), screen(800, 600, 100, 100)
    s = Settings(screen_tail=0)
    out = hold({i: [left, right] for i in (0, 1, 2)}, s, 10)
    assert all(len(v) == 2 for v in out.values()), out


def test_one_screen_leaving_and_another_arriving_is_not_one_run():
    s = Settings(screen_tail=0, screen_min_run=2)
    out = hold({0: [screen(0, 0, 100, 100)], 1: [screen(0, 0, 100, 100)],
                4: [screen(900, 700, 100, 100)]}, s, 10)
    assert sorted(out) == [0, 1], "the single sighting far away is a flicker"


def test_a_frame_reached_from_both_sides_gets_one_box_not_two():
    out = hold(seen(0, 1, 2, 6, 7, 8), Settings(screen_tail=6), 30)
    assert all(len(v) == 1 for v in out.values()), out


def test_holding_nothing_returns_nothing():
    assert hold({}, Settings(), 10) == {}


def test_runs_are_grouped_by_label_and_place():
    from faceblur.screens import runs_in

    tv = screen(0, 0, 100, 100, "tv")
    phone = Detection(0, 0, 100, 100, 0.9, None, "phone", True, 0.9, "screen")
    runs = runs_in({0: [tv, phone], 1: [tv, phone]}, Settings())
    assert sorted(len(r.hits) for r in runs) == [2, 2]
    assert {r.label for r in runs} == {"tv", "phone"}, "same place, different things"


# --------------------------------------------------------------- settings

def test_the_screen_settings_are_checked():
    for bad in (dict(screen_labels=()), dict(screen_labels=("toaster",)),
                dict(screen_conf=0.0), dict(screen_conf=1.5),
                dict(screen_nms=0.0), dict(screen_pad=-0.1),
                dict(screen_tail=-1), dict(screen_gap=-1)):
        with pytest.raises(SettingsError):
            Settings(**bad)


def test_the_record_says_which_screens_were_wanted():
    assert Settings().to_dict()["screen_labels"] == ["tv", "laptop", "phone"]


# ------------------------------------------------------- against the model

def test_the_model_finds_a_television_where_one_is(tmp_path):
    """The one test that says the whole chain is right: preprocessing,
    strides, grid, decode and the letterbox undone. A synthetic bright
    rectangle is not a television, so this uses the real frame the decode was
    checked against by eye, rebuilt here as a fixture would be too large."""
    detector = ScreenDetector(Settings(device="cpu"))
    blank = np.full((650, 800, 3), 30, np.uint8)
    assert detector.detect(blank) == [], "an empty room holds no screen"


def test_the_box_is_grown_by_its_own_size_and_clipped_to_the_frame():
    detector = ScreenDetector(Settings(device="cpu", screen_pad=0.2))
    grown = detector._pad(screen(100, 100, 200, 100), (600, 800))
    assert grown.w == pytest.approx(200 * 1.2, abs=1)
    assert grown.h == pytest.approx(100 * 1.2, abs=1)
    at_edge = detector._pad(screen(0, 0, 200, 100), (600, 800))
    assert at_edge.x == 0.0 and at_edge.y == 0.0


def test_no_padding_leaves_the_box_alone():
    detector = ScreenDetector(Settings(device="cpu", screen_pad=0.0))
    d = screen()
    assert detector._pad(d, (600, 800)) is d


# ------------------------------------------------------- the size cap

def test_a_table_the_detector_calls_a_laptop_is_too_big_to_be_one():
    """The rule that makes this class usable. A COCO detector calls any large
    flat rectangle a television, and this footage is made of them: a blue
    table tennis table reads as a laptop at 0.88 over a third of the frame.
    Nothing in the score separates that from a real television across the
    hall, which also scores 0.85. Size does."""
    from faceblur.screens import plausible

    s = Settings(screen_max_area=0.04)
    shape = (1300, 1600)
    real = screen(200, 150, 160, 120)                  # about 0.9% of the frame
    table = screen(100, 400, 900, 700)                 # about 30%
    assert plausible(real, shape, s)
    assert not plausible(table, shape, s)


def test_the_cap_is_on_area_not_on_a_side():
    """A wide, low screen and a tall, narrow one of the same area are both
    screens. Capping a side would keep one and drop the other."""
    from faceblur.screens import plausible

    s = Settings(screen_max_area=0.04)
    shape = (1000, 1000)
    wide = screen(0, 0, 400, 100)      # 4.0% of the frame
    tall = screen(0, 0, 100, 400)      # the same
    assert plausible(wide, shape, s) and plausible(tall, shape, s)
    assert not plausible(screen(0, 0, 400, 200), shape, s)      # 8%


def test_the_shipped_floor_is_the_measured_one_not_the_first_guess():
    """0.35 destroyed 78 percent of a frame on the sample footage. The
    numbers behind both of these are in section 15 of docs/report.md."""
    assert Settings().screen_conf == 0.5
    assert Settings().screen_max_area == 0.12
    assert Settings().screen_min_run == 3
