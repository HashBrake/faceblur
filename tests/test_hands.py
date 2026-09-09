"""The hand rule: MediaPipe's two models, and what the check does with them.

The rule exists because every face detector in this project calls the wearer's
own hand a face, and a check that reads one frame of the output at a time has
no track-level rule to fall back on. These tests cover the parts that can be
wrong quietly: the anchor layout the palm model's output is decoded against,
the letterbox that decides where a box lands, the rectangle a palm implies,
and the arithmetic of coverage. The two thresholds are exercised against a
stand-in, so the test says what the rule does rather than what the models
happen to think today.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from faceblur.detect import Detection
from faceblur.hands import (LANDMARK_MODEL, PALM_MODEL, PALM_SIZE, RECT_SCALE, Hand,
                            HandLandmarks, HandRule, PalmDetector, anchors, covered_by,
                            decode, letterbox, rect_for)
from faceblur.settings import Settings


def square(x=100.0, y=100.0, size=40.0) -> Detection:
    return Detection(x, y, size, size, 0.9)


def quad(cx, cy, side) -> tuple:
    h = side / 2
    return ((cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h))


# ------------------------------------------------------------------- anchors

def test_the_anchor_layout_is_the_length_the_model_writes():
    """2016 rows: 24x24x2 at stride 8 and 12x12x6 at stride 16.

    This is the one check that the transcription of MediaPipe's anchor
    calculator is the right one. A layout of another length cannot be decoded
    against the model's output at all, and one of the same length built from
    the wrong strides would put every box somewhere else.
    """
    a = anchors()
    assert a.shape == (2016, 2)
    assert 24 * 24 * 2 + 12 * 12 * 6 == 2016


def test_anchor_centres_sit_in_the_middle_of_their_cell():
    a = anchors()
    assert a[0] == pytest.approx([0.5 / 24, 0.5 / 24])
    assert a[1] == pytest.approx([0.5 / 24, 0.5 / 24])      # two per stride 8 cell
    assert a[2] == pytest.approx([1.5 / 24, 0.5 / 24])
    assert a[-1] == pytest.approx([11.5 / 12, 11.5 / 12])
    assert a.min() > 0.0 and a.max() < 1.0


def test_the_stride_16_layers_share_one_feature_map():
    """Three layers of stride 16 stack six anchors on a 12x12 grid, not three
    grids of two. Getting that wrong shifts every box of the second half."""
    a = anchors()
    tail = a[24 * 24 * 2:]
    assert len(tail) == 12 * 12 * 6
    assert np.allclose(tail[:6], tail[0])          # six anchors on the first cell


# ----------------------------------------------------------------- letterbox

def test_the_letterbox_centres_what_it_pads():
    image = np.full((100, 200, 3), 255, np.uint8)
    canvas, f, ox, oy = letterbox(image)
    assert canvas.shape == (PALM_SIZE, PALM_SIZE, 3)
    assert f == pytest.approx(PALM_SIZE / 200)
    assert ox == 0 and oy == pytest.approx((PALM_SIZE - 96) // 2)
    assert canvas[0, 0].sum() == 0                  # padding is black
    assert canvas[PALM_SIZE // 2, PALM_SIZE // 2].sum() > 0


def test_a_square_image_needs_no_padding():
    canvas, f, ox, oy = letterbox(np.zeros((384, 384, 3), np.uint8))
    assert (ox, oy) == (0, 0)
    assert f == pytest.approx(0.5)


# ------------------------------------------------- the rectangle a palm means

def test_the_rectangle_is_square_and_grown_by_mediapipe_s_factor():
    """A palm box 20 wide with the hand pointing straight up: the rectangle is
    the long side times 2.6, squared off, and shifted towards the fingers."""
    corners = rect_for(100.0, 100.0, 20.0, 20.0, wrist=(100.0, 110.0), knuckle=(100.0, 90.0))
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(20.0 * RECT_SCALE)
    assert max(ys) - min(ys) == pytest.approx(20.0 * RECT_SCALE)
    # shift_y is -0.5, so the centre moves half a height towards the knuckles.
    assert (min(ys) + max(ys)) / 2 == pytest.approx(90.0)
    assert (min(xs) + max(xs)) / 2 == pytest.approx(100.0)


def test_the_rectangle_turns_with_the_hand():
    """Wrist to knuckle running left to right: the rectangle is rotated a
    quarter turn, so the shift goes sideways instead of up."""
    corners = rect_for(100.0, 100.0, 20.0, 20.0, wrist=(90.0, 100.0), knuckle=(110.0, 100.0))
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    assert cx == pytest.approx(110.0)
    assert cy == pytest.approx(100.0)


def test_a_rectangle_takes_the_long_side_of_an_oblong_palm():
    corners = rect_for(0.0, 0.0, 10.0, 40.0, wrist=(0.0, 20.0), knuckle=(0.0, -20.0))
    xs = [p[0] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(40.0 * RECT_SCALE)


# ------------------------------------------------------------------- decoding

def _one_box_output(anchor: int, cx: float, cy: float, size: float, score: float):
    """Model output with a single box on one anchor, in the model's own units."""
    boxes = np.zeros((2016, 18), np.float32)
    scores = np.full((2016, 1), -30.0, np.float32)
    a = anchors()[anchor]
    boxes[anchor, 0] = cx - a[0] * PALM_SIZE
    boxes[anchor, 1] = cy - a[1] * PALM_SIZE
    boxes[anchor, 2] = boxes[anchor, 3] = size
    # keypoint 0 (wrist) below the centre, keypoint 2 (knuckle) above it
    boxes[anchor, 4] = cx - a[0] * PALM_SIZE
    boxes[anchor, 5] = cy + size - a[1] * PALM_SIZE
    boxes[anchor, 8] = cx - a[0] * PALM_SIZE
    boxes[anchor, 9] = cy - size - a[1] * PALM_SIZE
    scores[anchor, 0] = math.log(score / (1 - score))
    return boxes, scores


def test_a_decoded_box_lands_where_the_model_put_it():
    boxes, scores = _one_box_output(900, 96.0, 96.0, 24.0, 0.9)
    hands = decode(boxes, scores, conf=0.5, nms=0.3, f=1.0, ox=0, oy=0)
    assert len(hands) == 1
    x, y, w, h = hands[0].palm
    assert (x + w / 2, y + h / 2) == pytest.approx((96.0, 96.0))
    assert (w, h) == pytest.approx((24.0, 24.0))
    assert hands[0].score == pytest.approx(0.9, abs=1e-3)


def test_the_letterbox_is_undone_on_the_way_out():
    """Half scale with a 20 px top border: a box at 96, 96 in the model's
    picture is at 192, 152 in the frame."""
    boxes, scores = _one_box_output(900, 96.0, 96.0, 24.0, 0.9)
    hands = decode(boxes, scores, conf=0.5, nms=0.3, f=0.5, ox=0, oy=20)
    x, y, w, h = hands[0].palm
    assert (x + w / 2, y + h / 2) == pytest.approx((192.0, 152.0))
    assert (w, h) == pytest.approx((48.0, 48.0))


def test_nothing_above_the_threshold_means_no_hands():
    boxes, scores = _one_box_output(900, 96.0, 96.0, 24.0, 0.4)
    assert decode(boxes, scores, conf=0.5, nms=0.3, f=1.0, ox=0, oy=0) == []


# ------------------------------------------------------------------- coverage

def test_coverage_is_the_share_of_the_box_a_hand_sits_on():
    det = square(100, 100, 40)          # x, y, w, h: 100,100 to 140,140
    whole = Hand(quad(120, 120, 200), 0.9, (0, 0, 1, 1))
    assert covered_by(det, [whole]) == pytest.approx(1.0)
    half = Hand(quad(120, 110, 40), 0.9, (0, 0, 1, 1))   # covers y 90 to 130
    assert covered_by(det, [half]) == pytest.approx(0.75, abs=0.03)
    away = Hand(quad(500, 500, 40), 0.9, (0, 0, 1, 1))
    assert covered_by(det, [away]) == 0.0
    assert covered_by(det, []) == 0.0


def test_two_hands_over_the_same_box_are_not_counted_twice():
    det = square(100, 100, 40)
    left = Hand(quad(105, 120, 20), 0.9, (0, 0, 1, 1))
    same = Hand(quad(105, 120, 20), 0.9, (0, 0, 1, 1))
    assert covered_by(det, [left, same]) == pytest.approx(covered_by(det, [left]))


def test_a_rotated_hand_is_measured_as_the_polygon_it_is():
    """The quad's upright bounds take in a lot of room a diagonal hand does
    not occupy, and coverage has to read the polygon or it overstates."""
    det = square(0, 0, 100)
    # A diamond inscribed in the box: half its area, but its upright bounds
    # are the whole of it.
    diamond = Hand(((50.0, 0.0), (100.0, 50.0), (50.0, 100.0), (0.0, 50.0)),
                   0.9, (0, 0, 1, 1))
    assert covered_by(det, [diamond]) == pytest.approx(0.5, abs=0.02)
    assert diamond.bounds() == (0, 0, 100, 100)


def test_a_hand_can_be_regrown_about_its_centre():
    hand = Hand(quad(100, 100, 26), 0.9, (90, 90, 10, 10))
    smaller = hand.grown(1.0)
    xs = [p[0] for p in smaller.quad]
    assert max(xs) - min(xs) == pytest.approx(10.0)
    assert sum(p[0] for p in smaller.quad) / 4 == pytest.approx(100.0)


# ------------------------------------------------------------------ the rule

class _StubPalm:
    """A palm detector that answers from a script, one entry per window."""

    name = "palm"
    model_path = PALM_MODEL
    provider = "stub"

    def __init__(self, per_call):
        self.per_call = list(per_call)
        self.calls = []

    def detect(self, crop):
        self.calls.append(crop.shape[:2])
        return self.per_call.pop(0) if self.per_call else []


class _StubLandmarks:
    """A landmark model that returns whatever presence it was given."""

    name = "hand_landmark"
    model_path = LANDMARK_MODEL
    provider = "stub"

    def __init__(self, value):
        self.value = value

    def presence(self, crop, hand):
        return self.value


def rule_with(palm, landmarks, **changes) -> HandRule:
    rule = HandRule.__new__(HandRule)
    rule.settings = Settings(**changes)
    rule.palm = palm
    rule.landmarks = landmarks
    return rule


def test_a_confirmed_hand_over_the_box_is_a_hand():
    hand = Hand(quad(120, 120, 100), 0.9, (100, 100, 20, 20))
    rule = rule_with(_StubPalm([[hand], [hand]]), _StubLandmarks(0.95))
    assert rule.hand_cover(np.zeros((900, 900, 3), np.uint8), square(100, 100, 40)) == 1.0


def test_a_palm_the_landmark_model_will_not_confirm_is_not_a_hand():
    """The whole point of the second model: the palm detector puts a box on
    about one crop in five that holds no hand, and a spurious one is 2.6 times
    the size of what it found, so it swallows whatever face is nearby."""
    hand = Hand(quad(120, 120, 100), 0.9, (100, 100, 20, 20))
    rule = rule_with(_StubPalm([[hand], [hand]]), _StubLandmarks(0.3))
    assert rule.hand_cover(np.zeros((900, 900, 3), np.uint8), square(100, 100, 40)) == 0.0


def test_the_presence_threshold_is_the_setting_that_decides():
    hand = Hand(quad(120, 120, 100), 0.9, (100, 100, 20, 20))
    frame = np.zeros((900, 900, 3), np.uint8)
    loose = rule_with(_StubPalm([[hand], [hand]]), _StubLandmarks(0.55), hand_presence=0.5)
    assert loose.hand_cover(frame, square(100, 100, 40)) == 1.0
    strict = rule_with(_StubPalm([[hand], [hand]]), _StubLandmarks(0.55), hand_presence=0.9)
    assert strict.hand_cover(frame, square(100, 100, 40)) == 0.0


def test_one_window_seeing_a_hand_is_enough():
    """The windows are pooled, not made to agree: requiring both takes the
    hands set aside on the sample footage from 13 to 8 and removes no face,
    because the confirmation has already taken out what agreement was there
    to take."""
    hand = Hand(quad(120, 120, 100), 0.9, (100, 100, 20, 20))
    rule = rule_with(_StubPalm([[], [hand]]), _StubLandmarks(0.95))
    assert rule.hand_cover(np.zeros((900, 900, 3), np.uint8), square(100, 100, 40)) == 1.0


def test_the_windows_are_fixed_pixels_not_a_multiple_of_the_box():
    """Scaling the crop with the box was tried first and is unstable: what the
    palm detector makes of a crop depends on how large a hand is inside it,
    and a multiple of the box holds that constant only if the box is the hand.
    So a 20 px box and a 200 px box get the same two windows."""
    frame = np.zeros((1300, 1600, 3), np.uint8)
    for size in (20.0, 200.0):
        palm = _StubPalm([[], []])
        rule_with(palm, _StubLandmarks(0.0)).hand_cover(frame, square(700, 600, size))
        assert palm.calls == [(384, 384), (512, 512)]


def test_a_window_is_clipped_at_the_frame_edge():
    frame = np.zeros((300, 300, 3), np.uint8)
    palm = _StubPalm([[], []])
    rule_with(palm, _StubLandmarks(0.0)).hand_cover(frame, square(0, 0, 40))
    assert palm.calls == [(212, 212), (276, 276)]           # 20 + half the window


def test_hands_come_back_in_frame_coordinates():
    """Each window is cropped out of the frame, so what the models say about
    the crop has to be put back before it can be compared with a box."""
    hand = Hand(quad(200, 200, 40), 0.9, (180, 180, 40, 40))
    rule = rule_with(_StubPalm([[hand]]), _StubLandmarks(0.95), hand_windows=(384,))
    found = rule.hands_near(np.zeros((1300, 1600, 3), np.uint8), square(700, 600, 40))
    assert len(found) == 1
    cx = sum(p[0] for p in found[0].quad) / 4
    cy = sum(p[1] for p in found[0].quad) / 4
    assert (cx, cy) == pytest.approx((720 - 192 + 200, 620 - 192 + 200))
    assert found[0].presence == pytest.approx(0.95)


# ------------------------------------------------------- against the real models

def test_the_landmark_model_gives_a_probability_not_a_logit():
    """It is used without a sigmoid. If that were wrong, black and a hand
    would both land between 0.5 and 0.73 and there would be nothing to
    threshold; instead an empty crop reads near zero."""
    lm = HandLandmarks(device="cpu")
    hand = Hand(quad(112, 112, 200), 0.9, (60, 60, 100, 100))
    for frame in (np.zeros((224, 224, 3), np.uint8),
                  np.full((224, 224, 3), 200, np.uint8)):
        assert lm.presence(frame, hand) < 0.1


def test_the_palm_detector_finds_nothing_in_an_empty_frame():
    palm = PalmDetector(device="cpu")
    assert palm.detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_the_rule_does_not_call_a_face_a_hand(lena):
    """A photograph of a face, at the size the check sees one. The rule is
    tuned so that this never happens: on the 108 boxes the check found in the
    four sample files it set aside 13 of the 20 hands and none of the 52
    faces."""
    import cv2

    face = cv2.imread(str(lena))
    frame = cv2.resize(face, (600, 600))
    rule = HandRule(Settings(device="cpu"))
    assert rule.hand_cover(frame, square(200, 180, 220)) < Settings().hand_cover
