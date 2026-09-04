"""Box merging and temporal propagation."""
from __future__ import annotations

import pytest

from faceblur.settings import Settings
from faceblur.track import nms_merge, propagate


def test_nms_keeps_one_box_from_a_cluster():
    boxes = [[100, 100, 50, 50, 0.9], [102, 101, 50, 50, 0.7], [101, 99, 48, 52, 0.5]]
    kept = nms_merge(boxes, 0.35)
    assert len(kept) == 1
    assert kept[0][4] == pytest.approx(0.9)


def test_nms_keeps_boxes_that_do_not_overlap():
    boxes = [[0, 0, 50, 50, 0.9], [400, 400, 50, 50, 0.8]]
    assert len(nms_merge(boxes, 0.35)) == 2


def test_nms_on_nothing_returns_nothing():
    assert nms_merge([], 0.35) == []


def test_propagation_covers_the_gap_and_the_edges():
    """Phase 1 verify: detections on frames 10 and 14, persist 6, cover 4 to 20."""
    per_frame = [[] for _ in range(25)]
    box = [100.0, 100.0, 40.0, 40.0, 0.9]
    per_frame[10] = [list(box)]
    per_frame[14] = [list(box)]

    out = propagate(per_frame, window=6, grow=0.06, nms_thr=0.60)

    covered = [i for i, boxes in enumerate(out) if boxes]
    assert covered == list(range(4, 21))


def test_propagation_grows_the_copied_box():
    per_frame = [[] for _ in range(9)]
    per_frame[4] = [[100.0, 100.0, 40.0, 40.0, 0.9]]
    out = propagate(per_frame, window=4, grow=0.10, nms_thr=0.60)

    at_source = out[4][0]
    four_away = out[0][0]
    assert at_source[2] == pytest.approx(40.0)
    # Four frames away at 10 percent per frame is 40 percent larger.
    assert four_away[2] == pytest.approx(40.0 * 1.4)
    # The box grows around its centre, so the centre does not move.
    assert at_source[0] + at_source[2] / 2 == pytest.approx(four_away[0] + four_away[2] / 2)


def test_propagation_stays_inside_the_sequence():
    per_frame = [[] for _ in range(5)]
    per_frame[0] = [[10.0, 10.0, 20.0, 20.0, 0.9]]
    out = propagate(per_frame, window=6, grow=0.06, nms_thr=0.60)
    assert len(out) == 5
    assert all(boxes for boxes in out)


def test_propagation_of_nothing_changes_nothing():
    per_frame = [[] for _ in range(5)]
    assert propagate(per_frame, 6, 0.06, 0.60) == [[], [], [], [], []]


def test_window_never_falls_below_stride():
    assert Settings(persist=2, stride=5).window == 5
    assert Settings(persist=9, stride=2).window == 9
