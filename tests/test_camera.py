"""The tracker with the camera's movement taken into account."""
from __future__ import annotations

import pytest

from faceblur.detect import Detection
from faceblur.track import Tracker

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9):
    return Detection(float(x), float(y), float(w), float(h), score, LM)


def still(n):
    return [(0.0, 0.0)] * n


def test_a_pan_that_breaks_overlap_keeps_one_track_with_compensation():
    """The camera whips 60 px a frame; a 40 px face has no overlap between
    frames. With the shift known, it stays one track."""
    frames = [[det(100 + 60 * i)] for i in range(6)]
    shifts = [(0.0, 0.0)] + [(60.0, 0.0)] * 5
    _, plain = Tracker(min_track=2, max_gap=2, tail=0).run(frames)
    _, comp = Tracker(min_track=2, max_gap=2, tail=0).run(frames, shifts=shifts)
    assert not any(len(t) == 6 for t in plain)
    assert len(comp) == 1 and len(comp[0]) == 6


def test_the_face_own_motion_is_separated_from_the_camera():
    """Face drifts 5 px a frame against a 40 px camera pan. The prediction for
    the next frame lands on the face, not on where the camera alone puts it."""
    frames = [[det(100 + 45 * i)] for i in range(4)] + [[]] + [[det(100 + 45 * 5)]]
    shifts = [(0.0, 0.0)] + [(40.0, 0.0)] * 5
    _, tracks = Tracker(min_track=2, max_gap=2, tail=0).run(frames, shifts=shifts)
    assert len(tracks) == 1 and 5 in tracks[0].dets


def test_centre_distance_links_a_small_fast_face():
    """A 24 px face moving 20 px a frame has no overlap frame to frame. The
    distance gate links it; without the gate the track breaks."""
    frames = [[det(100 + 20 * i, w=24, h=24)] for i in range(5)]
    _, without = Tracker(min_track=2, max_gap=2, tail=0).run(frames)
    _, with_gate = Tracker(min_track=2, max_gap=2, tail=0, link_dist=1.0).run(frames)
    assert not any(len(t) == 5 for t in without)
    assert len(with_gate) == 1 and len(with_gate[0]) == 5


def test_distance_gate_never_links_boxes_of_very_different_size():
    frames = [[det(100, w=24, h=24)], [det(110, w=90, h=90)], [det(120, w=24, h=24)]]
    _, tracks = Tracker(min_track=1, max_gap=2, tail=0, link_dist=1.0).run(frames)
    sizes = {round(d.w) for t in tracks for d in t.dets.values() if len(t) > 1}
    assert 90 not in sizes


def test_a_longer_gap_is_allowed_while_the_camera_moves_fast():
    frames = [[det(100)], [det(100)]] + [[]] * 6 + [[det(100 + 7 * 30)]]
    shifts = [(0.0, 0.0), (0.0, 0.0)] + [(30.0, 0.0)] * 7
    _, slow = Tracker(min_track=2, max_gap=3, tail=0).run(frames, shifts=shifts)
    _, fast = Tracker(min_track=2, max_gap=3, tail=0, fast_shift=10.0,
                      max_gap_fast=8).run(frames, shifts=shifts)
    assert len(slow[0]) == 2
    assert len(fast) == 1 and 8 in fast[0].dets


def test_the_long_gap_is_not_allowed_when_the_camera_is_still():
    frames = [[det(100)], [det(100)]] + [[]] * 6 + [[det(100)]]
    _, tracks = Tracker(min_track=2, max_gap=3, tail=0, fast_shift=10.0,
                        max_gap_fast=8).run(frames, shifts=still(9))
    assert 8 not in tracks[0].dets


def test_entry_tail_reaches_back_until_the_face_leaves_the_picture():
    """A face moving right at 30 px a frame, first seen at x=100: it entered
    from the left about four frames earlier. The tail stops there, not at
    the fixed count."""
    frames = [[] for _ in range(20)]
    frames[12] = [det(100)]
    frames[13] = [det(130)]
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=0, tail_before=2, link_dist=1.0,
                     tail_before_max=12).run(frames, shape=(650, 800))
    covered = [i for i in range(12) if out[i]]
    assert covered == [8, 9, 10, 11]       # centre 120 crosses x=0 four frames back
    assert out[7] == []


def test_a_still_face_keeps_the_short_entry_tail():
    frames = [[] for _ in range(20)]
    frames[12] = [det(300)]
    frames[13] = [det(300)]
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=0, tail_before=2,
                     tail_before_max=12).run(frames, shape=(650, 800))
    assert [i for i in range(12) if out[i]] == [10, 11]


def test_a_face_that_was_shrinking_gets_a_larger_tail_before():
    """Seen at 100 px then 90 px: it was larger still before the first
    sighting, and the tail follows that trend on top of the safety growth."""
    frames = [[] for _ in range(10)]
    frames[5] = [det(100, w=100, h=100)]
    frames[6] = [det(100, w=90, h=90)]
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=0, tail_before=3, tail_grow=0.0).run(frames)
    assert out[2][0].w == pytest.approx(100 * 1.3)      # 10 percent a frame, three frames back


def test_tails_follow_the_camera():
    frames = [[] for _ in range(10)]
    frames[5] = [det(300)]
    frames[6] = [det(300)]
    shifts = [(0.0, 0.0)] * 7 + [(-25.0, 0.0)] * 3     # shift i: from frame i-1 to i
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=3, tail_before=0).run(frames, shifts=shifts)
    assert out[9][0].cx == pytest.approx(320 - 75)


def test_no_shifts_behaves_like_before():
    frames = [[det(100 + 5 * i)] for i in range(5)]
    a, _ = Tracker(min_track=2, max_gap=2, tail=1).run(frames)
    b, _ = Tracker(min_track=2, max_gap=2, tail=1).run(frames, shifts=still(5))
    assert [[d.box() for d in f] for f in a] == [[d.box() for d in f] for f in b]
