"""Tracking: flicker rejection, gap filling, tails. Plus the old merge helpers."""
from __future__ import annotations

import pytest

from faceblur.detect import Detection
from faceblur.track import Tracker, nms_merge, propagate

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9, lm=LM):
    return Detection(float(x), float(y), float(w), float(h), score, lm)


def test_a_face_seen_once_is_never_masked():
    frames = [[] for _ in range(10)]
    frames[4] = [det(100)]
    out, tracks = Tracker(min_track=3, max_gap=3, tail=2).run(frames)
    assert tracks == []
    assert all(f == [] for f in out)


def test_three_sightings_make_a_track():
    frames = [[] for _ in range(10)]
    for i in (3, 4, 5):
        frames[i] = [det(100 + 2 * i)]
    out, tracks = Tracker(min_track=3, max_gap=3, tail=0).run(frames)
    assert len(tracks) == 1
    assert [i for i, f in enumerate(out) if f] == [3, 4, 5]


def test_gaps_inside_a_track_are_interpolated_not_grown():
    frames = [[] for _ in range(12)]
    frames[2] = [det(100)]
    frames[6] = [det(120)]
    frames[8] = [det(140)]
    out, tracks = Tracker(min_track=3, max_gap=3, tail=0).run(frames)
    assert len(tracks) == 1
    covered = [i for i, f in enumerate(out) if f]
    assert covered == list(range(2, 9))
    mid = out[4][0]
    assert mid.x == pytest.approx(110.0)
    assert mid.w == pytest.approx(40.0)      # no growth
    assert mid.source == "track"
    assert mid.landmarks is not None


def test_a_gap_longer_than_max_gap_splits_the_track():
    frames = [[] for _ in range(20)]
    for i in (0, 1, 2):
        frames[i] = [det(100)]
    for i in (10, 11, 12):
        frames[i] = [det(100)]
    out, tracks = Tracker(min_track=3, max_gap=3, tail=0).run(frames)
    assert len(tracks) == 2
    assert all(out[i] == [] for i in range(3, 10))


def test_tails_extend_the_ends_at_the_same_size():
    frames = [[] for _ in range(12)]
    for i in (5, 6, 7):
        frames[i] = [det(100)]
    out, _ = Tracker(min_track=3, max_gap=3, tail=2).run(frames)
    assert [i for i, f in enumerate(out) if f] == [3, 4, 5, 6, 7, 8, 9]
    assert out[3][0].w == pytest.approx(40.0)
    assert out[3][0].source == "track"


def test_tails_stop_at_the_video_edges():
    frames = [[] for _ in range(4)]
    for i in (0, 1, 2):
        frames[i] = [det(100)]
    out, _ = Tracker(min_track=3, max_gap=3, tail=5).run(frames)
    assert len(out) == 4
    assert all(f for f in out)


def test_two_faces_keep_separate_tracks():
    frames = [[] for _ in range(6)]
    for i in range(6):
        frames[i] = [det(100), det(600)]
    out, tracks = Tracker(min_track=3, max_gap=3, tail=0).run(frames)
    assert len(tracks) == 2
    assert all(len(f) == 2 for f in out)


def test_a_moving_face_is_followed_by_prediction():
    frames = [[] for _ in range(8)]
    for i in range(8):
        frames[i] = [det(100 + 15 * i)]     # moves 15 px per frame, box is 40 px
    out, tracks = Tracker(min_track=3, max_gap=3, tail=0, iou_thr=0.3).run(frames)
    assert len(tracks) == 1
    assert len(tracks[0]) == 8


def test_nms_keeps_one_box_from_a_cluster():
    boxes = [[100, 100, 50, 50, 0.9], [102, 101, 50, 50, 0.7], [101, 99, 48, 52, 0.5]]
    kept = nms_merge(boxes, 0.35)
    assert len(kept) == 1
    assert kept[0][4] == pytest.approx(0.9)


def test_the_old_propagation_still_spreads_boxes():
    """Kept for reference. The pipeline no longer calls it."""
    per_frame = [[] for _ in range(25)]
    per_frame[10] = [[100.0, 100.0, 40.0, 40.0, 0.9]]
    per_frame[14] = [[100.0, 100.0, 40.0, 40.0, 0.9]]
    out = propagate(per_frame, window=6, grow=0.06)
    assert [i for i, b in enumerate(out) if b] == list(range(4, 21))
