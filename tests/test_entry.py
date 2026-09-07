"""Faces entering the picture: backward continuation and motion tails."""
from __future__ import annotations

import numpy as np
import pytest

from faceblur.detect import Detection
from faceblur.redact import FaceEllipse, build_alpha, ellipse_for, redact
from faceblur.settings import Settings
from faceblur.track import Tracker

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9):
    return Detection(float(x), float(y), float(w), float(h), score, LM)


def test_weak_sightings_before_confirmation_are_covered():
    """A face seen weakly for two frames, then confirmed, is masked from the
    first weak frame, and the tail reaches back before that."""
    frames = [[] for _ in range(12)]
    weak = [[] for _ in range(12)]
    weak[3] = [det(90, score=0.45)]
    weak[4] = [det(95, score=0.5)]
    frames[5] = [det(100)]
    frames[6] = [det(105)]
    out, tracks = Tracker(established_after=2, min_track=2, max_gap=2, tail=2, tail_before=6).run(frames, weak)
    assert len(tracks) == 1
    assert out[3][0].source == "continued" and out[4][0].source == "continued"
    assert all(out[i] for i in range(0, 9))        # 3 - 6 tail reaches frame 0
    assert out[9] == [] and out[10] == []


def test_a_lone_confirmed_sighting_joins_the_track_that_follows():
    frames = [[] for _ in range(10)]
    frames[3] = [det(100)]                 # alone: too short for a track of its own
    frames[6] = [det(100)]
    frames[7] = [det(100)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=0, tail_before=0).run(frames)
    assert len(tracks) == 1
    assert 3 in tracks[0].dets
    assert all(out[i] for i in range(3, 8))


def test_backward_walk_stops_at_the_gap_limit():
    frames = [[] for _ in range(12)]
    weak = [[] for _ in range(12)]
    weak[1] = [det(100, score=0.5)]        # too far back to reach with max_gap 2
    frames[6] = [det(100)]
    frames[7] = [det(100)]
    out, tracks = Tracker(min_track=2, max_gap=2, tail=0, tail_before=0).run(frames, weak)
    assert 1 not in tracks[0].dets
    assert out[1] == []


def test_the_head_tail_follows_the_motion_backwards_and_grows():
    frames = [[] for _ in range(12)]
    frames[6] = [det(100)]
    frames[7] = [det(110)]                 # moving right at 10 px per frame
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=0, tail_before=3, tail_grow=0.1).run(frames)
    three_back = out[3][0]
    assert three_back.cx == pytest.approx(120 - 30)        # centre 120 at frame 6, back 30
    assert three_back.w == pytest.approx(40 * 1.3)
    assert three_back.source == "track"


def test_the_foot_tail_follows_the_motion_forwards():
    frames = [[] for _ in range(12)]
    frames[3] = [det(100)]
    frames[4] = [det(110)]
    out, _ = Tracker(established_after=2, min_track=2, max_gap=2, tail=2, tail_before=0).run(frames)
    assert out[6][0].cx == pytest.approx(130 + 20)


def test_backward_boxes_are_claimed_once():
    """Two tracks cannot both absorb the same earlier box."""
    frames = [[] for _ in range(10)]
    weak = [[] for _ in range(10)]
    weak[2] = [det(100, score=0.5)]
    frames[3] = [det(100)]
    frames[4] = [det(100)]
    frames[3].append(det(600))
    frames[4].append(det(600))
    out, tracks = Tracker(established_after=2, min_track=2, max_gap=2, tail=0, tail_before=0).run(frames, weak)
    owners = [t for t in tracks if 2 in t.dets]
    assert len(owners) == 1


def test_roi_blend_matches_a_whole_frame_blend():
    """Blending only inside the ellipses' box gives the same pixels."""
    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    s = Settings()
    faces = [Detection(100.0, 100.0, 80.0, 90.0, 0.9, LM), Detection(400.0, 300.0, 50.0, 60.0, 0.8, None)]
    out, alpha = redact(frame, faces, s)

    from faceblur.redact import _cover_region
    cover = frame.copy()
    for d in faces:
        _cover_region(frame, cover, ellipse_for(d, s), s.mode, s.strength, s.feather)
    a = alpha[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - a) + cover.astype(np.float32) * a
    reference = np.where(a > 0, np.clip(blended + 0.5, 0, 255).astype(np.uint8), frame)
    assert np.array_equal(out, reference)


def test_roi_blend_leaves_the_rest_of_the_frame_untouched():
    rng = np.random.default_rng(6)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out, alpha = redact(frame, [Detection(100.0, 100.0, 80.0, 90.0, 0.9, LM)], Settings())
    assert np.array_equal(out[alpha == 0], frame[alpha == 0])
    assert not np.array_equal(out[alpha >= 0.5], frame[alpha >= 0.5])
