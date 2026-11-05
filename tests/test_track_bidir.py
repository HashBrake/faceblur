"""The tracker reaches forward past a track's last sighting the way it reaches back."""
from __future__ import annotations

from faceblur.detect import Detection
from faceblur.track import Track, Tracker

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9):
    return Detection(float(x), float(y), float(w), float(h), score, LM)


def tracker(**kw):
    args = dict(min_track=2, established_after=3, max_gap=2, tail=0, tail_before=0,
                conf_weak=0.4, conf_weak_long=0.3, weak_run=30, sure_conf=0.0)
    args.update(kw)
    return Tracker(**args)


def confirmed_then_weak(n, weak_frames, x=100):
    frames = [[] for _ in range(n)]
    weak = [[] for _ in range(n)]
    for i in range(3, 8):
        frames[i] = [det(x)]
    for i in weak_frames:
        weak[i] = [det(x, score=0.35)]
    return frames, weak


def covered(out):
    return [i for i, f in enumerate(out) if f]


def test_an_established_track_reaches_forward_over_weak_boxes():
    frames, weak = confirmed_then_weak(15, (8, 9, 10))
    out, tracks = tracker().run(frames, weak)
    assert len(tracks) == 1
    assert covered(out) == list(range(3, 11))
    assert all(out[i][0].source == "continued" for i in (8, 9, 10))
    assert all(out[i] == [] for i in range(11, 15))


def test_a_track_not_established_does_not_reach_forward():
    frames, weak = confirmed_then_weak(15, (8, 9, 10))
    out, tracks = tracker(established_after=10).run(frames, weak)
    assert max(tracks[0].dets) == 7
    assert covered(out) == list(range(3, 8))


def test_the_forward_walk_stops_at_the_gap_limit():
    frames, weak = confirmed_then_weak(15, (8, 11))
    out, tracks = tracker(max_gap=1).run(frames, weak)
    assert 8 in tracks[0].dets
    assert 11 not in tracks[0].dets
    assert out[11] == []


def test_weak_run_caps_the_forward_walk():
    frames, weak = confirmed_then_weak(15, range(8, 13))
    out, tracks = tracker(weak_run=2).run(frames, weak)
    assert max(tracks[0].dets) == 9
    assert covered(out) == list(range(3, 10))


def test_the_walk_takes_a_box_the_velocity_prediction_missed():
    """A face moving right that stops dead as it leaves: linking predicted it
    further on and lost it; the walk from where it was last seen finds it."""
    frames = [[] for _ in range(12)]
    weak = [[] for _ in range(12)]
    for i in range(3, 8):
        frames[i] = [det(100 + 20 * (i - 3))]       # 100 .. 180
    weak[8] = [det(172, score=0.35)]
    linked_only = tracker()
    linked_only._extend_forwards = lambda *a: None
    _, before = linked_only.run(frames, weak)
    assert 8 not in before[0].dets
    out, tracks = tracker().run(frames, weak)
    assert tracks[0].dets[8].source == "continued"
    assert out[8]


def test_a_box_claimed_by_another_track_is_not_taken():
    """Track A runs right and stops; track B starts where A stopped. Linking
    keeps them apart; A's forward walk must not take B's confirmed box."""
    frames = [[] for _ in range(14)]
    for i in range(3, 8):
        frames[i] = [det(100 + 20 * (i - 3))]       # A: 100 .. 180
    for i in (8, 9, 10):
        frames[i] = [det(172)]                       # B
    weak = [[] for _ in range(14)]
    _, tracks = tracker().run(frames, weak)
    assert len(tracks) == 2
    a, b = sorted(tracks, key=lambda t: t.first)
    assert max(a.dets) == 7
    assert b.first == 8 and b.dets[8].source != "continued"


def walk(shifts):
    """Run only the forward walk on a hand-built track, so linking (which
    knows the camera too) cannot take the boxes first."""
    n = 12
    frames = [[] for _ in range(n)]
    weak = [[] for _ in range(n)]
    t = Track(0, {i: det(100 + 30 * i) for i in range(3, 8)}, confirmed=5, last_confirmed=7,
              best_conf=0.9)
    for i in (8, 9, 10):
        weak[i] = [det(100 + 30 * i, score=0.35)]   # where the camera puts the face
    tr = tracker()
    tr._set_shifts(shifts, n)
    tr._extend_forwards([t], frames, weak)
    return t


def test_the_forward_walk_follows_the_camera():
    t = walk([(30.0, 0.0)] * 12)
    assert sorted(t.dets) == list(range(3, 11))
    assert all(t.dets[i].source == "continued" for i in (8, 9, 10))


def test_without_the_camera_the_boxes_are_out_of_reach():
    t = walk(None)
    assert max(t.dets) == 7
