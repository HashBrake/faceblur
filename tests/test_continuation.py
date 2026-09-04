"""Track continuation: weak boxes extend confirmed tracks and nothing else."""
from __future__ import annotations

from faceblur.detect import Detection, weak_candidates
from faceblur.settings import Settings
from faceblur.track import Tracker

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9, lm=LM):
    return Detection(float(x), float(y), float(w), float(h), score, lm)


def test_a_weak_box_never_starts_a_track():
    frames = [[] for _ in range(10)]
    weak = [[det(100, score=0.4)] for _ in range(10)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=0).run(frames, weak)
    assert tracks == []
    assert all(f == [] for f in out)


def test_a_confirmed_track_continues_on_weak_boxes():
    frames = [[] for _ in range(12)]
    weak = [[] for _ in range(12)]
    frames[0] = [det(100)]
    frames[1] = [det(102)]
    for i in range(2, 12):
        weak[i] = [det(100 + 2 * i, score=0.4)]
    out, tracks = Tracker(min_track=2, max_gap=2, tail=0).run(frames, weak)
    assert len(tracks) == 1
    assert len(tracks[0]) == 12
    assert tracks[0].confirmed == 2
    assert all(f for f in out)
    assert out[6][0].source == "continued"


def test_weak_boxes_do_not_count_toward_min_track():
    frames = [[] for _ in range(8)]
    weak = [[] for _ in range(8)]
    frames[0] = [det(100)]
    for i in range(1, 8):
        weak[i] = [det(100, score=0.4)]
    out, tracks = Tracker(min_track=2, max_gap=2, tail=0).run(frames, weak)
    assert tracks == []


def test_a_weak_box_far_from_the_prediction_is_ignored():
    frames = [[] for _ in range(8)]
    weak = [[] for _ in range(8)]
    frames[0] = [det(100)]
    frames[1] = [det(100)]
    for i in range(2, 8):
        weak[i] = [det(600, score=0.4)]       # a hand somewhere else
    out, tracks = Tracker(min_track=2, max_gap=2, tail=0).run(frames, weak)
    assert len(tracks) == 1 and len(tracks[0]) == 2
    assert all(f == [] for f in out[2:])


def test_a_strong_box_wins_over_a_weak_one_for_the_same_track():
    frames = [[] for _ in range(4)]
    weak = [[] for _ in range(4)]
    frames[0] = [det(100)]
    frames[1] = [det(100)]
    frames[2] = [det(101, score=0.95)]
    weak[2] = [det(103, score=0.4)]
    out, tracks = Tracker(min_track=2, max_gap=2, tail=0).run(frames, weak)
    assert tracks[0].confirmed == 3
    assert out[2][0].score == 0.95


def test_weak_candidates_hold_the_two_kinds_of_unconfirmed_box():
    s = Settings(conf=0.6, conf_weak=0.3, verify=True, verify_conf=0.3)
    raw = {
        "yunet": [
            Detection(0, 0, 50, 50, 0.45, None),      # between weak and full threshold
            Detection(200, 200, 50, 50, 0.9, None),   # full threshold, unconfirmed
            Detection(400, 400, 50, 50, 0.9, None),   # full threshold, confirmed
            Detection(600, 600, 50, 50, 0.1, None),   # below weak
        ],
        "centerface": [Detection(402, 401, 50, 50, 0.7, None)],
    }
    weak = weak_candidates(raw, s, (1300, 1600))
    assert sorted(d.x for d in weak) == [0, 200]


def test_conf_weak_cannot_exceed_conf():
    import pytest
    from faceblur.settings import SettingsError
    with pytest.raises(SettingsError):
        Settings(conf=0.5, conf_weak=0.6)
