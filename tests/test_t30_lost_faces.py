"""T30: real faces the T25/T26 rules unmasked on the P0 right-camera footage.

P0_D5: a woman with her head turned, seen only weakly (YuNet 0.5-0.7, never
confirmed), was masked only by the gap walk of a 38 px track beside her, which
masked every box near its path; the walk on the path dropped her. P0_EV: far
office faces of 8.7-11.3 px fell under the 12 px floor, a 9.6 px one peaked at
0.585 under face_small_conf 0.6, and confirmed faces lost their gap masks where
the camera-path box sat beside the face in fast pans.
"""
from __future__ import annotations

import pytest

from faceblur.detect import Detection
from faceblur.pipeline import tracker_for
from faceblur.settings import Settings, SettingsError
from faceblur.track import Track, Tracker


def face(x, y=100.0, w=40.0, conf=0.9, score=0.9):
    return Detection(float(x), float(y), float(w), float(w), score, None, "yunet", True, conf)


def weak_box(x, y=100.0, w=40.0, score=0.6):
    return Detection(float(x), float(y), float(w), float(w), score, None)


def covers(out, k, x, y=100.0):
    return any(d.x <= x <= d.x + d.w and d.y <= y <= d.y + d.h for d in out[k])


def tracker(**kw):
    args = dict(min_track=2, established_after=3, max_gap=2, tail=0, tail_before=0,
                conf_weak=0.4, conf_weak_long=0.3, weak_run=30, sure_conf=0.0,
                walk_on_path=True)
    args.update(kw)
    return Tracker(**args)


# ------------------------------------------- the D5 woman: a weak-only face beside a track

def beside(w):
    """A sure track of width w at x=100 seen at frames 0 and 10; a weak-only
    face 60 px to its right on frames 1-9 (never confirmed, never a track)."""
    n = 11
    t = Track(0, {0: face(100, w=w), 10: face(100, w=w)}, confirmed=5, last_confirmed=10,
              best_conf=0.9)
    weak = [[] for _ in range(n)]
    for k in range(1, 10):
        weak[k] = [weak_box(100 + w + 20, w=w)]
    return t, [[] for _ in range(n)], weak


def walk(tr, w):
    t, frames, weak = beside(w)
    tr._set_shifts(None, len(frames))
    tr._fill_gaps([t], frames, weak)
    return tr.render([t], len(frames)), 100 + w + 20 + w / 2


def test_a_weak_face_beside_a_wide_track_keeps_its_mask():
    out, fx = walk(tracker(path_px=32.0), 40.0)
    assert all(covers(out, k, fx) for k in range(1, 10))


def test_the_walk_on_the_path_alone_drops_it():
    out, fx = walk(tracker(path_px=0.0), 40.0)
    assert not any(covers(out, k, fx) for k in range(1, 10))


def test_a_narrow_track_still_keeps_to_its_path():
    # The 9K pegboard: a 14 px track must not mask the specks beside it.
    out, fx = walk(tracker(path_px=32.0), 14.0)
    assert not any(covers(out, k, fx) for k in range(1, 10))


def test_the_default_settings_keep_the_d5_face():
    s = Settings()
    tr = tracker_for(s)
    assert tr.walk_on_path and tr.path_px == 32.0
    out, fx = walk(tr, 40.0)
    assert all(covers(out, k, fx) for k in range(1, 10))


# ------------------------------------------- a confirmed face through a long gap

def test_a_confirmed_40px_face_is_painted_through_a_20_frame_gap():
    frames = [[face(100)] if i < 10 or 30 <= i < 40 else [] for i in range(40)]
    out, tracks = tracker_for(Settings()).run(frames, [[] for _ in range(40)])
    assert len(tracks) == 1
    assert all(covers(out, k, 120.0, 120.0) for k in range(40))


def test_the_straight_box_is_kept_when_the_camera_shift_is_wrong():
    # The camera "moves" 30 px a frame during the gap but the face did not:
    # the camera-path box leaves the face, the straight box stays on it.
    n = 40
    frames = [[face(100)] if i < 10 or 30 <= i < 40 else [] for i in range(n)]
    shifts = [(0.0, 0.0)] * n
    for k in range(10, 20):
        shifts[k] = (30.0, 0.0)
    for k in range(20, 30):
        shifts[k] = (-30.0, 0.0)
    kw = dict(stitch_gap=45, stitch_slack=8.0, gap_camera=True)
    out, _ = tracker(**kw).run(frames, None, shifts)
    assert not all(covers(out, k, 120.0, 120.0) for k in range(10, 30))
    out, _ = tracker(gap_straight=True, **kw).run(frames, None, shifts)
    assert all(covers(out, k, 120.0, 120.0) for k in range(10, 30))


# ------------------------------------------- the far EV faces: tiny tracks

def tiny_track(confs, w, tid=0):
    dets = {i * 3: face(100 + 50 * tid, w=w, conf=c) for i, c in enumerate(confs)}
    return Track(tid, dets, confirmed=len(confs), last_confirmed=max(dets), best_conf=max(confs))


def drawn(out, tid):
    return sum(1 for f in out for d in f if round(d.x) == 100 + 50 * tid)


def test_a_tiny_track_needs_a_face_like_median_not_a_floor():
    real = tiny_track([0.55, 0.585, 0.5, 0.52], w=9.6)           # P0_EV far desk
    texture = tiny_track([0.45, 0.62, 0.42, 0.47], w=11.3, tid=1)   # 9K pegboard
    larger = tiny_track([0.45, 0.62, 0.42, 0.47], w=13.0, tid=2)    # not tiny: other rules
    tr = Tracker(min_track=2, tail=0, tail_before=0, paint_min_px=8.0, tiny_px=12.0,
                 tiny_median=0.5)
    out = tr.render([real, texture, larger], 20)
    assert drawn(out, 0) > 0 and drawn(out, 1) == 0 and drawn(out, 2) > 0
    # The old 12 px floor took the real face as well.
    out = Tracker(min_track=2, tail=0, tail_before=0, paint_min_px=12.0).render([real], 20)
    assert drawn(out, 0) == 0


def test_the_default_small_rules_keep_the_far_ev_faces():
    s = Settings()
    tr = Tracker(min_track=2, tail=0, tail_before=0, paint_min_px=s.face_paint_min_px,
                 tiny_px=s.face_tiny_px, tiny_median=s.face_tiny_median,
                 small_px=s.face_small_px, small_conf=s.face_small_conf,
                 small_spread=s.face_small_spread)
    peak_0585 = tiny_track([0.55, 0.585, 0.5, 0.52], w=9.6)
    head = tiny_track([0.46, 0.85, 0.46, 0.45, 0.5], w=12.5, tid=1)   # spread 0.39
    out = tr.render([peak_0585, head], 20)
    # Both drawn whole: frames 0-9 and 0-12, the gaps interpolated.
    assert drawn(out, 0) == 10 and drawn(out, 1) == 13


# ------------------------------------------- settings

def test_t30_settings_defaults_and_passing():
    s = Settings()
    assert (s.gap_walk_path_px, s.gap_also_straight, s.face_tiny_px, s.face_tiny_median) == (
        32.0, True, 12.0, 0.5)
    assert (s.face_paint_min_px, s.face_small_conf, s.face_small_spread, s.stitch_slack_frac) == (
        8.0, 0.55, 0.4, 0.5)
    tr = tracker_for(Settings(gap_walk_path_px=0, gap_also_straight=False, face_tiny_px=0,
                              face_tiny_median=0.3))
    assert (tr.path_px, tr.gap_straight, tr.tiny_px, tr.tiny_median) == (0.0, False, 0.0, 0.3)
    d = s.to_dict()
    for key in ("gap_walk_path_px", "gap_also_straight", "face_tiny_px", "face_tiny_median"):
        assert key in d


@pytest.mark.parametrize("changes", [
    {"gap_walk_path_px": -1},
    {"face_tiny_px": -1},
    {"face_tiny_median": 1.5},
    {"gap_also_straight": "yes"},
])
def test_bad_t30_settings_are_refused(changes):
    with pytest.raises(SettingsError):
        Settings(**changes)
