"""Tracks joined across a long loss, established tracks, blur-sized masks,
and two detectors agreeing below the threshold."""
from __future__ import annotations

import pytest

from faceblur.detect import Detection, filter_candidates, weak_candidates
from faceblur.settings import Settings
from faceblur.track import Tracker

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=40.0, score=0.9):
    return Detection(float(x), float(y), float(w), float(h), score, LM)


def test_a_face_lost_for_a_second_is_one_track_and_the_gap_is_masked():
    """Seen for frames 0-4, lost for 25 frames, seen again 30-34 in the same
    place: joined, and every frame between carries a mask."""
    frames = [[] for _ in range(40)]
    for i in list(range(5)) + list(range(30, 35)):
        frames[i] = [det(300)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=45).run(frames)
    assert len(tracks) == 1
    assert all(out[i] for i in range(0, 35))
    _, apart = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=0).run(frames)
    assert len(apart) == 2


def test_stitching_follows_the_face_and_the_camera():
    """The face drifts 2 px a frame and the camera pans 10 px a frame during
    the loss; the second piece sits where both put it."""
    frames = [[] for _ in range(40)]
    shifts = [(0.0, 0.0)] * 5 + [(10.0, 0.0)] * 25 + [(0.0, 0.0)] * 10
    for i in range(5):
        frames[i] = [det(300 + 2 * i)]
    for i in range(30, 35):
        frames[i] = [det(300 + 2 * i + 250)]        # own 2 px a frame plus 25 x 10 px of camera
    _, tracks = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=45).run(frames, shifts=shifts)
    assert len(tracks) == 1 and len(tracks[0]) == 10


def test_no_stitch_when_the_second_piece_is_elsewhere():
    frames = [[] for _ in range(40)]
    for i in range(5):
        frames[i] = [det(300)]
    for i in range(30, 35):
        frames[i] = [det(900)]
    _, tracks = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=45).run(frames)
    assert len(tracks) == 2


def test_no_stitch_beyond_the_gap():
    frames = [[] for _ in range(80)]
    for i in list(range(5)) + list(range(70, 75)):
        frames[i] = [det(300)]
    _, tracks = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=45).run(frames)
    assert len(tracks) == 2


def test_a_single_sighting_joins_an_established_track_by_stitching():
    frames = [[] for _ in range(30)]
    for i in range(5):
        frames[i] = [det(300)]
    frames[20] = [det(300)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=0, stitch_gap=45).run(frames)
    assert len(tracks) == 1 and 20 in tracks[0].dets
    assert all(out[i] for i in range(0, 21))


def test_only_an_established_track_takes_the_lowest_weak_boxes():
    """Weak boxes at 0.35 sit between conf_weak_long (0.3) and conf_weak (0.4).
    A track with two confirmations cannot use them; one with three can."""
    frames = [[] for _ in range(12)]
    weak = [[] for _ in range(12)]
    frames[0] = [det(300)]
    frames[1] = [det(300)]
    for i in range(2, 8):
        weak[i] = [det(300, score=0.35)]
    kw = dict(min_track=2, max_gap=3, tail=0, conf_weak=0.4, conf_weak_long=0.3, established_after=3)
    out, tracks = Tracker(**kw).run(frames, weak)
    assert 7 not in tracks[0].dets
    frames[2] = [det(300)]                       # a third confirmation
    out, tracks = Tracker(**kw).run(frames, weak)
    assert 7 in tracks[0].dets and all(out[i] for i in range(8))


def test_only_an_established_track_gets_an_exit_tail():
    frames = [[] for _ in range(20)]
    for i in range(2):
        frames[i] = [det(300)]
    out, _ = Tracker(min_track=2, max_gap=2, tail=2, tail_long=5, established_after=3).run(frames)
    assert [i for i in range(2, 20) if out[i]] == []        # not established: no tail at all
    frames[2] = [det(300)]
    out, _ = Tracker(min_track=2, max_gap=2, tail=2, tail_long=5, established_after=3).run(frames)
    assert [i for i in range(3, 20) if out[i]] == [3, 4, 5, 6, 7]


def test_masks_grow_by_the_camera_shift_while_it_moves_fast():
    shifts = [(0.0, 0.0)] * 3 + [(30.0, 0.0)] * 3
    frames = [[det(300 + 30 * max(0, i - 2))] for i in range(6)]      # the face rides the pan
    out, _ = Tracker(min_track=2, max_gap=2, tail=0, blur_shift=8.0).run(frames, shifts=shifts)
    assert out[1][0].w == pytest.approx(40)
    assert out[4][0].w == pytest.approx(70) and out[4][0].cx == pytest.approx(380)
    still, _ = Tracker(min_track=2, max_gap=2, tail=0, blur_shift=0.0).run(frames, shifts=shifts)
    assert still[4][0].w == pytest.approx(40)


def test_two_detectors_agreeing_make_a_face_below_the_threshold():
    s = Settings(conf=0.6, conf_agree=0.4, verify_conf_agree=0.5, agree_max_px=48)
    y = Detection(100.0, 100.0, 30.0, 30.0, 0.5, LM)
    raw = {"yunet": [y], "centerface": [Detection(100.0, 100.0, 30.0, 30.0, 0.6, None, "centerface")],
           "centerface_flip": [], "centerface_tight": [], "ultraface": []}
    strong = filter_candidates(raw, s, (1300, 1600))
    assert len(strong) == 1 and strong[0].verified
    raw["centerface"][0] = Detection(100.0, 100.0, 30.0, 30.0, 0.4, None, "centerface")
    assert filter_candidates(raw, s, (1300, 1600)) == []
    # A box over agree_max_px never takes this road: hands are large.
    big = {"yunet": [Detection(100.0, 100.0, 120.0, 120.0, 0.5, LM)],
           "centerface": [Detection(100.0, 100.0, 120.0, 120.0, 0.9, None, "centerface")],
           "centerface_flip": [], "centerface_tight": [], "ultraface": []}
    assert filter_candidates(big, s, (1300, 1600)) == []


def test_weak_boxes_reach_down_to_the_long_floor():
    s = Settings(conf_weak=0.4, conf_weak_long=0.3)
    raw = {"yunet": [Detection(100.0, 100.0, 30.0, 30.0, 0.32, LM)], "centerface": [],
           "centerface_flip": [], "centerface_tight": [], "ultraface": []}
    assert len(weak_candidates(raw, s, (1300, 1600))) == 1
    assert weak_candidates(raw, Settings(conf_weak=0.4, conf_weak_long=0.4), (1300, 1600)) == []


def test_continuation_needs_continue_after_confirmations_and_ends_after_weak_run():
    """Two confirmations and forty weak frames: with continue_after 3 nothing
    continues; with 2 and weak_run 10 the continuation stops ten frames
    after the last confirmed detection."""
    frames = [[] for _ in range(50)]
    weak = [[] for _ in range(50)]
    frames[0] = [det(300)]
    frames[1] = [det(300)]
    for i in range(2, 42):
        weak[i] = [det(300, score=0.5)]
    _, strict = Tracker(min_track=2, max_gap=3, tail=0, conf_weak=0.4,
                        continue_after=3).run(frames, weak)
    assert max(strict[0].dets) == 1
    _, limited = Tracker(min_track=2, max_gap=3, tail=0, conf_weak=0.4,
                         continue_after=2, weak_run=10).run(frames, weak)
    assert max(limited[0].dets) == 11
    _, free = Tracker(min_track=2, max_gap=3, tail=0, conf_weak=0.4,
                      continue_after=2, weak_run=0).run(frames, weak)
    assert max(free[0].dets) == 41


def test_a_stitched_gap_follows_the_weak_boxes_the_detector_saw():
    """Sightings at x=300 (frames 0-2) and x=450 (frames 20-22), nothing the
    tracker could continue on for five frames either side, and the smeared
    face seen weakly along an arc in the middle. The two pieces stitch, and
    the gap masks sit on those weak boxes, not on the straight line."""
    frames = [[] for _ in range(30)]
    weak = [[] for _ in range(30)]
    for i in range(3):
        frames[i] = [det(300)]
    for i in range(20, 23):
        frames[i] = [det(450)]
    for k in range(8, 15):
        x = 300 + (k - 2) * 150 / 18
        y = 100 - 60 * (1 - abs(k - 11) / 9)                 # an arc, up to 60 px above the line
        weak[k] = [det(x, y=y, w=52, h=52, score=0.33)]
    kw = dict(min_track=2, max_gap=3, tail=0, stitch_gap=45, stitch_slack=8.0, conf_weak=0.4,
              established_after=3)
    out, tracks = Tracker(conf_weak_long=0.3, **kw).run(frames, weak)
    assert len(tracks) == 1
    # The interpolated box on the line, and the weak box on the arc, both masked.
    assert any(d.cy == pytest.approx(weak[11][0].cy, abs=1) and d.w == pytest.approx(52) for d in out[11])
    assert any(d.cy == pytest.approx(120, abs=1) for d in out[11])
    plain, tracks = Tracker(conf_weak_long=0.4, **kw).run(frames, weak)
    assert len(tracks) == 1
    assert [round(d.cy) for d in plain[11]] == [120]           # the straight line only


def test_backward_reach_and_entry_tail_need_an_established_track():
    """Two confirmations with weak boxes before them: nothing reaches back.
    Five confirmations: the weak boxes join and the entry tail appears."""
    frames = [[] for _ in range(30)]
    weak = [[] for _ in range(30)]
    for i in range(6, 12):
        weak[i] = [det(300, score=0.5)]
    frames[12] = [det(300)]
    frames[13] = [det(300)]
    kw = dict(min_track=2, max_gap=3, tail=0, tail_before=6, conf_weak=0.4, established_after=5)
    out, tracks = Tracker(**kw).run(frames, weak)
    assert [i for i in range(12) if out[i]] == []
    for i in range(14, 17):
        frames[i] = [det(300)]
    out, tracks = Tracker(**kw).run(frames, weak)
    assert all(out[i] for i in range(0, 12))            # weak boxes 6-11, then the tail to 0
