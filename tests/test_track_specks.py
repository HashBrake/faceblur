"""T25: the tracker rules against tiny short-lived face specks and flicker.

The field test of 2026-11 showed thousands of one-to-five-frame face boxes a
minute: the gap walk masked every box near a track's path, stitched gaps were
drawn as straight lines on the screen while the camera moved, the stitch slack
let a 15 px face join a box hundreds of pixels away, and a short track blinked.
"""
from __future__ import annotations

import pytest

from faceblur.detect import Detection
from faceblur.pipeline import tracker_for
from faceblur.settings import Settings, SettingsError
from faceblur.track import Track, Tracker, _merge

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))


def det(x, y=100.0, w=40.0, h=None, score=0.9):
    return Detection(float(x), float(y), float(w), float(w if h is None else h), score, LM)


def tracker(**kw):
    args = dict(min_track=2, established_after=3, max_gap=2, tail=0, tail_before=0,
                conf_weak=0.4, conf_weak_long=0.3, weak_run=30, sure_conf=0.0)
    args.update(kw)
    return Tracker(**args)


def gap_track():
    """A sure track seen at frames 0 and 10 at x=100, nothing between."""
    return Track(0, {0: det(100), 10: det(100)}, confirmed=5, last_confirmed=10, best_conf=0.9)


def walk_and_render(tr, weak, n=11, shifts=None):
    frames = [[] for _ in range(n)]
    t = gap_track()
    tr._set_shifts(shifts, n)
    tr._fill_gaps([t], frames, weak)
    return t, tr.render([t], n)


# ---------------------------------------------------------------- gap walk

def speck_weak(n=11):
    weak = [[] for _ in range(n)]
    weak[3] = [det(104, score=0.35),            # on the path
               det(200, score=0.35)]            # a speck two face widths away
    return weak


def test_the_old_walk_masks_every_box_near_the_path():
    _, out = walk_and_render(tracker(), speck_weak())
    assert len(out[3]) == 3                     # the gap box and both boxes


def test_the_walk_on_path_masks_one_box_a_frame_and_no_speck():
    t, out = walk_and_render(tracker(walk_on_path=True), speck_weak())
    assert [d.x for d in t.extras[3]] == [104.0]
    assert len(out[3]) == 1
    merged = out[3][0]
    assert (merged.x, merged.x + merged.w) == (100.0, 144.0)
    assert all(len(f) == 1 for f in out)        # one mask a frame, gap or not


def test_the_walk_on_path_takes_no_box_off_its_prediction():
    weak = [[] for _ in range(11)]
    weak[3] = [det(150, score=0.35)]            # 10 px clear of the 40 px face
    t, out = walk_and_render(tracker(walk_on_path=True), weak)
    assert t.extras == {}
    assert [len(f) for f in out] == [1] * 11


def test_the_walk_on_path_follows_the_camera():
    """The camera pans 60 px a frame; the face's box moves with it, so only a
    walk that moves its prediction with the camera can step onto it."""
    weak = [[] for _ in range(11)]
    weak[1] = [det(160, score=0.35)]
    t, _ = walk_and_render(tracker(walk_on_path=True), weak, shifts=[(60.0, 0.0)] * 11)
    assert 1 in t.extras
    t, _ = walk_and_render(tracker(walk_on_path=True), weak, shifts=None)
    assert 1 not in t.extras


def test_merge_grows_to_every_touching_box_and_drops_the_rest():
    box = det(100)
    merged = _merge(box, [det(130), det(300)])
    assert (merged.x, merged.w) == (100.0, 70.0)
    assert merged.landmarks is None             # padded ellipse of the whole box
    inside = _merge(box, [det(110, y=110, w=10)])
    assert inside is box                        # nothing grew, landmarks kept
    assert _merge(box, [det(300)]) is box


# ---------------------------------------------------------------- gap and camera

def test_a_gap_box_follows_the_camera_path():
    """The camera pans right five frames and back; the face, fixed in the
    scene, is at x=100 at both ends but 150 in the middle of the gap."""
    shifts = [(0.0, 0.0)] + [(10.0, 0.0)] * 5 + [(-10.0, 0.0)] * 5
    t = gap_track()
    plain = tracker()
    plain._set_shifts(shifts, 11)
    assert plain.render([t], 11)[5][0].x == pytest.approx(100.0)
    follow = tracker(gap_camera=True)
    follow._set_shifts(shifts, 11)
    out = follow.render([t], 11)
    assert out[5][0].x == pytest.approx(150.0)
    assert out[2][0].x == pytest.approx(120.0)
    assert out[8][0].x == pytest.approx(120.0)


# ---------------------------------------------------------------- stitch slack

def two_pieces(w, apart=150.0, n=40):
    frames = [[] for _ in range(n)]
    for i in range(0, 5):
        frames[i] = [det(100, w=w)]
    for i in range(34, 39):
        frames[i] = [det(100 + apart, w=w)]
    return frames


def test_the_stitch_slack_cap_keeps_a_small_face_from_a_far_box():
    frames = two_pieces(15)
    _, joined = tracker(stitch_gap=45, stitch_slack=8.0).run(frames)
    assert len(joined) == 1
    _, apart = tracker(stitch_gap=45, stitch_slack=8.0, stitch_slack_frac=0.25).run(frames)
    assert len(apart) == 2


def test_the_stitch_slack_cap_leaves_a_large_face_alone():
    _, joined = tracker(stitch_gap=45, stitch_slack=8.0, stitch_slack_frac=0.25).run(
        two_pieces(100))
    assert len(joined) == 1


# ---------------------------------------------------------------- minimum time shown

def short_track(n=20):
    frames = [[] for _ in range(n)]
    frames[5] = [det(100)]
    frames[6] = [det(100)]
    return frames


def covered(out):
    return [i for i, f in enumerate(out) if f]


def test_a_short_track_is_shown_for_the_hold():
    out, _ = tracker(hold=10).run(short_track())
    assert covered(out) == list(range(5, 15))
    out, _ = tracker().run(short_track())
    assert covered(out) == [5, 6]


def test_the_held_box_moves_with_the_camera():
    tr = tracker(hold=10)
    out, _ = tr.run(short_track(), shifts=[(5.0, 0.0)] * 20)
    assert out[10][0].x == pytest.approx(100.0 + 5 * 4)


def test_the_hold_never_shows_an_unconfirmed_box():
    weak = [[] for _ in range(20)]
    weak[5] = [det(100, score=0.45)]
    weak[6] = [det(100, score=0.45)]
    out, tracks = tracker(hold=10).run([[] for _ in range(20)], weak)
    assert tracks == [] and covered(out) == []
    frames = [[] for _ in range(20)]
    frames[5] = [det(100)]                       # one confirmation: no track
    out, _ = tracker(hold=10).run(frames)
    assert covered(out) == []


def test_the_hold_does_not_change_a_long_track():
    frames = [[det(100)] if 2 <= i < 16 else [] for i in range(20)]
    assert covered(tracker(hold=10).run(frames)[0]) == list(range(2, 16))


# ---------------------------------------------------------------- paint floor

def test_a_track_narrower_than_the_floor_is_not_painted():
    frames = [[det(100, w=12), det(300, w=20)] if i < 6 else [] for i in range(8)]
    out, tracks = tracker(paint_min_px=14).run(frames)
    assert len(tracks) == 2                      # still tracked, only not drawn
    assert all([d.w for d in f] == [20.0] for f in out[:6])
    out, _ = tracker().run(frames)
    assert all(len(f) == 2 for f in out[:6])


def test_the_floor_is_per_track_so_a_jittering_face_does_not_blink():
    widths = [13, 15, 13, 15, 15, 13, 15]        # median 15, some boxes under 14
    frames = [[det(100, w=w)] for w in widths]
    out, _ = tracker(paint_min_px=14).run(frames)
    assert all(len(f) == 1 for f in out)


# ---------------------------------------------------------------- settings

def test_the_new_settings_and_their_defaults():
    s = Settings()
    assert s.stitch_slack_frac == 0.25
    assert s.gap_walk_on_path is True
    assert s.gap_follow_camera is True
    assert s.face_hold == 10
    assert s.face_paint_min_px == 14.0
    d = s.to_dict()
    for key in ("stitch_slack_frac", "gap_walk_on_path", "gap_follow_camera", "face_hold",
                "face_paint_min_px"):
        assert key in d


def test_tracker_for_passes_the_new_settings():
    tr = tracker_for(Settings(stitch_slack_frac=0.5, gap_walk_on_path=False,
                              gap_follow_camera=False, face_hold=3))
    assert (tr.stitch_slack_frac, tr.walk_on_path, tr.gap_camera, tr.hold) == (0.5, False, False, 3)
    tr = tracker_for(Settings())
    assert (tr.stitch_slack_frac, tr.walk_on_path, tr.gap_camera, tr.hold) == (0.25, True, True, 10)
    assert tr.paint_min_px == 14.0
    assert tracker_for(Settings(face_paint_min_px=0)).paint_min_px == 0.0


@pytest.mark.parametrize("changes", [
    {"stitch_slack_frac": -0.1},
    {"face_hold": -1},
    {"face_paint_min_px": -1},
    {"gap_walk_on_path": "yes"},
    {"gap_follow_camera": 1},
])
def test_bad_new_settings_are_refused(changes):
    with pytest.raises(SettingsError):
        Settings(**changes)


def test_the_face_consensus_thresholds_are_unchanged():
    s = Settings()
    assert (s.conf, s.verify_conf, s.conf_weak, s.conf_weak_long, s.conf_agree,
            s.verify_conf_agree, s.min_track, s.track_sure_conf) == (
        0.5, 0.3, 0.4, 0.3, 0.35, 0.35, 2, 0.5)
