"""T26: no tracker-drawn face box on the wearer's arm, and a minimum time shown
for screens, text and plates.

The field test's AM window put solid face discs on the wearer's raised forearm:
a background face behind the arm keeps its track, and the boxes the tracker
draws between and around its sightings land on the arm. `zone.off_arm` drops
those boxes in the strip from a hand through its wrist to the frame's border.
Text, screens and plates blinked as short runs came and went; `*_hold` keeps
each run up for a minimum number of frames, as `face_hold` does for faces.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceblur import heads, plates, text
from faceblur.detect import Detection
from faceblur.screens import hold as hold_screens
from faceblur.settings import Settings, SettingsError
from faceblur.zone import ZoneFrame, arm_corridor, off_arm

SHAPE = (1000, 1000)          # height, width


def hand_quad(cx, cy, side=100.0, wrist="down"):
    """A hand quad in `Hand.quad` order: fingers along the first edge, wrist
    along the third. `wrist` is the side the forearm leaves by."""
    h = side / 2
    up = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    if wrist == "down":
        return tuple(up)
    if wrist == "right":     # fingers point left
        return ((cx - h, cy + h), (cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h))
    raise ValueError(wrist)


def face(x, y, source="track", kind="face", w=30.0):
    return Detection(float(x) - w / 2, float(y) - w / 2, w, w, 0.9, None, source, kind=kind)


def on(**kw):
    return Settings(arm_corridor_scale=1.0, **kw)


def test_the_corridor_runs_from_the_hand_through_the_wrist_to_the_border():
    strip = arm_corridor(hand_quad(500, 300), SHAPE, 1.0)
    xs, ys = strip[:, 0], strip[:, 1]
    assert xs.min() == pytest.approx(450) and xs.max() == pytest.approx(550)
    assert ys.min() == pytest.approx(300) and ys.max() == pytest.approx(1000)
    side = arm_corridor(hand_quad(500, 300, wrist="right"), SHAPE, 2.0)
    assert side[:, 0].max() == pytest.approx(1000) and side[:, 0].min() == pytest.approx(500)
    assert side[:, 1].min() == pytest.approx(200) and side[:, 1].max() == pytest.approx(400)


def test_the_corridor_follows_the_wrist_not_the_nearest_edge():
    # A hand raised at the right of the picture with its arm coming up from
    # below: the right edge is nearest, the arm is below.
    zone = ZoneFrame(hands=(hand_quad(900, 300),))
    out = off_arm([[face(900, 700), face(990, 300)]], [zone], SHAPE, on())
    assert [d.cy for d in out[0]] == [300.0]


def test_a_tracker_box_on_the_arm_is_dropped_and_a_detection_is_kept():
    zone = ZoneFrame(hands=(hand_quad(500, 300),))
    dets = [face(500, 700), face(500, 650, source="yunet"), face(500, 600, source="continued"),
            face(200, 700)]
    out = off_arm([dets], [zone], SHAPE, on())
    assert [(d.source, d.cx) for d in out[0]] == [("yunet", 500.0), ("continued", 500.0),
                                                  ("track", 200.0)]


def test_other_kinds_in_the_corridor_are_untouched():
    zone = ZoneFrame(hands=(hand_quad(500, 300),))
    dets = [face(500, 700, source="track", kind="text"), face(500, 700, source="track", kind="screen")]
    assert off_arm([dets], [zone], SHAPE, on()) == [dets]


def test_off_by_default_and_with_the_zone_off():
    zone = ZoneFrame(hands=(hand_quad(500, 300),))
    dets = [face(500, 700)]
    assert Settings().arm_corridor_scale == 0.0
    assert off_arm([dets], [zone], SHAPE, Settings()) == [dets]
    assert off_arm([dets], [zone], SHAPE, on(zone=False)) == [dets]


def test_no_hand_no_corridor():
    dets = [face(500, 700)]
    assert off_arm([dets], [ZoneFrame()], SHAPE, on(arm_corridor_carry=0)) == [dets]


def test_the_last_hands_are_carried_with_the_camera_then_dropped():
    zones = [ZoneFrame(hands=(hand_quad(500, 300),)), ZoneFrame(), ZoneFrame(), ZoneFrame()]
    shifts = [(0.0, 0.0), (200.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    frames = [[face(700, 700), face(500, 700)] for _ in zones]
    out = off_arm(frames, zones, SHAPE, on(arm_corridor_carry=2), shifts)
    # Frame 0 is cut at x=500; frames 1 and 2 at x=700, where the camera took
    # the hand; frame 3 is past the carry.
    assert [d.cx for d in out[0]] == [700.0]
    assert [d.cx for d in out[1]] == [500.0]
    assert [d.cx for d in out[2]] == [500.0]
    assert [d.cx for d in out[3]] == [700.0, 500.0]


def test_new_arm_settings_are_validated():
    Settings(arm_corridor_scale=1.5, arm_corridor_carry=0)
    for bad in ({"arm_corridor_scale": -1.0}, {"arm_corridor_carry": -1}):
        with pytest.raises(SettingsError):
            Settings(**bad)
    d = Settings().to_dict()
    assert d["arm_corridor_scale"] == 0.0 and d["arm_corridor_carry"] == 10


# ---------------------------------------------------------------- the holds

def box(x=100.0, source="dbnet", kind="text"):
    return Detection(x, 100.0, 80.0, 20.0, 0.9, None, source, kind=kind)


def test_a_short_screen_run_is_held_for_the_minimum():
    found = {10: [box(kind="screen", source="yolox")], 11: [box(kind="screen", source="yolox")]}
    s = Settings(screen_tail=2, screen_hold=0, screen_min_run=2)
    assert sorted(hold_screens(found, s, 100)) == [10, 11, 12, 13]
    held = hold_screens(found, s.with_changes(screen_hold=8), 100)
    assert sorted(held) == list(range(10, 18))


def test_the_hold_leaves_a_long_run_and_unconfirmed_runs_alone():
    found = {i: [box(kind="screen", source="yolox")] for i in range(10, 30)}
    s = Settings(screen_tail=2, screen_hold=8, screen_min_run=2)
    assert sorted(hold_screens(found, s, 100)) == list(range(10, 32))
    once = {10: [box(kind="screen", source="yolox")]}
    assert hold_screens(once, s, 100) == {}


def test_the_hold_stops_at_the_end_of_the_video():
    found = {95: [box(kind="screen", source="yolox")], 96: [box(kind="screen", source="yolox")]}
    held = hold_screens(found, Settings(screen_tail=0, screen_hold=20, screen_min_run=2), 100)
    assert sorted(held) == [95, 96, 97, 98, 99]


def test_text_and_plates_use_their_own_hold_and_heads_none():
    found = {10: [box()], 11: [box()]}
    s = Settings(text_tail=0, text_hold=6, screen_hold=50, plate_hold=50, text_min_run=2)
    assert sorted(text.hold(found, s, 100)) == list(range(10, 16))
    pl = {10: [box(kind="plate", source="plate")], 11: [box(kind="plate", source="plate")]}
    s = Settings(plate_tail=0, plate_hold=5, screen_hold=50, text_hold=50, plate_min_run=2)
    assert sorted(plates.hold(pl, s, 100)) == list(range(10, 15))
    hd = {10: [box(kind="face", source="person_head")], 11: [box(kind="face", source="person_head")]}
    s = Settings(head_tail=0, head_min_run=2, screen_hold=50)
    assert sorted(heads.hold(hd, s, 100)) == [10, 11]


def test_hold_settings_are_validated():
    for name in ("screen_hold", "text_hold", "plate_hold"):
        with pytest.raises(SettingsError):
            Settings(**{name: -1})
    d = Settings().to_dict()
    for name in ("screen_hold", "text_hold", "plate_hold"):
        assert name in d


# ------------------------------------- small tracks with borrowed confirmations

def small_track(confs, w=14.0, tid=0):
    from faceblur.track import Track
    dets = {i * 3: Detection(100.0, 100.0, w, w, 0.9, None, "yunet", True, c)
            for i, c in enumerate(confs)}
    return Track(tid, dets, confirmed=len(confs), last_confirmed=max(dets),
                 best_conf=max(confs))


def render(tracks, n=40, **kw):
    from faceblur.track import Tracker
    args = dict(min_track=2, tail=0, tail_before=0, small_px=20.0, small_conf=0.6)
    args.update(kw)
    return Tracker(**args).render(tracks, n)


def widths(out):
    return {round(d.w) for f in out for d in f}


def test_a_small_track_whose_strong_box_is_an_outlier_is_not_drawn():
    texture = small_track([0.45, 0.9, 0.44, 0.5, 0.42], w=13.0)    # took one from a face
    real = small_track([0.88, 0.9, 0.87, 0.89], w=14.0, tid=1)
    assert widths(render([texture, real])) == {13, 14}
    out = render([texture, real], small_spread=0.3)
    # Only the texture track's strong box (frame 3) is left of it, for one
    # frame with no hold; the real face is drawn whole.
    assert [k for k, f in enumerate(out) if any(round(d.w) == 13 for d in f)] == [3]
    assert sum(1 for f in out for d in f if round(d.w) == 14) == 10


def test_the_strong_box_of_a_dropped_track_is_held():
    texture = small_track([0.45, 0.9, 0.44, 0.5, 0.42], w=13.0)
    out = render([texture], small_spread=0.3, hold=5)
    assert [k for k, f in enumerate(out) if f] == [3, 4, 5, 6, 7]


def test_a_small_face_scoring_alike_throughout_is_kept():
    # 004100's small labelled faces: best 0.64-0.69, median 0.49-0.56.
    face = small_track([0.49, 0.65, 0.5, 0.55, 0.52], w=16.0)
    assert len([f for f in render([face], small_spread=0.3) if f]) == 13


def test_the_spread_rule_leaves_large_tracks_alone_and_skips_continuations():
    big = small_track([0.45, 0.9, 0.44], w=30.0)
    assert len([f for f in render([big], small_spread=0.3) if f]) == 7
    t = small_track([0.9, 0.88, 0.9], w=14.0)
    for k in (1, 2, 4, 5):
        t.dets[k] = Detection(100.0, 100.0, 14.0, 14.0, 0.3, None, "continued")
    assert len([f for f in render([t], small_spread=0.3) if f]) == 7


def test_face_small_spread_setting():
    from faceblur.pipeline import tracker_for
    assert Settings().face_small_spread == 0.3
    assert tracker_for(Settings(face_small_spread=0.2)).small_spread == 0.2
    assert "face_small_spread" in Settings().to_dict()
    for bad in (-0.1, 1.5):
        with pytest.raises(SettingsError):
            Settings(face_small_spread=bad)
