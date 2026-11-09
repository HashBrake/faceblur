"""T29: a close face keeps its mask, to the last frame and from the first.

PV_BJ_072232 (second field run) lost a close face for the last 19 frames of
the stream. Every frame was detected; the face was confirmed by both
detectors on the final frames, but at 260-350 px of 1600 it was over
max_face_frac, so it could only continue a track, never start or keep one.
"""
from __future__ import annotations

from faceblur.detect import Detection, filter_candidates, weak_candidates
from faceblur.pipeline import tracker_for
from faceblur.settings import Settings
from faceblur.track import Tracker

SHAPE = (1200, 1600)
N = 60


def box(x, y, w, h, score, source="yunet"):
    return Detection(float(x), float(y), float(w), float(h), score, None, source)


def raw_face(x, size, cf=0.8):
    """One frame's raw candidates: a YuNet face both detectors see."""
    return {"yunet": [box(x, 400, size, size, 0.8)],
            "centerface": [box(x, 400, size, size, cf, "centerface")],
            "centerface_flip": [], "centerface_tight": [], "ultraface": []}


def empty():
    return {"yunet": [], "centerface": [], "centerface_flip": [], "centerface_tight": [],
            "ultraface": []}


def painted(raws, settings=None):
    """The tracker's face boxes per frame for a sequence of raw candidates."""
    s = settings or Settings()
    strong = [filter_candidates(r, s, SHAPE) for r in raws]
    weak = [weak_candidates(r, s, SHAPE) for r in raws]
    out, _ = tracker_for(s).run(strong, weak, None, SHAPE)
    return out


def test_a_close_face_that_appears_near_the_end_is_masked_to_the_last_frame():
    raws = [empty() for _ in range(N)]
    for i in range(N - 7, N):
        raws[i] = raw_face(800 - 5 * i, 300)
    out = painted(raws)
    assert all(out[i] for i in range(N - 7, N))


def test_a_close_face_on_the_first_frames_is_masked_from_frame_zero():
    raws = [empty() for _ in range(N)]
    for i in range(7):
        raws[i] = raw_face(800 + 5 * i, 300)
    out = painted(raws)
    assert all(out[i] for i in range(7))


def test_a_face_that_comes_close_stays_masked_past_weak_run():
    """200 px, then 300 px for longer than weak_run: the whole time masked,
    not just weak_run frames past the last box under max_face_frac."""
    s = Settings()
    raws = [raw_face(800, 200) for _ in range(5)]
    raws += [raw_face(800, 300) for _ in range(s.weak_run + 25)]
    out = painted(raws, s)
    assert all(out)


def test_the_hdf5_settings_mask_a_close_face_to_the_last_frame():
    from faceblur import h5

    s = Settings(mask=("face", "text", "plate", "screen"), zone_stride=h5.H5_ZONE_STRIDE,
                 head_backstop=True)
    raws = [empty() for _ in range(N)]
    for i in range(N - 7, N):
        raws[i] = raw_face(800, 300)
    out = painted(raws, s)
    assert all(out[i] for i in range(N - 7, N))


def test_a_large_box_without_confirmation_is_still_nothing():
    raws = [raw_face(800, 300, cf=0.0) for _ in range(N)]
    assert not any(painted(raws))


def det(x, score=0.9):
    return Detection(float(x), 100.0, 60.0, 60.0, score, None, "yunet", True, 0.8)


def test_a_face_confirmed_on_the_last_two_frames_is_masked_on_both():
    frames = [[] for _ in range(20)]
    frames[18] = [det(100)]
    frames[19] = [det(101)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=2).run(frames)
    assert len(tracks) == 1 and out[18] and out[19]


def test_a_face_confirmed_on_the_first_two_frames_is_masked_on_both():
    frames = [[] for _ in range(20)]
    frames[0] = [det(100)]
    frames[1] = [det(101)]
    out, tracks = Tracker(min_track=2, max_gap=3, tail=2).run(frames)
    assert len(tracks) == 1 and out[0] and out[1]
