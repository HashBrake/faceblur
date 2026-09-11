"""The second chance: what the check found in the copy goes back in as a seed.

The pipeline cannot see its own misses, and package F2 measured what lowering
a threshold to catch them costs: on two files of three it puts masks on the
wearer's own hands, which is the one thing this project never does. So the
faces that get a second chance here are not found by asking the detector to
be less sure. They are found by the check, on the finished copy, on pixels
the first pass never changed, at a threshold the run already trusts.

These tests are about what may seed a pass and what may not, and about the
file that comes out the other end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE
from faceblur.settings import Settings, SettingsError
from faceblur.verify import second_chance_seeds

FAST = dict(device="cpu", copy_clean=False, chunk_seconds=0, encode_seconds=0)


def row(frame=3, px=40, score=0.8, **extra):
    out = {"kind": "face", "frame": frame, "px": px, "x": 100.0, "y": 100.0,
           "w": float(px), "h": float(px), "score": score, "applied": 0.1}
    out.update(extra)
    return out


# ------------------------------------------------------------ what may seed

def test_a_face_the_check_found_becomes_a_seed_on_its_own_frame():
    seeds = second_chance_seeds({"residual_list": [row(frame=7)]}, Settings())
    assert list(seeds) == [7]
    d = seeds[7][0]
    assert d.kind == "face"
    assert d.score == 0.8, "a seed cannot claim more confidence than the check had"
    assert (d.x, d.y) == (80.0, 80.0), "the row holds the centre and a Detection a corner"


def test_a_hand_never_seeds_a_pass():
    """MediaPipe's models called it the wearer's hand. Section 14 measured how
    often that is right, and the rule of 2026-09-11 makes it absolute."""
    seeds = second_chance_seeds({"residual_list": [row(hand=0.9)]}, Settings())
    assert seeds == {}


def test_a_box_the_zone_set_aside_never_seeds_a_pass():
    """The zone left it visible on purpose. Seeding from it would mask the
    thing the wearer is holding, one round later."""
    seeds = second_chance_seeds({"residual_list": [row(zone=0.8)]}, Settings())
    assert seeds == {}


def test_a_screen_does_not_seed_a_face_pass():
    """Screens are held by `screens.hold`, which asks how long it was there.
    A run of one frame is what a table looks like, report 15.2."""
    seeds = second_chance_seeds({"residual_list": [row(kind="screen")]}, Settings())
    assert seeds == {}


def test_a_box_under_the_floor_does_not_seed():
    s = Settings(second_chance_min_px=24)
    assert second_chance_seeds({"residual_list": [row(px=16)]}, s) == {}
    assert second_chance_seeds({"residual_list": [row(px=24)]}, s) != {}


def test_two_faces_on_one_frame_both_seed_it():
    report = {"residual_list": [row(frame=2), row(frame=2, x=300.0), row(frame=5)]}
    seeds = second_chance_seeds(report, Settings())
    assert sorted(seeds) == [2, 5]
    assert len(seeds[2]) == 2


def test_a_check_that_found_nothing_seeds_nothing():
    assert second_chance_seeds({"residual_list": []}, Settings()) == {}
    assert second_chance_seeds({}, Settings()) == {}


# ------------------------------------------------------------ what is refused

def test_a_second_chance_without_the_check_is_refused():
    """The seeds are what the check found. Without it there are none, and a
    setting that quietly does nothing is worse than one that says so."""
    with pytest.raises(SettingsError, match="check_output"):
        Settings(second_chance=1)


def test_a_second_chance_with_a_stride_is_refused():
    """A stride means one frame in N was looked at. A face found on a checked
    frame says nothing about the frames between, and a tracker seeded from it
    would put a mask where nothing was looked at."""
    with pytest.raises(SettingsError, match="check_stride"):
        Settings(second_chance=1, check_output=True, check_stride=2)


def test_it_is_off_by_default():
    assert Settings().second_chance == 0
    assert "second_chance" in Settings().to_dict()


# --------------------------------------------------------- the file it writes

class _MissesOnce:
    """A tracker that keeps nothing the first time and everything after.

    The first pass writes a copy with the face still in it, which is what a
    real miss looks like from the outside. What it fakes is only the miss:
    the check, the seeds, the second write and the second check are all the
    real ones.
    """

    calls = 0

    def run(self, strong, weak, shifts, shape):
        _MissesOnce.calls += 1
        if _MissesOnce.calls == 1:
            return [[] for _ in strong], []
        kept = [list(f) for f in strong]
        return kept, [[d] for f in kept for d in f]


def test_a_face_the_first_pass_missed_is_masked_by_the_second(tmp_path, video_with_face,
                                                              monkeypatch):
    import faceblur.batch as batch

    _MissesOnce.calls = 0
    monkeypatch.setattr(batch, "tracker_for", lambda settings: _MissesOnce())
    out = tmp_path / "again.mp4"
    record = run_video(video_with_face, out,
                       Settings(quarantine=True, second_chance=1, **FAST), serial_submit)

    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces_first_pass > 0, "the first pass was supposed to miss it"
    assert record.second_chance_rounds == 1
    assert record.second_chance_seeds > 0
    assert record.second_chance_seconds > 0.0
    assert record.residual_faces == 0, "the second pass was supposed to catch it"
    assert record.quarantined is False
    assert out.is_file(), "a copy that comes back clean ships"
    assert not (out.parent / "quarantine" / out.name).exists()
    assert not out.with_suffix(".part.mp4").exists()


def test_without_the_second_chance_the_same_run_is_held_back(tmp_path, video_with_face,
                                                             monkeypatch):
    """The control. Same tracker, same footage, one setting different."""
    import faceblur.batch as batch

    _MissesOnce.calls = 0
    monkeypatch.setattr(batch, "tracker_for", lambda settings: _MissesOnce())
    out = tmp_path / "held.mp4"
    record = run_video(video_with_face, out, Settings(quarantine=True, **FAST),
                       serial_submit)

    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces > 0
    assert record.second_chance_rounds == 0
    assert record.residual_faces_first_pass == 0, "no round ran, so there is no first pass"
    assert record.quarantined is True
    assert (out.parent / "quarantine" / out.name).is_file()


def test_a_round_is_not_spent_when_the_check_found_nothing(tmp_path, video_with_face):
    """The pipeline finds the face on its own here, so the check has nothing
    to seed with and no second write happens."""
    out = tmp_path / "clean.mp4"
    record = run_video(video_with_face, out,
                       Settings(check_output=True, second_chance=1, **FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces == 0
    assert record.second_chance_rounds == 0
    assert record.second_chance_seeds == 0


class _KeepsNothing:
    """A tracker that drops every track, the way `min_track` drops a seed that
    stands alone on one frame."""

    def run(self, strong, weak, shifts, shape):
        return [[] for _ in strong], []


def test_a_seed_no_track_picked_up_is_masked_on_its_own_frame(tmp_path, video_with_face,
                                                              monkeypatch):
    """`min_track` is there so one stray detection cannot become a mask. A
    seed is not a stray detection: a second detection pass found it in the
    finished copy on pixels the first pass never changed. Without this the
    check's evidence is paid for and thrown away."""
    import faceblur.batch as batch

    monkeypatch.setattr(batch, "tracker_for", lambda settings: _KeepsNothing())
    out = tmp_path / "kept.mp4"
    record = run_video(video_with_face, out,
                       Settings(quarantine=True, second_chance=1, **FAST), serial_submit)

    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_seeds > 0
    assert record.second_chance_tracks_added == 0, "the tracker kept nothing, by design"
    # Fewer put back than seeds, because each carries the run's own tail and
    # covers the seeds on the frames either side of it. This clip is a still
    # face on every frame, so one put back covers several seeds.
    assert 0 < record.second_chance_seeds_kept <= record.second_chance_seeds
    assert record.residual_faces < record.residual_faces_first_pass
