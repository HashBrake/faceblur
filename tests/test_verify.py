"""The check that reads the output: a face left in the copy is found and held.

The pipeline cannot see its own misses; every other measurement in the project
reads the source and inherits that blindness. These tests set up both answers
on purpose: a copy where the face was masked, and a copy where it was not.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import run_ffmpeg
from faceblur.batch import run_video, serial_submit
from faceblur.detect import Detection
from faceblur.pipeline import STATUS_DONE, sidecar_path
from faceblur.settings import Settings
from faceblur.verify import (FLAT_DIFF, MISSED_APPLIED, box_diff, check, classify,
                             describe, frame_noise, not_hands, runs_of, summarise)

LM = ((10.0, 10.0), (30.0, 10.0), (20.0, 20.0), (12.0, 30.0), (28.0, 30.0))
FAST = dict(device="cpu", copy_clean=False, chunk_seconds=0, encode_seconds=0)


def det(x=100.0, y=100.0, size=40.0):
    return Detection(x, y, size, size, 0.9, LM)


# ------------------------------------------------------------------ the rule

def test_a_box_the_copy_never_changed_is_a_miss():
    assert classify(changed=0.5, would=20.0) == "missed"


def test_a_box_the_copy_changed_as_much_as_a_mask_would_is_the_blur_itself():
    """A detector fires on a blurred face too; that is not a miss."""
    assert classify(changed=19.0, would=20.0) == "masked"
    assert classify(changed=MISSED_APPLIED * 20.0 + 0.1, would=20.0) == "masked"


def test_a_box_a_mask_would_barely_change_decides_nothing():
    """A flat wall or a dark corner: the ratio is noise, so it is not read."""
    assert classify(changed=0.0, would=FLAT_DIFF - 0.1) == "flat"


def test_the_encoder_s_own_noise_is_taken_off_both_sides():
    """The shape of the measured case: a dark 40 px face the run missed.

    A mask would move that box by about 15, and re-encoding the file moves it
    by about 3.5 all by itself. Left in, the encoder's own noise is most of
    what the ratio reads on a dark face, and a plain miss creeps up towards
    the line; taken off both sides, it reads as the miss it is.
    """
    assert classify(changed=5.0, would=15.0) == "masked"
    assert classify(changed=5.0, would=15.0, noise=3.5) == "missed"


def test_noise_is_the_median_change_over_the_frame_not_the_masked_part():
    before = np.zeros((80, 80, 3), np.uint8)
    after = before + 2                      # the whole frame moved a little
    after[0:20, 0:20] = 200                 # one masked face moved a lot
    assert frame_noise(before, after, step=1) == pytest.approx(2.0)


def test_the_difference_is_measured_inside_the_box_only():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[0:20, 0:20] = 255                      # far from the box
    assert box_diff(a, b, det(40, 40, 20)) == 0.0
    b[40:60, 40:60] = 255
    assert box_diff(a, b, det(40, 40, 20)) == pytest.approx(255.0)


def test_a_box_off_the_edge_of_the_frame_does_not_raise():
    a = np.zeros((50, 50, 3), np.uint8)
    assert box_diff(a, a.copy(), det(-80, -80, 20)) == 0.0


def test_the_gate_reads_every_row_not_the_two_hundred_the_record_carries():
    """`residual_list` is truncated for the record. A file with more finds
    than that would otherwise be judged on the first 200 of them, and the one
    that mattered could be number 300."""
    found = [{"frame": i, "px": 10, "x": 1, "y": 1, "score": 0.9, "applied": 0.0}
             for i in range(400)]
    found.append({"frame": 500, "px": 120, "x": 1, "y": 1, "score": 0.9, "applied": 0.0})
    out = summarise([{"start": 0, "end": 600, "checked": 600, "flat": 0, "found": found}])
    assert len(out["residual_list"]) == 200
    assert out["residual_max_px"] == 120


def test_frames_become_stretches():
    assert runs_of([3, 4, 5, 40, 41]) == [[3, 5], [40, 41]]
    assert runs_of([3, 6, 20], gap=5) == [[3, 6], [20, 20]]
    assert runs_of([]) == []


def test_the_summary_counts_by_size():
    found = [{"frame": 2, "px": 50, "x": 1, "y": 1, "score": 0.9, "applied": 0.0},
             {"frame": 2, "px": 30, "x": 9, "y": 9, "score": 0.9, "applied": 0.1},
             {"frame": 9, "px": 10, "x": 9, "y": 9, "score": 0.9, "applied": 0.0}]
    out = summarise([{"start": 0, "end": 10, "checked": 10, "flat": 2, "found": found}])
    assert out["residual_faces"] == 3 and out["residual_frames"] == 2
    assert out["residual_by_size"] == {"40+ px": 1, "24-40 px": 1, "under 24 px": 1}
    assert out["residual_runs"] == [[2, 2], [9, 9]]
    assert out["flat_boxes"] == 2
    assert out["residual_hands"] == 0
    assert out["residual_max_px"] == 50
    assert "3 faces still in the copy" in describe(out)


def test_a_hand_stays_in_the_record_and_out_of_every_count():
    """A rule that quietly dropped what it disagreed with would be worth
    nothing to an auditor, so the row stays, marked with the coverage that
    decided it. Nothing a decision hangs on counts it."""
    found = [{"frame": 2, "px": 50, "x": 1, "y": 1, "score": 0.9, "applied": 0.0},
             {"frame": 4, "px": 90, "x": 9, "y": 9, "score": 0.9, "applied": 0.1,
              "hand": 0.94}]
    out = summarise([{"start": 0, "end": 10, "checked": 10, "flat": 0,
                      "hands": 1, "found": found}])
    assert out["residual_faces"] == 1
    assert out["residual_hands"] == 1
    assert out["residual_frames"] == 1
    assert out["residual_by_size"] == {"40+ px": 1, "24-40 px": 0, "under 24 px": 0}
    assert out["residual_runs"] == [[2, 2]]
    assert out["residual_max_px"] == 50                 # not the 90 px hand
    assert len(out["residual_list"]) == 2               # the hand is still there
    assert "1 more set aside as the wearer's hands" in describe(out)
    assert [row["frame"] for row in not_hands(found)] == [2]


# ------------------------------------------------------------ on real video

@pytest.fixture(scope="module")
def blurred(tmp_path_factory, video_with_face):
    """The face clip, run through the pipeline the ordinary way."""
    out = tmp_path_factory.mktemp("verified") / "face_blurred.mp4"
    record = run_video(video_with_face, out, Settings(**FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    return out


def test_a_copy_with_the_face_masked_holds_nothing_back(video_with_face, blurred):
    report = check(video_with_face, blurred, Settings(**FAST))
    assert report["frames_checked"] > 0
    assert report["residual_faces"] == 0, report["residual_list"]


def test_a_copy_that_masked_nothing_is_caught(tmp_path, video_with_face):
    """The same file re-encoded and not masked: every frame still shows a face."""
    untouched = tmp_path / "untouched.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-i", str(video_with_face),
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                         "-pix_fmt", "yuv420p", str(untouched)])
    assert result.returncode == 0, result.stderr[-400:]
    report = check(video_with_face, untouched, Settings(**FAST))
    assert report["residual_faces"] > 0
    assert report["residual_by_size"]["40+ px"] > 0
    assert report["residual_runs"][0][0] == 0
    assert all(row["applied"] < MISSED_APPLIED for row in report["residual_list"])


def test_a_run_that_checks_itself_says_so_in_the_audit_record(tmp_path, video_with_face):
    out = tmp_path / "checked.mp4"
    record = run_video(video_with_face, out, Settings(check_output=True, **FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.checked_frames > 0
    assert record.residual_faces == 0
    assert record.check_seconds >= 0.0


def test_a_copy_that_only_shows_hands_is_not_held_back(tmp_path, video_with_face,
                                                      monkeypatch):
    """The gate reads the faces, not the hands. With every flagged box called
    a hand, the copy ships and the record still names them."""
    import faceblur.batch as batch

    monkeypatch.setattr(batch, "tracker_for", lambda settings: _NothingFound())

    class _AllHands:
        model_hashes = {"palm": "stub"}
        compute = {"palm": "stub"}

        def hand_cover(self, frame, det):
            return 1.0

    monkeypatch.setattr(batch, "_hand_rule", lambda settings: _AllHands())
    out = tmp_path / "shipped.mp4"
    record = run_video(video_with_face, out, Settings(quarantine=True, **FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces == 0
    assert record.residual_hands > 0
    assert record.quarantined is False
    assert out.is_file()
    assert any("hand" in row for row in record.residual_list)


def test_the_hand_rule_can_be_turned_off(tmp_path, video_with_face, monkeypatch):
    """Without it the check reports the same boxes as missed faces, which is
    what it did before it could tell them apart."""
    import faceblur.batch as batch

    monkeypatch.setattr(batch, "tracker_for", lambda settings: _NothingFound())
    called = []
    monkeypatch.setattr(batch, "_hand_rule", lambda settings: called.append(1))
    out = tmp_path / "no_rule.mp4"
    record = run_video(video_with_face, out, Settings(quarantine=True, hand_rule=False, **FAST),
                       serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert called == []
    assert record.residual_faces > 0
    assert record.residual_hands == 0
    assert record.quarantined is True


def test_a_clean_copy_ships_even_with_the_gate_wide_open(tmp_path, video_with_face):
    """`--quarantine-px 0` means every face holds a copy back, not that a copy
    with no face in it does."""
    out = tmp_path / "clean.mp4"
    record = run_video(video_with_face, out,
                       Settings(quarantine=True, quarantine_min_px=0, **FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces == 0
    assert record.quarantined is False
    assert out.is_file()


def test_a_copy_that_still_shows_a_face_is_moved_to_quarantine(tmp_path, video_with_face,
                                                               monkeypatch):
    """With no mask applied, the whole file is held back instead of shipped."""
    import faceblur.batch as batch

    # A pipeline that finds nothing: the copy is a re-encode with the face intact.
    monkeypatch.setattr(batch, "tracker_for",
                        lambda settings: _NothingFound())
    out = tmp_path / "held.mp4"
    record = run_video(video_with_face, out, Settings(quarantine=True, **FAST), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.residual_faces > 0
    assert record.quarantined is True
    assert not out.exists()
    held = out.parent / "quarantine" / out.name
    assert held.is_file()
    assert sidecar_path(held).is_file()
    assert not sidecar_path(out).exists()


class _NothingFound:
    """A tracker that keeps no face, so nothing is masked."""

    def run(self, strong, weak, shifts, shape):
        return [[] for _ in strong], []
