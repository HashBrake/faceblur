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
from faceblur.redact import redact
from faceblur.verify import (FLAT_DIFF, KINDS_CHECKED, MISSED_APPLIED, applied, box_diff,
                             check, classify, describe, frame_noise, held_for,
                             kinds_checked, not_hands, of_kind, runs_of, screen_runs,
                             summarise)

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


# ------------------------------------------------------- screens in the gate
#
# A screen the run missed is the same question as a face the run missed, asked
# of a different shape, with one rule of its own: how long was it there. The
# screen detector calls a table a laptop for one frame and stops, and no real
# screen behaves that way (docs/report.md section 15.2), so a gate without
# persistence would hold every copy back for a table.

SCREENS = dict(mask=("face", "screen"))


def screen_det(x=80.0, y=60.0, w=200.0, h=120.0, score=0.8, label="tv"):
    return Detection(x, y, w, h, score, None, label, True, score, "screen")


def screen_rows(frames, px=200, x=180, y=120, w=200, h=120, label="tv"):
    """Rows shaped the way `residual_job` writes them, one per frame."""
    return [{"kind": "screen", "frame": i, "px": px, "x": x, "y": y,
             "w": w, "h": h, "label": label, "score": 0.8, "applied": 0.0}
            for i in frames]


def screen_report(frames, settings=None, **box):
    rows = screen_rows(frames, **box)
    return summarise([{"start": 0, "end": 100, "checked": 100, "flat": 0,
                       "flat_by_kind": {"face": 0, "screen": 0}, "found": rows}],
                     settings or Settings(**SCREENS))


def test_a_masked_screen_reads_as_masked_and_an_untouched_one_as_missed():
    """The whole check, asked of a quad instead of an ellipse. `applied` takes
    its region from `redact.region_for`, so the control masks the shape this
    kind actually gets and the comparison happens inside it."""
    rng = np.random.default_rng(7)
    before = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    settings = Settings(**SCREENS)
    det = screen_det()
    after, _ = redact(before, [det], settings)
    changed, would = applied(before, after, det, settings)
    assert classify(changed, would) == "masked"
    assert classify(*applied(before, before, det, settings)) == "missed"


def test_a_screen_seen_in_one_frame_does_not_hold_a_copy_back():
    """14 of the 29 false screen runs on the sample footage last a single
    frame and not one real screen does. A gate without this holds back every
    copy that has a table in it."""
    settings = Settings(**SCREENS)
    out = screen_report([4], settings)
    assert out["residual_screens"] == 1
    assert [run["hits"] for run in out["residual_screen_runs"]] == [1]
    assert held_for(out, settings) == ()


def test_three_frames_of_the_same_screen_do():
    settings = Settings(**SCREENS)
    out = screen_report([4, 5, 6], settings)
    assert out["residual_screens"] == 3
    assert out["residual_screen_frames"] == 3
    assert out["residual_screen_runs"] == [
        {"label": "tv", "first": 4, "last": 6, "hits": 3, "px": 200,
         "x": 180, "y": 120, "w": 200, "h": 120}]
    assert held_for(out, settings) == ("screen",)


def test_two_flickers_on_different_things_are_not_one_screen():
    """Runs are grouped by where the box is, not by the frame number. A table
    that flickers at the left and a mirror that flickers at the right in the
    next frame are two runs of one, not one run of two."""
    settings = Settings(**SCREENS)
    rows = (screen_rows([4], x=100, y=100) + screen_rows([5], x=1200, y=900))
    out = summarise([{"start": 0, "end": 100, "checked": 100, "flat": 0, "found": rows}],
                    settings)
    assert [run["hits"] for run in out["residual_screen_runs"]] == [1, 1]
    assert held_for(out, settings) == ()


def test_a_screen_under_the_size_floor_is_recorded_and_holds_nothing_back():
    """A 24 px screen shows nothing a person could read, so it is reported and
    it decides nothing. It stays in the record, as a hand does."""
    settings = Settings(**SCREENS)
    out = screen_report([4, 5, 6, 7], settings, px=30, w=30, h=20)
    assert out["residual_screens"] == 4
    assert out["residual_screen_max_px"] == 30
    assert out["residual_screen_runs"] == []
    assert held_for(out, settings) == ()
    assert len(out["residual_list"]) == 4


def test_a_run_that_reaches_the_floor_only_at_its_widest_still_counts():
    """The floor is asked of each find, and the run is what is left. A screen
    the camera walks towards crosses the floor part way through."""
    settings = Settings(**SCREENS)
    rows = (screen_rows([4], px=30, w=30, h=20) + screen_rows([5, 6, 7]))
    out = summarise([{"start": 0, "end": 100, "checked": 100, "flat": 0, "found": rows}],
                    settings)
    assert out["residual_screens"] == 4
    assert [run["hits"] for run in out["residual_screen_runs"]] == [3]
    assert held_for(out, settings) == ("screen",)


def test_a_run_of_screens_holds_nothing_back_when_screens_were_not_asked_for():
    """A copy nobody asked to have screens masked in is not missing one."""
    faces_only = Settings()
    out = screen_report([4, 5, 6], faces_only)
    assert out["residual_screens"] == 3
    assert held_for(out, faces_only) == ()


def test_the_check_looks_only_for_what_the_run_was_asked_to_mask():
    assert kinds_checked(Settings()) == ("face",)
    assert kinds_checked(Settings(**SCREENS)) == ("face", "screen")
    assert kinds_checked(Settings(mask=("screen",))) == ("screen",)


def test_a_face_and_a_screen_can_hold_the_same_copy_back():
    settings = Settings(**SCREENS)
    rows = (screen_rows([4, 5, 6])
            + [{"kind": "face", "frame": 9, "px": 60, "x": 5, "y": 5,
                "score": 0.9, "applied": 0.0}])
    out = summarise([{"start": 0, "end": 100, "checked": 100, "flat": 0, "found": rows}],
                    settings)
    assert out["residual_faces"] == 1 and out["residual_screens"] == 3
    assert held_for(out, settings) == ("face", "screen")


def test_the_record_keeps_two_hundred_rows_of_each_kind_not_two_hundred_in_all():
    """On the table tennis file the check finds 67 faces and 129 screens. A
    shared cap would let a file with hundreds of screen finds push the faces
    out of the record an auditor reads, which is the one thing in it that
    nobody can afford to miss."""
    faces = [{"kind": "face", "frame": i, "px": 50, "x": 1, "y": 1, "score": 0.9,
              "applied": 0.0} for i in range(400)]
    out = summarise([{"start": 0, "end": 900, "checked": 900, "flat": 0,
                      "found": faces + screen_rows(range(400, 800))}],
                    Settings(**SCREENS))
    kept = out["residual_list"]
    assert len(of_kind(kept, "face")) == 200
    assert len(of_kind(kept, "screen")) == 200
    assert out["residual_faces"] == 400, "the counts are over every row"
    assert out["residual_screens"] == 400


def test_a_row_written_before_kinds_existed_is_read_as_a_face():
    """An audit record from before 2026-09-10 has no kind on its rows, and
    everything in one is a face, because a face is all this check looked for."""
    old = [{"frame": 2, "px": 50, "x": 1, "y": 1, "score": 0.9, "applied": 0.0}]
    assert of_kind(old, "face") == old
    assert of_kind(old, "screen") == []
    out = summarise([{"start": 0, "end": 10, "checked": 10, "flat": 0, "found": old}])
    assert out["residual_faces"] == 1 and out["residual_screens"] == 0


def test_flat_boxes_are_counted_by_kind_and_as_one_number():
    out = summarise([{"start": 0, "end": 10, "checked": 10, "flat": 5,
                      "flat_by_kind": {"face": 3, "screen": 2}, "found": []},
                     {"start": 10, "end": 20, "checked": 10, "flat": 1,
                      "flat_by_kind": {"face": 1, "screen": 0}, "found": []}])
    assert out["flat_boxes"] == 6
    assert out["flat_by_kind"] == {"face": 4, "screen": 2}


def test_the_grouping_survives_a_row_with_no_box_on_it():
    """A row from an older record carries no width or height. Grouping falls
    back to the long side rather than raising, so an old record still reads."""
    rows = [{"kind": "screen", "frame": i, "px": 90, "x": 100, "y": 100,
             "score": 0.8, "applied": 0.0} for i in (1, 2, 3)]
    runs = screen_runs(rows, Settings(**SCREENS))
    assert [run["hits"] for run in runs] == [3]


def test_the_summary_says_what_it_looked_for():
    settings = Settings(**SCREENS)
    out = screen_report([4, 5, 6], settings)
    text = describe(out, settings)
    assert "3 screen boxes still visible" in text
    assert "no face" in text or "0 faces" in text
    assert "screen" not in describe(out, Settings())


def test_every_ready_kind_is_covered_by_this_check():
    """Condition 2 of "ready" in the build plan: a kind whose masks nothing
    checks is a promise nobody has tested. This is here as well as in
    test_classes so that a reader of either file sees it."""
    from faceblur.classes import available

    for kind in available():
        assert kind.name in KINDS_CHECKED, f"{kind.name} is ready and nothing checks it"


# ------------------------------------------------------ screens on real video

def test_a_run_with_screens_on_writes_the_screen_fields_and_they_round_trip(
        tmp_path, video_with_face):
    """The record has to read back: an auditor gets the JSON, not the run."""
    import json

    out = tmp_path / "screened.mp4"
    record = run_video(video_with_face, out,
                       Settings(check_output=True, mask=("face", "screen"), **FAST),
                       serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.checked_frames > 0
    read_back = json.loads(json.dumps(record.to_dict()))
    for field in ("residual_screens", "residual_screen_frames", "residual_screen_max_px"):
        assert isinstance(read_back[field], int)
    assert isinstance(read_back["residual_screen_runs"], list)
    assert isinstance(read_back["held_back_for"], list)
    assert read_back["flat_by_kind"].get("screen") is not None
    assert all(row.get("kind") in ("face", "screen") for row in read_back["residual_list"])
