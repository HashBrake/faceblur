"""The second chance, scoped to the stretches whose masks changed.

A round used to write and check the whole file again, and it was a third of
the wall on every sample file (docs/bench/profile_2026-10-16.md). With the
scope "changed" it encodes again, from the source, only the keyframe
intervals of the first copy that hold a changed mask, copies the rest from
the first copy, and checks again only what it encoded. These tests hold it to
the same answer as the scope "all".
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from faceblur import batch
from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE
from faceblur.segments import decode_range, frame_times, verify
from faceblur.settings import Settings, SettingsError
from faceblur.verify import merge_checks, summarise
from faceblur.video import probe

from conftest import run_ffmpeg

# Copies on, so the first copy has keyframes to cut at; one second pieces, so
# the check and the encode run in several jobs and the merge has work to do.
# The real one, before any test wraps it: a second wrap would wrap the first.
_WRITE_AND_JOIN = batch._write_and_join

SCOPED = dict(device="cpu", quarantine=True, second_chance=1, copy_clean=True,
              chunk_seconds=1, encode_seconds=1)


@pytest.fixture(scope="module")
def face_then_wall(tmp_path_factory, lena) -> Path:
    """One second of a still face, then five of a grey wall, 10 fps, a
    keyframe every second. Only the first second has anything to mask, so a
    round changes the masks of a sixth of the file."""
    out = tmp_path_factory.mktemp("clips") / "face_then_wall.mp4"
    result = run_ffmpeg([
        "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", "10", "-t", "1", "-i", str(lena),
        "-f", "lavfi", "-i", "color=c=gray:s=640x640:r=10:d=5",
        "-filter_complex",
        "[0:v]scale=160:-2,pad=640:640:240:240:gray,setsar=1,format=yuv420p[a];"
        "[1:v]setsar=1,format=yuv420p[b];[a][b]concat=n=2:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
        "-g", "10", "-keyint_min", "10", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the clip: {result.stderr[-500:]}")
    return out


class _MissesOnce:
    """Keeps nothing the first time and every detection after, the pattern of
    tests/test_second_chance.py: the miss is faked, the rest is real."""

    calls = 0

    def run(self, strong, weak, shifts, shape):
        _MissesOnce.calls += 1
        if _MissesOnce.calls == 1:
            return [[] for _ in strong], []
        kept = [list(f) for f in strong]
        return kept, [[d] for f in kept for d in f]


def _run(src, out, monkeypatch, **changes):
    """Run with the faked miss, keeping a copy of the first write and the
    segments of every write."""
    _MissesOnce.calls = 0
    monkeypatch.setattr(batch, "tracker_for", lambda settings: _MissesOnce())
    real = _WRITE_AND_JOIN
    writes = []
    first = out.parent / (out.stem + "_first.mp4")

    def spy(src_, dst_, *args, **kwargs):
        got = real(src_, dst_, *args, **kwargs)
        writes.append({"segments": args[3], "copy_src": kwargs.get("copy_src")})
        if len(writes) == 1 and got is not None and not got[3]:
            shutil.copyfile(dst_, first)
        return got

    monkeypatch.setattr(batch, "_write_and_join", spy)
    record = run_video(src, out, Settings(**SCOPED, **changes), serial_submit)
    return record, writes, first


def _frames(path: Path) -> list[np.ndarray]:
    info, times = probe(path), frame_times(path)
    return [f.copy() for f in decode_range(path, times, 0, len(times), info)]


def test_changed_gives_the_answer_all_gives(tmp_path, face_then_wall, monkeypatch):
    changed_out = tmp_path / "changed.mp4"
    all_out = tmp_path / "all.mp4"
    changed, writes, _ = _run(face_then_wall, changed_out, monkeypatch,
                              second_chance_scope="changed")
    whole, _, _ = _run(face_then_wall, all_out, monkeypatch, second_chance_scope="all")

    for record in (changed, whole):
        assert record.status == STATUS_DONE, record.error
        assert record.residual_faces_first_pass > 0, "the first pass was supposed to miss"
    for name in ("residual_faces", "residual_by_size", "frames", "frames_with_mask",
                 "second_chance_rounds", "second_chance_seeds_kept", "checked_frames",
                 "flat_boxes", "residual_list"):
        assert getattr(changed, name) == getattr(whole, name), name
    assert changed.second_chance_rounds == 1
    assert 0 < changed.second_chance_reencoded_frames < changed.frames
    # The record keeps saying how much came from the source untouched.
    assert changed.frames_copied == changed.frames
    # The second write copies from the first copy, not from the source.
    assert len(writes) == 2 and writes[1]["copy_src"] is not None

    times = frame_times(face_then_wall)
    for out, record in ((changed_out, changed), (all_out, whole)):
        delivered = out if out.is_file() else out.parent / "quarantine" / out.name
        assert verify(delivered, record.frames, times[-1] - times[0]) == ""


def test_frames_outside_the_changed_stretch_are_the_first_copy(tmp_path, face_then_wall,
                                                               monkeypatch):
    out = tmp_path / "changed.mp4"
    record, writes, first = _run(face_then_wall, out, monkeypatch)
    assert record.status == STATUS_DONE, record.error
    assert len(writes) == 2
    encoded = set()
    for seg in writes[1]["segments"]:
        if not seg.copy:
            encoded.update(range(seg.start, seg.end))
    assert 0 < len(encoded) < record.frames
    assert record.second_chance_reencoded_frames == len(encoded)

    delivered = out if out.is_file() else out.parent / "quarantine" / out.name
    final, before = _frames(delivered), _frames(first)
    assert len(final) == len(before) == record.frames
    outside = [i for i in range(record.frames) if i not in encoded]
    assert outside
    for i in outside:
        assert np.array_equal(final[i], before[i]), f"frame {i} is not the first copy's"
    # And the round did something where it encoded: the face is masked now.
    assert any(not np.array_equal(final[i], before[i]) for i in encoded)


def test_a_round_without_seeds_writes_nothing_more(tmp_path, video_with_face, monkeypatch):
    """The pipeline masks the face itself here, the check has nothing to seed
    with, and there is one write."""
    real = _WRITE_AND_JOIN
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(batch, "_write_and_join", counting)
    out = tmp_path / "clean.mp4"
    record = run_video(video_with_face, out,
                       Settings(device="cpu", check_output=True, second_chance=1,
                                copy_clean=False, chunk_seconds=0, encode_seconds=0),
                       serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_rounds == 0
    assert record.second_chance_reencoded_frames == 0
    assert len(calls) == 1


# ------------------------------------------------------------ why a round wrote it all

def test_a_scoped_round_that_wrote_ranges_names_no_fallback(tmp_path, face_then_wall,
                                                             monkeypatch):
    record, _, _ = _run(face_then_wall, tmp_path / "changed.mp4", monkeypatch)
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_ranges
    assert record.second_chance_scope_fallback is None


def test_the_record_counts_the_frames_the_re_track_changed(tmp_path, face_then_wall,
                                                           monkeypatch):
    record, writes, _ = _run(face_then_wall, tmp_path / "changed.mp4", monkeypatch)
    assert record.status == STATUS_DONE, record.error
    assert 0 < record.second_chance_changed_frames <= record.second_chance_reencoded_frames
    # The scope "all" does not compare masks, so it counts nothing.
    whole, _, _ = _run(face_then_wall, tmp_path / "all.mp4", monkeypatch,
                       second_chance_scope="all")
    assert whole.status == STATUS_DONE, whole.error
    assert whole.second_chance_rounds == 1
    assert whole.second_chance_changed_frames == 0


def test_the_scope_all_names_no_fallback(tmp_path, face_then_wall, monkeypatch):
    record, _, _ = _run(face_then_wall, tmp_path / "all.mp4", monkeypatch,
                        second_chance_scope="all")
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_rounds == 1
    assert record.second_chance_scope_fallback is None


def test_a_margin_over_the_whole_file_falls_back_as_all_changed(tmp_path, face_then_wall,
                                                                 monkeypatch):
    record, writes, _ = _run(face_then_wall, tmp_path / "wide.mp4", monkeypatch,
                             second_chance_margin_seconds=60.0)
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_rounds == 1
    assert record.second_chance_scope_fallback == "all_changed"
    assert record.second_chance_ranges == []
    assert record.second_chance_reencoded_frames == record.frames - record.frames_copied
    # The second write is a whole one, from the source.
    assert len(writes) == 2 and writes[1]["copy_src"] is None


def test_a_scoped_join_that_did_not_verify_falls_back_as_join_failed(tmp_path,
                                                                     face_then_wall,
                                                                     monkeypatch):
    _MissesOnce.calls = 0
    monkeypatch.setattr(batch, "tracker_for", lambda settings: _MissesOnce())
    real = _WRITE_AND_JOIN
    writes = []

    def failing_scoped(src_, dst_, *args, **kwargs):
        got = real(src_, dst_, *args, **kwargs)
        writes.append(kwargs.get("copy_src"))
        if kwargs.get("copy_src") is not None and got is not None:
            masked, by_kind, prevented, _, rest = got
            return masked, by_kind, prevented, "stubbed: the join did not verify", rest
        return got

    monkeypatch.setattr(batch, "_write_and_join", failing_scoped)
    record = run_video(face_then_wall, tmp_path / "join.mp4", Settings(**SCOPED),
                       serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_scope_fallback == "join_failed"
    assert record.second_chance_ranges == []
    # The first write, the scoped one that failed, then the whole file.
    assert len(writes) == 3 and writes[1] is not None and writes[2] is None


# ------------------------------------------------------------ the merge

def _job(start, end, frames, stride=1):
    """A result shaped the way `residual_job` returns one, from per frame
    truth: {frame: {"found": rows, "flat": {kind: n}, "zone": row or None}}."""
    found, zone, flat_frames = [], [], {}
    flat = {"face": 0, "screen": 0}
    checked = 0
    for i in range(start, end):
        if i % stride:
            continue
        checked += 1
        truth = frames.get(i, {})
        found.extend(truth.get("found", []))
        if truth.get("zone"):
            zone.append(truth["zone"])
        if truth.get("flat"):
            flat_frames[i] = dict(truth["flat"])
            for name, count in truth["flat"].items():
                flat[name] += count
    return {"start": start, "end": end, "stride": stride, "checked": checked,
            "flat": sum(flat.values()), "flat_by_kind": flat, "flat_frames": flat_frames,
            "hands": sum(1 for r in found if "hand" in r),
            "in_zone": sum(1 for r in found if "zone" in r),
            "found": found, "zone": zone,
            "hand_models": {"palm": "abc"}, "hand_compute": {"palm": "cpu"}}


def _face(frame, px=40, **extra):
    row = {"kind": "face", "frame": frame, "px": px, "x": 100.0, "y": 100.0,
           "w": float(px), "h": float(px), "score": 0.8, "applied": 0.1}
    row.update(extra)
    return row


def _screen(frame, px=120):
    return {"kind": "screen", "frame": frame, "px": px, "x": 300.0, "y": 200.0,
            "w": float(px), "h": float(px * 0.6), "score": 0.7, "applied": 0.0,
            "label": "tv"}


def test_merged_checks_summarise_like_a_full_recheck():
    first_truth = {
        1: {"found": [_face(1)], "flat": {"face": 1}},
        4: {"found": [_face(4, px=30, hand=0.9)], "zone": {"frame": 4, "changed": 0.2,
                                                           "zone_px": 900}},
        6: {"found": [_face(6), _face(6, px=20, zone=0.8)], "flat": {"screen": 2}},
        7: {"found": [_screen(7)], "zone": {"frame": 7, "changed": 0.05, "zone_px": 500}},
        8: {"found": [_screen(8)]},
        12: {"found": [_face(12, px=50)], "flat": {"face": 3}},
        15: {"found": [_face(15)], "zone": {"frame": 15, "changed": 0.2, "zone_px": 100}},
        18: {"found": [_screen(18), _face(18, px=26)], "flat": {"face": 1, "screen": 1}},
    }
    # The round encoded [5, 13) again; there the check now sees this instead.
    ranges = [[5, 13]]
    again_truth = {
        6: {"found": [_face(6, px=20, zone=0.8)], "flat": {"face": 1}},
        9: {"found": [_face(9, px=24, hand=0.7)],
            "zone": {"frame": 9, "changed": 0.3, "zone_px": 700}},
        12: {"flat": {"screen": 1}},
    }
    first = [_job(0, 10, first_truth), _job(10, 20, first_truth)]
    again = [_job(5, 10, again_truth), _job(10, 13, again_truth)]

    def inside(i):
        return 5 <= i < 13

    truth = {i: v for i, v in first_truth.items() if not inside(i)}
    truth.update(again_truth)
    full = [_job(0, 10, truth), _job(10, 20, truth)]

    settings = Settings()
    merged = summarise(merge_checks(first, again, ranges), settings)
    assert merged == summarise(full, settings)
    # And the merge did take something out and put something in.
    assert merged != summarise(first, settings)
    assert merged["frames_checked"] == 20


def test_merge_keeps_every_frame_when_nothing_was_rechecked():
    truth = {3: {"found": [_face(3)], "flat": {"face": 2}}}
    first = [_job(0, 10, truth)]
    settings = Settings()
    assert summarise(merge_checks(first, [], []), settings) == summarise(first, settings)


# ------------------------------------------------------------ the settings

def test_the_scope_defaults_to_changed_with_half_a_second_of_margin():
    s = Settings()
    assert s.second_chance_scope == "changed"
    assert s.second_chance_margin_seconds == 0.5
    assert s.to_dict()["second_chance_scope"] == "changed"


def test_an_unknown_scope_is_refused():
    with pytest.raises(SettingsError, match="second_chance_scope"):
        Settings(second_chance_scope="some")
    Settings(second_chance_scope="all")


def test_a_negative_margin_is_refused():
    with pytest.raises(SettingsError, match="second_chance_margin_seconds"):
        Settings(second_chance_margin_seconds=-1.0)
