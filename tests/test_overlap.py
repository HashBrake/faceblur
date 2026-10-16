"""The check of each piece runs while the next pieces encode (`check_overlap`).

The encode is CPU work and the check is GPU work, and one after the other the
GPU idled through the whole encode (docs/bench/profile_2026-10-16.md). These
tests hold the overlapped order to the old order's answer, frame for frame,
and to the old order's promises: nothing left behind when a run stops, and no
more NVENC sessions than allowed.
"""
from __future__ import annotations

import multiprocessing
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from faceblur import batch, verify
from faceblur.batch import PoolSubmit, run_video, serial_stream, serial_submit
from faceblur.pipeline import STATUS_DONE, STATUS_STOPPED
from faceblur.segments import Segment, decode_range, frame_times
from faceblur.segments import verify as verify_file
from faceblur.settings import Settings
from faceblur.video import probe, unchecked_path

from conftest import run_ffmpeg

# Copies on and one second pieces, so the copy is several encoded pieces and
# several copied ones, and the check runs in several jobs. One second of clean
# keyframe interval is enough to copy, so the clip can stay short.
PIECES = dict(device="cpu", check_output=True, quarantine=True, copy_clean=True,
              chunk_seconds=1, encode_seconds=1, min_copy_seconds=1)

SAME = ("checked_frames", "residual_faces", "residual_by_size", "residual_list", "flat_boxes",
        "residual_hands", "zone_list", "held_back_for", "quarantined")


@pytest.fixture(scope="module")
def face_wall_face(tmp_path_factory, lena) -> Path:
    """A grey wall for three seconds, a still face for one, the wall for four
    more, 10 fps, a keyframe every second. The masks reach a second or two
    into the wall either side of the face, which leaves a clean stretch to
    copy at each end and one second pieces to encode in the middle."""
    out = tmp_path_factory.mktemp("clips") / "face_wall_face.mp4"
    face = "scale=160:-2,pad=640:640:240:240:gray,setsar=1,format=yuv420p"
    result = run_ffmpeg([
        "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=640x640:r=10:d=3",
        "-loop", "1", "-framerate", "10", "-t", "1", "-i", str(lena),
        "-f", "lavfi", "-i", "color=c=gray:s=640x640:r=10:d=4",
        "-filter_complex",
        f"[0:v]setsar=1,format=yuv420p[a];[1:v]{face}[b];[2:v]setsar=1,format=yuv420p[c];"
        "[a][b][c]concat=n=3:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
        "-g", "10", "-keyint_min", "10", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the clip: {result.stderr[-500:]}")
    return out


def _delivered(out: Path, record) -> Path:
    return out.parent / "quarantine" / out.name if record.quarantined else out


def _segments_of(monkeypatch) -> list:
    """Spy on the writes: the segments of each, in order."""
    real = batch._write_and_join
    writes = []

    def spy(*args, **kwargs):
        writes.append(args[5])
        return real(*args, **kwargs)

    monkeypatch.setattr(batch, "_write_and_join", spy)
    return writes


def test_overlap_gives_the_answer_the_old_order_gives(tmp_path, face_wall_face, monkeypatch):
    writes = _segments_of(monkeypatch)
    on_out, off_out = tmp_path / "on.mp4", tmp_path / "off.mp4"
    on = run_video(face_wall_face, on_out, Settings(**PIECES, check_overlap=True),
                   serial_submit)
    segments = writes[0]
    off = run_video(face_wall_face, off_out, Settings(**PIECES, check_overlap=False),
                    serial_submit)

    assert sum(1 for s in segments if s.copy) >= 2, "the clip was supposed to copy pieces"
    assert sum(1 for s in segments if not s.copy) >= 2, "and to encode pieces"
    for record in (on, off):
        assert record.status == STATUS_DONE, record.error
        assert record.checked_frames == record.frames
    for name in SAME:
        assert getattr(on, name) == getattr(off, name), name
    assert on.overlap_check_seconds > 0 and off.overlap_check_seconds == 0
    times = frame_times(face_wall_face)
    for out, record in ((on_out, on), (off_out, off)):
        assert verify_file(_delivered(out, record), record.frames, times[-1] - times[0]) == ""


class _MissesOnce:
    """Keeps nothing the first time and every detection after, the pattern of
    tests/test_second_chance_scope.py: the miss is faked, the rest is real."""

    calls = 0

    def run(self, strong, weak, shifts, shape):
        _MissesOnce.calls += 1
        if _MissesOnce.calls == 1:
            return [[] for _ in strong], []
        kept = [list(f) for f in strong]
        return kept, [[d] for f in kept for d in f]


def _scoped(src, out, monkeypatch, overlap):
    """The faked miss, and how often the check ran and how often it read the
    whole copy again (`plan_jobs` probes it and cuts it into jobs)."""
    _MissesOnce.calls = 0
    monkeypatch.setattr(batch, "tracker_for", lambda settings: _MissesOnce())
    calls = {"check": 0, "plan": 0}
    real_check, real_plan = verify.check, verify.plan_jobs

    def counting_check(*args, **kwargs):
        calls["check"] += 1
        return real_check(*args, **kwargs)

    def counting_plan(*args, **kwargs):
        calls["plan"] += 1
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(batch, "check", counting_check)
    monkeypatch.setattr(verify, "plan_jobs", counting_plan)
    record = run_video(src, out, Settings(**PIECES, second_chance=1, check_overlap=overlap),
                       serial_submit)
    monkeypatch.setattr(verify, "plan_jobs", real_plan)
    monkeypatch.setattr(batch, "check", real_check)
    return record, calls


def test_the_scoped_second_chance_gives_the_same_answer(tmp_path, face_wall_face,
                                                        monkeypatch):
    on, on_calls = _scoped(face_wall_face, tmp_path / "on.mp4", monkeypatch, True)
    off, off_calls = _scoped(face_wall_face, tmp_path / "off.mp4", monkeypatch, False)
    for record in (on, off):
        assert record.status == STATUS_DONE, record.error
        assert record.residual_faces_first_pass > 0, "the first pass was supposed to miss"
        assert record.second_chance_rounds == 1
    assert 0 < on.second_chance_reencoded_frames < on.frames, "the round was to be scoped"
    for name in SAME + ("residual_faces_first_pass", "second_chance_ranges",
                        "second_chance_seeds_kept", "masked_mean"):
        assert getattr(on, name) == getattr(off, name), name
    times = frame_times(face_wall_face)
    for name, record in (("on.mp4", on), ("off.mp4", off)):
        delivered = _delivered(tmp_path / name, record)
        assert verify_file(delivered, record.frames, times[-1] - times[0]) == ""
    # `check` is called once a round either way: with overlap it only sums
    # the pieces' results and never plans, decodes or detects on the whole
    # copy again.
    assert off_calls == {"check": 2, "plan": 2}
    assert on_calls == {"check": 2, "plan": 0}


def test_a_piece_decodes_to_the_joined_copy(tmp_path, face_wall_face, monkeypatch):
    """The check reads a piece before the join and the joined copy after it;
    the join is a concat without re-encoding, so the pixels are the same."""
    kept = tmp_path / "pieces"
    kept.mkdir()
    real_concat = batch.concat
    names = []

    def keeping(parts, *args, **kwargs):
        for part in parts:
            shutil.copyfile(part, kept / Path(part).name)
            names.append(Path(part).name)
        return real_concat(parts, *args, **kwargs)

    monkeypatch.setattr(batch, "concat", keeping)
    writes = _segments_of(monkeypatch)
    out = tmp_path / "out.mp4"
    settings = Settings(**{**PIECES, "check_output": False, "quarantine": False})
    record = run_video(face_wall_face, out, settings, serial_submit)
    assert record.status == STATUS_DONE, record.error

    info, times = probe(out), frame_times(out)
    joined = [f.copy() for f in decode_range(out, times, 0, len(times), info)]
    encoded = 0
    for seg, name in zip(writes[0], names):
        piece = kept / name
        frames = [f.copy() for f in decode_range(piece, frame_times(piece), 0, seg.frames,
                                                 probe(piece))]
        assert len(frames) == seg.frames
        for offset, frame in enumerate(frames):
            assert np.array_equal(frame, joined[seg.start + offset]), (name, offset)
        encoded += not seg.copy
    assert encoded >= 2


def test_a_run_cancelled_mid_write_leaves_nothing(tmp_path, face_wall_face, monkeypatch):
    stop = threading.Event()
    real = batch.encode_job

    def then_stop(job):
        got = real(job)
        stop.set()
        return got

    monkeypatch.setattr(batch, "encode_job", then_stop)
    out = tmp_path / "out.mp4"
    record = run_video(face_wall_face, out, Settings(**PIECES), serial_submit, cancel=stop)
    assert stop.is_set()
    assert record.status == STATUS_STOPPED
    assert not out.exists()
    assert not unchecked_path(out).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [], list(tmp_path.iterdir())


# ------------------------------------------------------------ the job stream

def _fake_job(job: dict) -> dict:
    """Stands in for `write_or_check_job`: half a second of nothing, and when."""
    t0 = time.time()
    time.sleep(0.5)
    t1 = time.time()
    if job.get("kind") == "check":
        return {"found": [], "seconds": t1 - t0, "start": job["start"], "end": job["end"],
                "t": (t0, t1)}
    return {"segment": job["segment"], "out": job["out"], "t": (t0, t1)}


def _drive(monkeypatch, submit, encode_limit, workers):
    monkeypatch.setattr(batch, "write_or_check_job", _fake_job)
    monkeypatch.setattr(batch, "probe", lambda path: None)
    monkeypatch.setattr(batch, "frame_times", lambda path: [])
    encodes = [{"segment": Segment(k * 10, k * 10 + 10, False), "out": f"seg{k}.mp4"}
               for k in range(4)]
    copies = [{"segment": Segment(40, 50, True)}]
    checks = {"chunk": 5, "stride": 1, "hands": None, "results": [], "complete": True}
    results, feed = batch._overlapped(submit, encodes, encode_limit, workers, checks,
                                      "src.mp4", None, [], None, copies, 50)
    order = []
    for result in results:
        if "found" in result:
            order.append(("check", result["start"], result["t"]))
        else:
            feed(result)
            order.append(("encode", result["segment"].start, result["t"]))
    return order


def test_the_serial_stream_checks_each_piece_before_the_next_encode(monkeypatch):
    order = _drive(monkeypatch, serial_submit, 1, 1)
    assert [(kind, start) for kind, start, _ in order] == [
        ("check", 40), ("check", 45),
        ("encode", 0), ("check", 0), ("check", 5),
        ("encode", 10), ("check", 10), ("check", 15),
        ("encode", 20), ("check", 20), ("check", 25),
        ("encode", 30), ("check", 30), ("check", 35)]
    # And the stream is lazy: serial_submit itself is not, which is why the
    # overlap asks for the stream.
    assert list(serial_stream(lambda j: j, iter([1, None, 2]))) == [1, 2]


def test_the_pool_stream_keeps_the_encode_limit_and_the_pool_busy(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(3) as pool:
        submit = PoolSubmit(pool)
        submit.POLL_SECONDS = 0.2
        order = _drive(monkeypatch, submit, 1, 3)
    encodes = sorted(t for kind, _, t in order if kind == "encode")
    checks = [t for kind, _, t in order if kind == "check"]
    assert len(encodes) == 4 and len(checks) == 10
    # One NVENC session allowed: no two encodes ever ran at once.
    for (_, end), (start, _) in zip(encodes, encodes[1:]):
        assert start >= end
    # And the checks ran beside the encodes rather than after them all.
    last_encode = encodes[-1][1]
    assert sum(1 for t in checks if t[0] < last_encode) >= 6
