"""Copies keep the source's frame timestamps, exactly.

The source here is a small variable-rate video: frames a few milliseconds
early or late, and one dropped frame leaving a double gap, the way the
camera records. Every copy the pipeline writes from it, and every copy the
retime tool repairs, must carry the same pts in the same time base."""
from __future__ import annotations

import importlib.util
import json
import random
from fractions import Fraction
from pathlib import Path

import av
import pytest

from faceblur.batch import run_video, serial_submit
from faceblur.detect import Detection
from faceblur.pipeline import STATUS_DONE
from faceblur.segments import verify
from faceblur.settings import Settings
from faceblur.timestamps import (TimingError, check_retimed, copy_layout, decode_check,
                                 retime, source_timing)

from conftest import audio_streams, run_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
FAST = Settings(det_sizes=(320,), verify=False, min_track=1, tail=1, chunk_seconds=0.5,
                min_copy_seconds=0.3, encoder="x264")
N = 40
TB = 90000
DROPPED = 17                     # the camera dropped the frame before this one
EARLY = 30                       # this frame came 44 ms after the one before


def _vfr_pts(n: int = N, start: int = 0, seed: int = 3) -> list[int]:
    """10 fps nominal on a 1/90000 clock, jittered up to 0.8 ms, with one gap."""
    rnd = random.Random(seed)
    pts, t = [], start
    for i in range(n):
        pts.append(t)
        step = 9000 + rnd.randint(-70, 70)
        if i + 1 == DROPPED:
            step += 9000
        if i + 1 == EARLY:
            step = 4000             # two frames on one tick of a 1/10 s clock
        t += step
    return pts


def _cfr(path: Path, bf: int = 0, seconds: float = N / 10, codec: str = "libx264") -> Path:
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=size=128x96:rate=10:duration={seconds}",
                    "-c:v", codec, "-preset", "ultrafast", "-crf", "18", "-bf", str(bf),
                    "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", str(path)])
    assert r.returncode == 0, r.stderr
    return path


def _stamp(src: Path, dst: Path, pts: list[int]) -> Path:
    """Remux `src` (no B-frames) with these pts, one per packet."""
    with av.open(str(src)) as cin, av.open(str(dst), "w", format="mp4") as out:
        vin = cin.streams.video[0]
        vout = out.add_stream_from_template(vin)
        vout.time_base = Fraction(1, TB)
        i = 0
        for p in cin.demux(vin):
            if p.size == 0:
                continue
            nxt = pts[i + 1] - pts[i] if i + 1 < len(pts) else 9000
            p.pts = p.dts = pts[i]
            p.duration = nxt
            p.time_base = Fraction(1, TB)
            p.stream = vout
            out.mux(p)
            i += 1
    assert i == len(pts)
    return dst


@pytest.fixture(scope="module")
def vfr_source(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("vfr")
    return _stamp(_cfr(d / "cfr.mp4"), d / "vfr_source.mp4", _vfr_pts())


def _pts(path: Path) -> list[int]:
    return list(source_timing(path).pts)


def test_the_synthetic_source_is_variable_rate(vfr_source):
    t = source_timing(vfr_source)
    assert t.time_base == Fraction(1, TB)
    assert list(t.pts) == _vfr_pts()
    steps = [b - a for a, b in zip(t.pts, t.pts[1:])]
    assert max(steps) > 17000 and len(set(steps)) > 10


# ------------------------------------------------------------ retime itself

def test_retime_puts_the_source_clock_on_a_constant_rate_copy(vfr_source, tmp_path):
    copy = _cfr(tmp_path / "copy.mp4")
    assert _pts(copy) != _vfr_pts()
    out = tmp_path / "retimed.mp4"
    res = retime(copy, vfr_source, out, allow_reorder=False)
    assert _pts(out) == _vfr_pts()
    assert source_timing(out).time_base == source_timing(vfr_source).time_base
    assert res.frames == N and res.max_error_after_ms == 0.0
    assert res.max_error_before_ms > 50          # the dropped frame's step, lost
    # The pictures are the copy's packets, byte for byte.
    with av.open(str(copy)) as a, av.open(str(out)) as b:
        pa = [bytes(p) for p in a.demux(a.streams.video[0]) if p.size]
        pb = [bytes(p) for p in b.demux(b.streams.video[0]) if p.size]
    assert pa == pb
    assert decode_check(out, N) == ""
    assert audio_streams(out) == []
    # faststart: the index comes before the media data.
    head = out.read_bytes()
    assert head.find(b"moov") < head.find(b"mdat")


def test_retime_keeps_a_nonzero_start(tmp_path):
    src = _stamp(_cfr(tmp_path / "c.mp4"), tmp_path / "s.mp4", _vfr_pts(start=123456))
    out = tmp_path / "r.mp4"
    retime(_cfr(tmp_path / "copy.mp4"), src, out)
    assert _pts(out) == _vfr_pts(start=123456)


def test_retime_refuses_different_frame_counts(vfr_source, tmp_path):
    short = _cfr(tmp_path / "short.mp4", seconds=3.9)
    with pytest.raises(TimingError, match="differ"):
        retime(short, vfr_source, tmp_path / "x.mp4")
    assert not (tmp_path / "x.mp4").exists()
    assert not (tmp_path / "x.mp4.part").exists()


def test_retime_takes_the_first_frames_of_a_longer_source_only_when_told(vfr_source, tmp_path):
    short = _cfr(tmp_path / "short.mp4", seconds=3.9)        # 39 frames
    out = tmp_path / "r.mp4"
    retime(short, vfr_source, out, frames=39)
    assert _pts(out) == _vfr_pts()[:39]


def test_retime_refuses_a_copy_with_b_frames_when_asked(vfr_source, tmp_path):
    bf = _cfr(tmp_path / "bf.mp4", bf=2)
    assert copy_layout(bf).reorder >= 1
    with pytest.raises(TimingError, match="presentation order"):
        retime(bf, vfr_source, tmp_path / "x.mp4", allow_reorder=False)


def test_retime_handles_b_frames_when_allowed(vfr_source, tmp_path):
    bf = _cfr(tmp_path / "bf.mp4", bf=2)
    out = tmp_path / "r.mp4"
    retime(bf, vfr_source, out)
    assert _pts(out) == _vfr_pts()
    assert decode_check(out, N) == ""


def test_check_retimed_catches_a_wrong_clock(vfr_source, tmp_path):
    copy = _cfr(tmp_path / "copy.mp4")
    assert "off the source" in check_retimed(copy, _vfr_pts(), Fraction(1, TB))


# ------------------------------------------------------- the pipeline's write

def _face_at(frames) -> list:
    box = Detection(x=30, y=20, w=40, h=40, score=0.9, kind="face")
    return {i: [box] for i in frames}


def test_pipeline_copy_keeps_the_source_clock_copied_and_encoded(vfr_source, tmp_path,
                                                                 monkeypatch):
    """A face on frames 12-22 (around the dropped frame): those keyframe
    intervals are encoded, the rest stream copied, and the joined file
    carries every source pts."""
    import faceblur.batch as batch
    real = batch._write_and_join
    seen = {}

    def masks(src, dst, info, times, settings, segments, per_frame, *a, **k):
        faces = _face_at(range(12, 23))
        per_frame = [faces.get(i, []) for i in range(len(per_frame))]
        from faceblur.segments import plan_segments, split_long, keyframes
        keys = keyframes(src, times)
        segments = plan_segments([bool(f) for f in per_frame], keys, len(per_frame), 3, True)
        seen["segments"] = segments
        return real(src, dst, info, times, settings, segments, per_frame, *a, **k)

    monkeypatch.setattr(batch, "_write_and_join", masks)
    dst = tmp_path / "out.mp4"
    record = run_video(vfr_source, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    kinds = {s.copy for s in seen["segments"]}
    assert kinds == {True, False}, seen["segments"]
    assert _pts(dst) == _vfr_pts()
    assert source_timing(dst).time_base == source_timing(vfr_source).time_base
    assert verify(dst, N) == ""
    assert decode_check(dst, N) == ""


def test_pipeline_copy_keeps_the_source_clock_when_all_encoded(vfr_source, tmp_path):
    dst = tmp_path / "out.mp4"
    record = run_video(vfr_source, dst, FAST.with_changes(copy_clean=False), serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.frames_copied == 0
    assert _pts(dst) == _vfr_pts()


def test_pipeline_copy_keeps_the_source_clock_when_all_copied(vfr_source, tmp_path):
    dst = tmp_path / "out.mp4"
    record = run_video(vfr_source, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert record.frames_copied == N
    assert _pts(dst) == _vfr_pts()


def test_pipeline_keeps_the_audio_with_the_source_clock(video_with_audio, tmp_path):
    dst = tmp_path / "out.mp4"
    record = run_video(video_with_audio, dst, FAST, serial_submit)
    assert record.status == STATUS_DONE, record.error
    assert audio_streams(dst) != []
    assert _pts(dst) == _pts(video_with_audio)


# --------------------------------------------------------------- the patch

def _row(frame, x, y, px=30):
    return {"kind": "face", "frame": frame, "px": px, "x": x, "y": y, "w": px, "h": px,
            "score": 0.9, "landmarks": None, "applied": 0.05}


def test_patch_puts_the_source_clock_on_its_output(vfr_source, tmp_path):
    """The patch masks the delivered copy (constant rate), and its output
    carries the source's timestamps, both when it writes and when it has
    nothing to mask."""
    from faceblur.patch import patch
    copy = _cfr(tmp_path / "copy.mp4")
    out = tmp_path / "patched.mp4"
    record = patch(vfr_source, copy, out, FAST, hold=2, rows=[_row(15, 40, 48)])
    assert record["error"] == "", record["error"]
    assert _pts(out) == _vfr_pts()
    assert audio_streams(out) == []
    out2 = tmp_path / "untouched.mp4"
    record = patch(vfr_source, copy, out2, FAST, rows=[])
    assert record["error"] == "" and record["masked"] == 0
    assert _pts(out2) == _vfr_pts()


# ------------------------------------------------------- the cloud retime tool

def _tool():
    spec = importlib.util.spec_from_file_location("retime_tool", ROOT / "cloud" / "retime.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_retimes_verifies_and_records(vfr_source, tmp_path):
    tool = _tool()
    copy = _cfr(tmp_path / "copy.mp4")
    out = tmp_path / "out.mp4"
    rec = tool.retime_file(vfr_source, copy, out)
    assert rec["status"] == "retimed", rec
    assert rec["frames"] == N and rec["max_error_after_ms"] == 0.0
    assert rec["max_error_before_ms"] > 50
    assert _pts(out) == _vfr_pts()
    json.dumps(rec)
    # A second pass finds nothing to do.
    rec2 = tool.retime_file(vfr_source, out, tmp_path / "out2.mp4")
    assert rec2["status"] == "already" and not (tmp_path / "out2.mp4").exists()


def test_tool_refuses_different_counts_and_b_frames(vfr_source, tmp_path):
    tool = _tool()
    rec = tool.retime_file(vfr_source, _cfr(tmp_path / "s.mp4", seconds=3.9), tmp_path / "o.mp4")
    assert rec["status"] == "refused" and "differ" in rec["error"]
    rec = tool.retime_file(vfr_source, _cfr(tmp_path / "b.mp4", bf=2), tmp_path / "o.mp4")
    assert rec["status"] == "refused" and "presentation order" in rec["error"]
    assert not (tmp_path / "o.mp4").exists()


# ------------------------------------------- compose: only the index moves

def _local(path: Path):
    data = path.read_bytes()
    return lambda off, n: data[off:off + n]


def _fragmented(src: Path, dst: Path) -> Path:
    with av.open(str(src)) as cin, av.open(
            str(dst), "w", format="mp4",
            options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}) as out:
        vin = cin.streams.video[0]
        vout = out.add_stream_from_template(vin)
        vout.time_base = vin.time_base
        for p in cin.demux(vin):
            if p.size:
                p.stream = vout
                out.mux(p)
    return dst


def _assemble(tool, copy: Path, source: Path, tmp: Path) -> tuple[bytes, bytes]:
    """What compose does, on local files: (header, the copy's payload)."""
    si, ci = tmp / "si.mp4", tmp / "ci.mp4"
    tool.sparse_index(_local(source), source.stat().st_size, si)
    cboxes = tool.sparse_index(_local(copy), copy.stat().st_size, ci)
    timing = source_timing(si)
    kinds = [k for k, _, _, _ in cboxes]
    _, mo, ms, _ = cboxes[kinds.index(b"moov")]
    _, do, ds, dh = cboxes[-1]
    index = ci.read_bytes()[:do + dh]
    moov = tool.retimed_moov(index[mo:mo + ms], list(timing.pts), timing.last_duration,
                             timing.time_base, 0)
    data = copy.read_bytes()
    return index[:mo] + moov + index[mo + ms:do + dh], data[do + dh:do + ds]


def test_compose_moves_only_the_index(vfr_source, tmp_path):
    """The tool's compose mode reads only the indexes (sparse files, holes
    for the pictures) of a fragmented source, as the camera writes, and of a
    faststart copy, and assembles a new index + the copy's payload bytes.
    The result carries the source's pts, the copy's packets and key frames,
    and decodes."""
    tool = _tool()
    frag = _fragmented(vfr_source, tmp_path / "frag_source.mp4")
    assert _pts(frag) == _vfr_pts()
    copy = tmp_path / "copy.mp4"
    r = run_ffmpeg(["-y", "-v", "error", "-i", str(_cfr(tmp_path / "c0.mp4")), "-c", "copy",
                    "-movflags", "+faststart", str(copy)])
    assert r.returncode == 0
    si = tmp_path / "si_check.mp4"
    boxes = tool.sparse_index(_local(frag), frag.stat().st_size, si)
    assert sum(1 for b in boxes if b[0] == b"moof") > 1
    assert source_timing(si) == source_timing(frag)
    assert si.read_bytes() != frag.read_bytes()                # holes where the pictures were
    header, payload = _assemble(tool, copy, frag, tmp_path)
    out = tmp_path / "assembled.mp4"
    out.write_bytes(header + payload)
    assert _pts(out) == _vfr_pts()
    assert source_timing(out).time_base == Fraction(1, TB)
    with av.open(str(copy)) as a, av.open(str(out)) as b:
        pa = [(bytes(p), p.is_keyframe) for p in a.demux(a.streams.video[0]) if p.size]
        pb = [(bytes(p), p.is_keyframe) for p in b.demux(b.streams.video[0]) if p.size]
    assert pa == pb and sum(k for _, k in pa) == 4
    assert decode_check(out, N) == ""
    assert verify(out, N) == ""


def test_compose_refuses_a_source_that_does_not_start_at_zero(tmp_path):
    tool = _tool()
    src = _stamp(_cfr(tmp_path / "c.mp4"), tmp_path / "s.mp4", _vfr_pts(start=4500))
    copy = tmp_path / "copy.mp4"
    run_ffmpeg(["-y", "-v", "error", "-i", str(_cfr(tmp_path / "c0.mp4")), "-c", "copy",
                "-movflags", "+faststart", str(copy)])
    with pytest.raises(TimingError, match="not 0"):
        _assemble(tool, copy, src, tmp_path)


def test_compose_takes_the_source_time_scale(tmp_path):
    """Some sources are on 1/15360, the copies on 1/90000."""
    tool = _tool()
    pts = [p * 15360 // 90000 for p in _vfr_pts()]
    src = tmp_path / "s.mp4"
    with av.open(str(_cfr(tmp_path / "c.mp4"))) as cin, av.open(str(src), "w", format="mp4") as out:
        vin = cin.streams.video[0]
        vout = out.add_stream_from_template(vin)
        vout.time_base = Fraction(1, 15360)
        for i, p in enumerate(q for q in cin.demux(vin) if q.size):
            p.pts = p.dts = pts[i]
            p.duration = (pts[i + 1] - pts[i]) if i + 1 < N else 1536
            p.time_base = Fraction(1, 15360)
            p.stream = vout
            out.mux(p)
    assert source_timing(src).time_base == Fraction(1, 15360)
    copy = tmp_path / "copy.mp4"
    run_ffmpeg(["-y", "-v", "error", "-i", str(_cfr(tmp_path / "c0.mp4")), "-c", "copy",
                "-movflags", "+faststart", str(copy)])
    header, payload = _assemble(tool, copy, src, tmp_path)
    out = tmp_path / "assembled.mp4"
    out.write_bytes(header + payload)
    assert source_timing(out) == source_timing(src)
    assert decode_check(out, N) == ""
