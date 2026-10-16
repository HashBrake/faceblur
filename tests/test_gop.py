"""The encoded pieces keep a keyframe every second (DECISIONS D18).

The scoped second chance cuts the first copy at its keyframes. Without a set
GOP length NVENC put one only at each piece's start, so on the 004100 sample
11 seeds became 684 re-encoded frames of 984. These tests hold the encoder to
the cadence and the record to the ranges the round re-encoded.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from faceblur.pipeline import STATUS_DONE
from faceblur.segments import SegmentEncoder, decode_range, frame_times, keyframes
from faceblur.video import probe

from conftest import run_ffmpeg
from test_second_chance_scope import _run, face_then_wall  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def six_seconds(tmp_path_factory) -> Path:
    """Six seconds of a moving test pattern at 10 fps."""
    out = tmp_path_factory.mktemp("gop") / "six.mp4"
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=s=320x240:r=10:d=6", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)])
    if r.returncode != 0:
        raise RuntimeError(f"could not build the clip: {r.stderr[-500:]}")
    return out


def _nvenc_works(tmp_path: Path) -> str:
    """Empty if NVENC can encode here, else why not."""
    r = run_ffmpeg(["-encoders"])
    if "h264_nvenc" not in r.stdout:
        return "this ffmpeg does not list h264_nvenc"
    r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x240:r=10:d=1", "-c:v", "h264_nvenc",
                    str(tmp_path / "probe.mp4")])
    if r.returncode != 0:
        return f"a probe encode with h264_nvenc failed: {r.stderr.strip()[-200:]}"
    return ""


@pytest.mark.parametrize("encoder", ["x264", "nvenc"])
def test_pieces_have_a_keyframe_every_second(tmp_path, six_seconds, encoder):
    if encoder == "nvenc":
        why = _nvenc_works(tmp_path)
        if why:
            pytest.skip(why)
    info, times = probe(six_seconds), frame_times(six_seconds)
    out = tmp_path / f"{encoder}.mp4"
    enc = SegmentEncoder(out, info, 23, "ultrafast", encoder, 23, gop_seconds=1.0)
    for frame in decode_range(six_seconds, times, 0, len(times), info):
        enc.write(frame)
    assert enc.close() == ""
    out_times = frame_times(out)
    assert len(out_times) == 60
    step = round(info.fps)
    assert keyframes(out, out_times) == list(range(0, 60, step))


def test_the_record_lists_the_ranges_the_round_encoded(tmp_path, face_then_wall,  # noqa: F811
                                                       monkeypatch):
    out = tmp_path / "changed.mp4"
    record, writes, _ = _run(face_then_wall, out, monkeypatch)
    assert record.status == STATUS_DONE, record.error
    assert len(writes) == 2
    encoded = []
    for seg in writes[1]["segments"]:
        if seg.copy:
            continue
        if encoded and encoded[-1][1] == seg.start:
            encoded[-1][1] = seg.end
        else:
            encoded.append([seg.start, seg.end])
    assert encoded
    assert record.second_chance_ranges == encoded
    assert sum(b - a for a, b in encoded) == record.second_chance_reencoded_frames
    assert record.to_dict()["second_chance_ranges"] == encoded


def test_the_scope_all_lists_no_ranges(tmp_path, face_then_wall, monkeypatch):  # noqa: F811
    record, _, _ = _run(face_then_wall, tmp_path / "all.mp4", monkeypatch,
                        second_chance_scope="all")
    assert record.status == STATUS_DONE, record.error
    assert record.second_chance_rounds == 1
    assert record.second_chance_ranges == []
