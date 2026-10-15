"""Sources whose last frame was cut off mid-write.

In the first production run 13 camera files ended in a damaged last packet.
The demuxer still lists that packet, so frame_times counts it, but the decoder
cannot make a frame of it. The pipeline must write every frame that decodes
and say in the record how many it could not.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

import cli
from faceblur import batch
from faceblur.batch import run_video, serial_submit
from faceblur.pipeline import STATUS_DONE
from faceblur.segments import decode_range, frame_times
from faceblur.settings import Settings
from faceblur.video import probe

from conftest import run_ffmpeg

FAST = ["--det-sizes", "320", "--no-verify", "--min-track", "1", "--no-progress"]


def _last_packet_size(path) -> int:
    result = run_ffmpeg(["-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                         "-f", "framecrc", "-"])
    rows = [line for line in result.stdout.splitlines() if line.startswith("0,")]
    return int(rows[-1].split(",")[4])


@pytest.fixture(scope="session")
def video_cut_off(tmp_path_factory):
    """Two seconds at 10 fps, cut inside its last packet.

    With +faststart the index comes first and the only track's data runs to
    the end of the file, so the last packet is the last bytes. Keeping four of
    them leaves the index listing every packet while the decoder gets one
    frame fewer, which is what a camera stopped mid-write leaves behind.
    """
    out = tmp_path_factory.mktemp("clips") / "cut_off.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error",
                         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
                         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", str(out)])
    if result.returncode != 0:
        raise RuntimeError(f"could not build the test video: {result.stderr[-500:]}")
    os.truncate(out, os.path.getsize(out) - _last_packet_size(out) + 4)
    return out


def _decodable(path) -> int:
    times = frame_times(path)
    return sum(1 for _ in decode_range(path, times, 0, len(times), probe(path)))


def test_the_symptom_is_reproduced_by_the_fixture(video_cut_off):
    n = len(frame_times(video_cut_off))
    assert n == 20
    assert _decodable(video_cut_off) < n


def test_the_cli_writes_every_frame_that_decodes(video_cut_off, tmp_path):
    clip = tmp_path / "cam.mp4"
    shutil.copy(video_cut_off, clip)
    n, good = len(frame_times(clip)), _decodable(clip)
    out = tmp_path / "out"

    code = cli.main([str(clip), "-o", str(out), "--quarantine", "--workers", "1", *FAST])

    assert code == 0
    found = [p for p in (out / "cam_blurred.mp4", out / "quarantine" / "cam_blurred.mp4")
             if p.exists()]
    assert len(found) == 1
    copy = found[0]
    assert len(frame_times(copy)) == good
    record = json.loads((copy.parent / (copy.name + ".json")).read_text(encoding="utf-8"))
    assert record["status"] == STATUS_DONE
    assert record["frames"] == good
    assert record["source_frames_dropped"] == n - good >= 1


def test_a_clip_that_would_be_copied_whole_is_encoded_at_its_tail(video_cut_off, tmp_path):
    """No faces, so the whole clip is one clean stretch. Copying it would hand
    ffmpeg the damaged packet, so the stretch holding the end is encoded."""
    n, good = len(frame_times(video_cut_off)), _decodable(video_cut_off)
    out = tmp_path / "cam_blurred.mp4"

    record = run_video(video_cut_off, out, Settings(device="cpu", copy_clean=True),
                       serial_submit)

    assert record.status == STATUS_DONE, record.error
    assert record.source_frames_dropped == n - good >= 1
    assert record.frames == good
    assert record.frames_copied == 0
    assert len(frame_times(out)) == good


def test_an_intact_clip_drops_nothing_and_plans_as_before(video_silent, tmp_path, monkeypatch):
    calls = {"keyframes": 0}
    real = batch.keyframes

    def counting(*args, **kwargs):
        calls["keyframes"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(batch, "keyframes", counting)
    out = tmp_path / "silent_blurred.mp4"

    record = run_video(video_silent, out, Settings(device="cpu", copy_clean=True),
                       serial_submit)

    assert record.status == STATUS_DONE, record.error
    assert record.source_frames_dropped == 0
    assert record.frames == len(frame_times(video_silent))
    assert record.frames_copied == record.frames
    # The tail probe reuses the keyframes the write pass needs anyway.
    assert calls["keyframes"] == 1
