"""Video facts from packets, decoding nothing."""
from __future__ import annotations

import pytest

from faceblur.segments import frame_times
from faceblur.video import VideoError
from storage.probe import frame_pts_seconds, probe


def test_probe(video_silent):
    facts = probe(video_silent)
    assert facts.codec == "h264"
    assert (facts.width, facts.height) == (320, 240)
    assert facts.frames == 20
    assert facts.fps == pytest.approx(10.0)
    assert abs(facts.duration_s - 2.0) <= 0.1
    assert facts.size_bytes == video_silent.stat().st_size


def test_probe_with_audio(video_with_audio):
    facts = probe(video_with_audio)
    assert facts.frames == 30


def test_frame_pts_seconds(video_silent):
    times = frame_pts_seconds(video_silent)
    assert len(times) == 20
    assert all(b > a for a, b in zip(times, times[1:]))
    assert times == pytest.approx(frame_times(video_silent))


def test_not_a_video(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("not a video")
    with pytest.raises(VideoError):
        probe(bad)
    with pytest.raises(VideoError):
        frame_pts_seconds(bad)
