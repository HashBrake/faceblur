"""The per-file checks a stored copy must pass before it can replace its source."""
from __future__ import annotations

import pytest

from conftest import run_ffmpeg
from storage import checks
from storage.encode import encode_file


@pytest.fixture(scope="module")
def hevc_copy(video_long, tmp_path_factory):
    """video_long in HEVC, libx265 at crf 20: VMAF about 98 (the model's ceiling is ~98.6)."""
    dst = tmp_path_factory.mktemp("checks") / "hevc.mp4"
    encode_file(video_long, dst, "hevc", "software", quality=20)
    return dst


def test_run_all_passes(video_long, hevc_copy):
    report = checks.run_all(video_long, hevc_copy)
    assert report.passed, report.describe()
    assert [c.name for c in report.checks] == ["frame_count", "duration",
                                               "decodes_cleanly", "vmaf"]
    assert isinstance(report.vmaf_mean, float)
    assert report.vmaf_mean >= 90
    assert report.vmaf_min == report.vmaf_mean      # 12 s: one window, the whole file
    assert report.frames == 120
    assert len(report.describe().splitlines()) == 4


def test_one_frame_short_fails(video_long, tmp_path):
    short = tmp_path / "short.mp4"
    r = run_ffmpeg(["-y", "-v", "error", "-i", str(video_long), "-frames:v", "119",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    str(short)])
    assert r.returncode == 0, r.stderr
    report = checks.run_all(video_long, short)
    by_name = {c.name: c for c in report.checks}
    assert not report.passed
    assert not by_name["frame_count"].passed
    assert not by_name["duration"].passed
    assert not by_name["vmaf"].passed
    assert by_name["vmaf"].detail == "skipped: earlier check failed"
    assert report.vmaf_mean is None and report.vmaf_min is None


def test_vmaf_sanity(video_silent, tmp_path):
    same = checks.vmaf(video_silent, video_silent)
    assert same.mean >= 98
    assert same.frames == 20
    worse = tmp_path / "worse.mp4"
    r = run_ffmpeg(["-y", "-v", "error", "-i", str(video_silent), "-c:v", "libx264",
                    "-crf", "51", str(worse)])
    assert r.returncode == 0, r.stderr
    assert checks.vmaf(video_silent, worse).mean < same.mean - 10


def test_vmaf_windows(video_long, hevc_copy):
    result = checks.vmaf(video_long, hevc_copy, windows=3, window_s=0.5)
    assert len(result.per_window) == 3
    assert 12 <= result.frames <= 18
    assert min(result.per_window) <= result.mean <= max(result.per_window)


def test_vmaf_check_floor():
    result = checks.VmafResult(mean=96.0, per_window=[97.0, 95.0], frames=10)
    assert checks.vmaf_check(result, floor=95.0).passed
    assert not checks.vmaf_check(result, floor=97.0).passed


def test_duration_tolerance():
    from storage.probe import VideoFacts
    src = VideoFacts("h264", 320, 240, 10.0, 120, 12.0, 1)
    assert checks.duration(src, VideoFacts("hevc", 320, 240, 10.0, 120, 12.04, 1)).passed
    assert not checks.duration(src, VideoFacts("hevc", 320, 240, 10.0, 119, 11.9, 1)).passed


def test_decodes_cleanly(hevc_copy):
    check = checks.decodes_cleanly(hevc_copy)
    assert check.passed, check.detail
    assert "120" in check.detail


def test_decodes_cleanly_fails_on_truncated(video_silent, tmp_path):
    data = video_silent.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[:int(len(data) * 0.7)])
    assert not checks.decodes_cleanly(cut).passed
