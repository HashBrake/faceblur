"""The storage tools' ffmpeg: where it comes from and what it can do."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from storage import ffmpeg


@pytest.fixture
def fresh_cache():
    ffmpeg.ffmpeg_exe.cache_clear()
    yield
    ffmpeg.ffmpeg_exe.cache_clear()


def test_ffmpeg_exe_is_a_file():
    assert Path(ffmpeg.ffmpeg_exe()).is_file()


def test_tools_override(tmp_path, monkeypatch, fresh_cache):
    fake = tmp_path / "ffmpeg-fake.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(ffmpeg, "TOOLS_DIR", tmp_path)
    assert ffmpeg.ffmpeg_exe() == str(fake)


def test_tools_override_bin_folder(tmp_path, monkeypatch, fresh_cache):
    fake = tmp_path / "ffmpeg-8.0-full_build" / "bin" / "ffmpeg.exe"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"")
    monkeypatch.setattr(ffmpeg, "TOOLS_DIR", tmp_path)
    assert ffmpeg.ffmpeg_exe() == str(fake)


def test_empty_tools_falls_back_to_bundled(tmp_path, monkeypatch, fresh_cache):
    from faceblur import video
    monkeypatch.setattr(ffmpeg, "TOOLS_DIR", tmp_path)
    assert ffmpeg.ffmpeg_exe() == video.ffmpeg_exe()


def test_capabilities():
    assert ffmpeg.has_filter("libvmaf")
    assert ffmpeg.has_encoder("libx265")
    assert ffmpeg.has_encoder("hevc_nvenc")      # listed; whether it runs is nvenc_works
    assert not ffmpeg.has_encoder("no_such_encoder")
    assert not ffmpeg.has_encoder("=")


def test_nvenc_works_is_a_bool():
    assert isinstance(ffmpeg.nvenc_works(), bool)
    assert ffmpeg.nvenc_works("no_such_encoder") is False


def _vmaf(distorted: Path, reference: Path, log: Path) -> float:
    """The filter string documented in storage/ffmpeg.py."""
    quoted = "'" + log.resolve().as_posix().replace(":", r"\:") + "'"
    r = ffmpeg.run(["-v", "error", "-i", str(distorted), "-i", str(reference),
                    "-lavfi", f"libvmaf=log_fmt=json:log_path={quoted}", "-f", "null", "-"])
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-500:]
    return json.loads(log.read_text())["pooled_metrics"]["vmaf"]["mean"]


def test_libvmaf_sanity(video_silent, tmp_path):
    # vmaf_v0.6.1 tops out below 100 on identical low-motion input; this
    # 320x240 clip scores about 98.6 against itself.
    same = _vmaf(video_silent, video_silent, tmp_path / "same.json")
    assert same >= 98

    worse = tmp_path / "worse.mp4"
    r = ffmpeg.run(["-v", "error", "-i", str(video_silent), "-c:v", "libx264",
                    "-crf", "51", str(worse)])
    assert r.returncode == 0
    assert _vmaf(worse, video_silent, tmp_path / "worse.json") < same - 10
