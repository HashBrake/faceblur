"""Shared fixtures. Synthetic videos are built once per test session with ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from faceblur.video import ffmpeg_exe

DATA = Path(__file__).resolve().parent / "data"


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", *args],
                          capture_output=True, text=True, errors="replace")


def stream_lines(path: Path) -> list[str]:
    """Stream description lines for a file, from ffmpeg -i.

    ffprobe is not installed and imageio-ffmpeg ships ffmpeg only, so the tests
    read stream facts from ffmpeg's own report.
    """
    result = run_ffmpeg(["-i", str(path)])
    return [line.strip() for line in result.stderr.splitlines() if "Stream #" in line]


def audio_streams(path: Path) -> list[str]:
    return [line for line in stream_lines(path) if "Audio:" in line]


def audio_codec(path: Path) -> str:
    lines = audio_streams(path)
    if not lines:
        return ""
    after = lines[0].split("Audio:", 1)[1].strip()
    return after.split()[0].split(",")[0]


def _make(path: Path, seconds: float, fps: int, size: str, with_audio: bool) -> Path:
    if path.exists():
        return path
    args = ["-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    result = run_ffmpeg(args)
    if result.returncode != 0:
        raise RuntimeError(f"could not build the test video: {result.stderr[-500:]}")
    return path


@pytest.fixture(scope="session")
def video_with_audio(tmp_path_factory) -> Path:
    """Three seconds, 10 fps, 320x240, one aac audio track."""
    out = tmp_path_factory.mktemp("clips") / "with_audio.mp4"
    return _make(out, 3, 10, "320x240", True)


@pytest.fixture(scope="session")
def video_silent(tmp_path_factory) -> Path:
    """Two seconds, 10 fps, 320x240, no audio track. Like the Ego camera files."""
    out = tmp_path_factory.mktemp("clips") / "silent.mp4"
    return _make(out, 2, 10, "320x240", False)


@pytest.fixture(scope="session")
def video_long(tmp_path_factory) -> Path:
    """Long enough that a cancel lands in the middle of the detect pass."""
    out = tmp_path_factory.mktemp("clips") / "long.mp4"
    return _make(out, 12, 10, "320x240", False)


@pytest.fixture(scope="session")
def messi() -> Path:
    return DATA / "messi5.jpg"


@pytest.fixture(scope="session")
def lena() -> Path:
    return DATA / "lena.jpg"
