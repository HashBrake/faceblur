"""Decode with OpenCV, encode with the ffmpeg that imageio-ffmpeg bundles.

The user does not install ffmpeg. imageio-ffmpeg carries a static binary in its
wheel and this module asks it for the path.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoError(RuntimeError):
    """A video could not be read or written."""


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path to ffmpeg.

    A PyInstaller build carries the binary that imageio-ffmpeg ships. Look for
    the bundled copy first, because imageio-ffmpeg searches beside its own source
    file and a frozen app moves that.
    """
    override = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if override and Path(override).is_file():
        return override

    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for folder in (root / "imageio_ffmpeg" / "binaries", root):
            if folder.is_dir():
                for candidate in sorted(folder.glob("ffmpeg*")):
                    if candidate.is_file():
                        return str(candidate)

    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# On Windows, keep the console window of a child process hidden. The packaged UI
# is a windowed build and a visible console flash on every file looks like a bug.
def _no_window() -> dict:
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"startupinfo": startupinfo,
            "creationflags": subprocess.CREATE_NO_WINDOW}


def part_path(dst: Path) -> Path:
    """The file ffmpeg writes while an encode is in progress."""
    return Path(str(dst) + ".part")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def probe(src: Path) -> VideoInfo:
    """Read size, rate and length without decoding the whole file."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise VideoError(
            "Could not read this video. Check that the file is not open in "
            "another program."
        )
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise VideoError(
            "Could not read this video. Check that the file is not open in "
            "another program."
        )
    return VideoInfo(width, height, float(fps), count)


def read_frames(src: Path) -> Iterator[np.ndarray]:
    """Yield every frame as a BGR array."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise VideoError(
            "Could not read this video. Check that the file is not open in "
            "another program."
        )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


class Encoder:
    """Pipes raw BGR frames into ffmpeg and copies the source audio through.

    The source file is a second input. `-map 1:a?` takes its audio when there is
    audio and writes video only when there is none. Audio is copied, never
    re-encoded, so FaceBlur does not change it.
    """

    def __init__(self, src: Path, dst: Path, info: VideoInfo, crf: int, preset: str):
        self.dst = Path(dst)
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        # ffmpeg writes a .part file. It becomes the real file only when the
        # encode finished cleanly, so nothing that looks finished ever is not.
        self.part = part_path(self.dst)
        self.part.unlink(missing_ok=True)
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}", "-i", "-",
            "-i", str(src),
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-movflags", "+faststart", "-f", "mp4", str(self.part),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, **_no_window()
        )
        self._stderr = b""

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise VideoError(
                "Could not write the blurred copy. See the log file for details."
            ) from exc

    def close(self) -> str:
        """Finish the file. Returns ffmpeg's error text, empty when it succeeded."""
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self._stderr = self.proc.stderr.read() or b""
        self.proc.wait()
        if self.proc.returncode != 0:
            self.part.unlink(missing_ok=True)
            return self._stderr.decode(errors="replace").strip()
        os.replace(self.part, self.dst)
        return ""

    def abort(self) -> None:
        """Stop ffmpeg and delete the partial file. Never leaves half a video behind."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            self.proc.stderr.close()
        except OSError:
            pass
        self.part.unlink(missing_ok=True)
        self.dst.unlink(missing_ok=True)

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
