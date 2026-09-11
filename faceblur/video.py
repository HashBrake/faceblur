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


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True when the bundled ffmpeg can encode with the NVIDIA encoder here."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "color=size=256x256:rate=1",
             "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30, **_no_window())
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def video_codec_args(encoder: str, crf: int, preset: str, nvenc_cq: int) -> list[str]:
    """The ffmpeg video codec arguments for the chosen encoder."""
    # No B-frames. A segment that is joined to a stream copied piece of the
    # source must carry the same frame reordering delay as that piece, or the
    # concat demuxer places the two a few frames apart. Copies are only made
    # from sources without reordering (see `reorder_delay`), so the encoded
    # pieces get none either. This costs some file size at the same quality.
    use_nvenc = encoder == "nvenc" or (encoder == "auto" and nvenc_available())
    if use_nvenc:
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq",
                "-rc", "vbr", "-cq", str(nvenc_cq), "-b:v", "0", "-profile:v", "high",
                "-bf", "0"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-bf", "0"]


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
    # What the source carried besides pictures. Package M1: an auditor asking
    # "was there sound, and what did the container know about this file" can
    # read the answer without the file.
    audio: bool = False
    # The **names** of the container's metadata tags, never their values. A
    # tag can hold a location, a device serial or an account name, and this
    # project does not copy those anywhere, including into its own record.
    tags: tuple = ()


def source_metadata(src: Path) -> tuple[bool, tuple[str, ...]]:
    """(has audio, the names of the container's tags).

    One ffmpeg call that decodes nothing. `-f ffmetadata` writes the tags as
    `name=value` lines and everything to the left of the first `=` is kept,
    which is the whole point: the value is what would be sensitive.

    A file ffmpeg cannot read is not an error here. `probe` has already said
    so, or is about to, and a metadata line is not worth a second way to fail.
    """
    import subprocess

    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(src),
             "-map_metadata", "0", "-f", "ffmetadata", "-"],
            capture_output=True, timeout=120, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return False, ()
    names = []
    for line in r.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith((";", "#", "[")) or "=" not in line:
            continue
        name = line.split("=", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return has_audio(src), tuple(names)


def has_audio(src: Path) -> bool:
    """Is there an audio stream? Asked of the demuxer, decoding nothing."""
    import subprocess

    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-v", "error", "-i", str(src),
             "-map", "0:a:0", "-c", "copy", "-frames:a", "0", "-f", "null", "-"],
            capture_output=True, timeout=120, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


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
    audio, tags = source_metadata(src)
    return VideoInfo(width, height, float(fps), count, audio, tags)


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

    def __init__(self, src: Path, dst: Path, info: VideoInfo, crf: int, preset: str,
                 encoder: str = "x264", nvenc_cq: int = 16):
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
            *video_codec_args(encoder, crf, preset, nvenc_cq),
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
