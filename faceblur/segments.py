"""Cut a video into keyframe aligned segments, and put it back together.

Two reasons to work in segments:

- Long recordings split into ranges that run in parallel, on this machine or on
  many. Detection is per frame, so ranges need no overlap; the tracker runs
  once over the whole timeline afterwards.
- Stretches with no mask are copied from the source byte for byte, so they keep
  their original quality and cost nothing to encode. Only stretches that hold a
  mask are re-encoded. Every cut lands on a source keyframe, so a copied piece
  decodes on its own.

Everything here talks to the ffmpeg that imageio-ffmpeg bundles.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .video import VideoError, VideoInfo, _no_window, ffmpeg_exe, video_codec_args


@dataclass(frozen=True)
class Segment:
    start: int          # first frame, inclusive
    end: int            # last frame, exclusive
    copy: bool          # True: copy from the source. False: decode, redact, encode.

    @property
    def frames(self) -> int:
        return self.end - self.start


def _run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", *args], capture_output=True,
                          timeout=timeout, **_no_window())


def frame_times(src: Path) -> list[float]:
    """Presentation time of every video frame, in seconds, without decoding."""
    r = _run(["-v", "error", "-i", str(src), "-map", "0:v:0", "-c", "copy",
              "-f", "mkvtimestamp_v2", "-"])
    if r.returncode != 0:
        raise VideoError("Could not read this video. Check that the file is not open in "
                         "another program.")
    times = []
    for line in r.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            times.append(float(line) / 1000.0)
    times.sort()
    return times


def keyframes(src: Path, times: Sequence[float]) -> list[int]:
    """Indices of the frames a decoder can start at."""
    r = _run(["-v", "info", "-skip_frame", "nokey", "-i", str(src), "-map", "0:v:0",
              "-vf", "showinfo", "-f", "null", "-"])
    text = r.stderr.decode(errors="replace")
    pts = [float(m) for m in re.findall(r"pts_time:\s*([0-9.]+)", text)]
    arr = np.asarray(times)
    out = []
    for p in pts:
        i = int(np.argmin(np.abs(arr - p)))
        if abs(arr[i] - p) <= 0.005 and (not out or i > out[-1]):
            out.append(i)
    if not out or out[0] != 0:
        out.insert(0, 0)
    return out


def plan_segments(masked: Sequence[bool], keys: Sequence[int], n: int,
                  min_copy_frames: int, copy_clean: bool = True) -> list[Segment]:
    """Split [0, n) at keyframes into copy and encode segments.

    A group of pictures with no masked frame can be copied. Short clean runs are
    not worth a join, so they are encoded with their neighbours.
    """
    if n <= 0:
        return []
    bounds = sorted({k for k in keys if 0 <= k < n} | {0}) + [n]
    gops = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    kinds = []
    for a, b in gops:
        clean = copy_clean and not any(masked[a:b])
        kinds.append((a, b, clean))
    # Merge runs of the same kind.
    runs: list[list] = []
    for a, b, clean in kinds:
        if runs and runs[-1][2] == clean:
            runs[-1][1] = b
        else:
            runs.append([a, b, clean])
    # Short clean runs join the encode runs around them.
    changed = True
    while changed:
        changed = False
        for i, (a, b, clean) in enumerate(runs):
            if clean and b - a < min_copy_frames and len(runs) > 1:
                runs[i][2] = False
                changed = True
                break
        merged: list[list] = []
        for a, b, clean in runs:
            if merged and merged[-1][2] == clean:
                merged[-1][1] = b
            else:
                merged.append([a, b, clean])
        runs = merged
    return [Segment(a, b, clean) for a, b, clean in runs]


def split_long(segments: Sequence[Segment], keys: Sequence[int], max_frames: int) -> list[Segment]:
    """Cut encode segments longer than `max_frames` at keyframes, so several
    workers can encode one long masked stretch. Copies are left whole."""
    out: list[Segment] = []
    ks = sorted(set(keys))
    for seg in segments:
        if seg.copy or seg.frames <= max_frames:
            out.append(seg)
            continue
        start = seg.start
        while seg.end - start > max_frames:
            limit = start + max_frames
            cut = max((k for k in ks if start < k <= limit), default=None)
            if cut is None:
                break
            out.append(Segment(start, cut, False))
            start = cut
        out.append(Segment(start, seg.end, False))
    return out


def cut_copy(src: Path, times: Sequence[float], seg: Segment, out: Path) -> None:
    """Copy a keyframe aligned segment out of the source without re-encoding."""
    # A stream copy starts at the last keyframe at or before the seek time. Aim
    # a little past the keyframe's own timestamp, so rounding can never send
    # the seek back to the keyframe before it.
    step = (times[-1] - times[0]) / max(1, len(times) - 1)
    r = _run(["-y", "-v", "error", "-ss", f"{times[seg.start] + 0.4 * step:.6f}", "-i", str(src),
              "-map", "0:v:0", "-frames:v", str(seg.frames), "-c:v", "copy", "-an",
              "-avoid_negative_ts", "make_zero", "-video_track_timescale", "90000",
              "-f", "mp4", str(out)])
    if r.returncode != 0:
        raise VideoError("Could not write the blurred copy. See the log file for details. "
                         f"ffmpeg said: {r.stderr.decode(errors='replace')[:300]}")


def _first_frame_after_seek(src: Path, seek: float, times: Sequence[float]) -> int:
    """The index of the first frame ffmpeg outputs when told to seek to `seek`.

    Seeking by time lands within a frame of the target, either side, depending
    on how the container rounded its timestamps. The seek is deterministic, so
    asking once and reading the frame's own timestamp gives the exact index.
    """
    # -copyts keeps the source timestamps, so showinfo reports the frame's own
    # time rather than time since the seek point.
    r = _run(["-v", "info", "-ss", f"{seek:.6f}", "-copyts", "-i", str(src), "-map", "0:v:0",
              "-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"])
    text = r.stderr.decode(errors="replace")
    m = re.search(r"pts_time:\s*([0-9.]+)", text)
    if not m:
        raise VideoError("Could not read this video. Check that the file is not open in "
                         "another program.")
    pts = float(m.group(1))
    arr = np.asarray(times)
    return int(np.argmin(np.abs(arr - pts)))


def decode_range(src: Path, times: Sequence[float], start: int, end: int,
                 info: VideoInfo, hwaccel: str = "none") -> Iterator[np.ndarray]:
    """Yield frames start to end (exclusive) as BGR arrays, decoded by ffmpeg.

    Starts on exactly frame `start`: the seek is probed first and any frames
    it lands early on are skipped.
    """
    n = end - start
    if n <= 0:
        return
    seek = times[start]
    first = _first_frame_after_seek(src, seek, times) if start > 0 else 0
    if first > start:
        # Landed late. Seek a little earlier; the frame before is at most one
        # frame back.
        seek = times[max(0, start - 1)]
        first = _first_frame_after_seek(src, seek, times)
        if first > start:
            raise VideoError(f"Could not seek to frame {start} of this video.")
    skip = start - first
    accel = ["-hwaccel", hwaccel] if hwaccel and hwaccel != "none" else []
    cmd = [ffmpeg_exe(), "-hide_banner", "-v", "error", *accel, "-ss", f"{seek:.6f}",
           "-i", str(src), "-map", "0:v:0", "-frames:v", str(n + skip),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_no_window())
    try:
        got = 0
        while got < n + skip:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            if got >= skip:
                yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
            got += 1
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait()


class SegmentEncoder:
    """Encodes one segment's frames to a video only file, same rate as the source."""

    def __init__(self, out: Path, info: VideoInfo, crf: int, preset: str,
                 encoder: str, nvenc_cq: int):
        self.out = Path(out)
        cmd = [ffmpeg_exe(), "-hide_banner", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
               "-r", f"{info.fps:.6f}", "-i", "-",
               *video_codec_args(encoder, crf, preset, nvenc_cq),
               "-pix_fmt", "yuv420p", "-video_track_timescale", "90000",
               "-f", "mp4", str(self.out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, **_no_window())

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise VideoError("Could not write the blurred copy. See the log file for "
                             "details.") from exc

    def close(self) -> str:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        err = self.proc.stderr.read() or b""
        self.proc.wait()
        return err.decode(errors="replace").strip() if self.proc.returncode else ""

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        self.out.unlink(missing_ok=True)


def concat(parts: Sequence[Path], src: Path, dst: Path, list_file: Path) -> str:
    """Join segments in order and add the source audio. Returns ffmpeg's error text."""
    list_file.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts),
                         encoding="utf-8")
    r = _run(["-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
              "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
              "-movflags", "+faststart", "-f", "mp4", str(dst)])
    return r.stderr.decode(errors="replace").strip() if r.returncode else ""


def verify(dst: Path, expected_frames: int, expected_seconds: float | None = None) -> str:
    """Check a finished file: it demuxes clean, holds every frame, its
    timestamps only go forward, and it runs as long as the source. Returns an
    empty string when it passes."""
    r = _run(["-v", "error", "-i", str(dst), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"])
    err = r.stderr.decode(errors="replace").strip()
    if r.returncode != 0 or err:
        return f"demux: {err[:200]}"
    try:
        times = frame_times(dst)
    except VideoError as exc:
        return str(exc)
    if len(times) != expected_frames:
        return f"frame count {len(times)}, expected {expected_frames}"
    if any(b <= a for a, b in zip(times, times[1:])):
        return "timestamps do not increase"
    if expected_seconds is not None and len(times) > 1:
        span = times[-1] - times[0]
        tolerance = 2.5 * span / max(1, len(times) - 1)      # about two frames
        if abs(span - expected_seconds) > tolerance:
            return f"length {span:.3f} s, expected {expected_seconds:.3f} s"
    return ""
