"""Give a blurred copy its source's frame timestamps, packet for packet.

Every copy holds every frame of its source, frame n for frame n, but the
encoder that wrote it numbered the frames at a constant rate. The camera does
not record at one: its frames are a few milliseconds early or late, and a
dropped frame leaves a gap of two. A copy re-timed at a constant rate puts
each frame up to a frame away from where the sensor files (`*_pts.csv`,
`*_imu.csv`) and the other camera say it was.

`retime` fixes that without touching a picture: it remuxes the copy's video
packets, unchanged, and stamps packet n with the presentation time of the
source's n-th frame, in the source's own time base. Nothing is re-encoded and
nothing is rounded. `storage/encode.py` does the same for the hot-storage
HEVC by letting ffmpeg pass the decoded timestamps through (D14); the blur
write cannot, because its frames reach the encoder through a pipe of raw
pictures that carries no time at all, so the timestamps go on afterwards.

The pipeline calls it on every joined copy (`batch._write_and_join`), which
covers encoded stretches, stretches copied from the source or from an
earlier copy, and the patch pass. `cloud/retime.py` calls it on copies that
were delivered before this existed.

Needs PyAV, which `storage/` already uses.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

from .video import VideoError, _no_window, ffmpeg_exe, part_path


class TimingError(VideoError):
    """A copy could not be given its source's timestamps."""


def _av():
    try:
        import av
    except ImportError as exc:      # pragma: no cover - a broken install
        raise TimingError("PyAV is not installed; it is needed to keep the source's frame "
                          "timestamps (pip install av==19.0.1).") from exc
    return av


@dataclass(frozen=True)
class SourceTiming:
    """The presentation time of every frame of a video, in its own time base."""
    time_base: Fraction
    pts: tuple                      # sorted, one per frame
    last_duration: int              # how long the last frame shows, in time base ticks

    def __len__(self) -> int:
        return len(self.pts)

    def seconds(self) -> list[float]:
        return [float(p * self.time_base) for p in self.pts]


def source_timing(path: Path) -> SourceTiming:
    """Every frame's container pts, in presentation order, decoding nothing."""
    av = _av()
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise TimingError(f"{Path(path).name} has no video stream.")
            s = c.streams.video[0]
            tb = s.time_base
            rows = []
            for p in c.demux(s):
                if p.size == 0:
                    continue                    # the demuxer's end of stream marker
                if p.pts is None:
                    raise TimingError(f"{Path(path).name}: a frame without a timestamp.")
                rows.append((p.pts, p.duration or 0))
    except av.error.FFmpegError as exc:
        raise TimingError(f"Could not read {Path(path).name}: {exc}") from exc
    rows.sort()
    pts = tuple(r[0] for r in rows)
    if len(set(pts)) != len(pts):
        raise TimingError(f"{Path(path).name}: two frames share one timestamp.")
    last = rows[-1][1] if rows else 0
    if last <= 0 and len(pts) > 1:
        steps = sorted(b - a for a, b in zip(pts, pts[1:]))
        last = steps[len(steps) // 2]
    return SourceTiming(Fraction(tb.numerator, tb.denominator), pts, max(1, last))


@dataclass
class CopyLayout:
    """What `retime` needs to know about the copy before it writes."""
    frames: int
    reorder: int                    # frames the decoder holds back; 0 without B-frames
    ranks: list = field(default_factory=list)   # presentation index of each packet, decode order


def copy_layout(path: Path) -> CopyLayout:
    """Count the copy's video packets and work out their presentation order.

    With no B-frames the order is the decode order and `reorder` is 0; the
    check is per file, from the packets' own pts, not assumed.
    """
    av = _av()
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                raise TimingError(f"{Path(path).name} has no video stream.")
            s = c.streams.video[0]
            pts = []
            for p in c.demux(s):
                if p.size == 0:
                    continue
                if p.pts is None:
                    raise TimingError(f"{Path(path).name}: a frame without a timestamp.")
                pts.append(p.pts)
    except av.error.FFmpegError as exc:
        raise TimingError(f"Could not read {Path(path).name}: {exc}") from exc
    order = sorted(range(len(pts)), key=lambda i: pts[i])
    if len(set(pts)) != len(pts):
        raise TimingError(f"{Path(path).name}: two frames share one timestamp.")
    ranks = [0] * len(pts)
    for rank, i in enumerate(order):
        ranks[i] = rank
    reorder = max([i - r for i, r in enumerate(ranks)] + [0])
    return CopyLayout(len(pts), reorder, ranks)


@dataclass
class RetimeResult:
    frames: int
    time_base: str
    reorder: int
    max_error_before_ms: float      # copy vs source, after removing one constant offset
    max_error_after_ms: float
    packets_sha256: str             # of the video payloads, which retime leaves unchanged
    audio_packets: int = 0


def _max_error_ms(copy_seconds: list[float], src_seconds: list[float]) -> float:
    """Largest gap between the two clocks after removing one constant offset
    (the median difference), the way `pipeline_demo/vm/ts_check.py` measures."""
    n = min(len(copy_seconds), len(src_seconds))
    if n == 0:
        return 0.0
    d = sorted(copy_seconds[i] - src_seconds[i] for i in range(n))
    off = d[len(d) // 2]
    return max(abs(copy_seconds[i] - src_seconds[i] - off) for i in range(n)) * 1000.0


def retime(copy: Path, source: Path, dst: Path, audio_from: Optional[Path] = None,
           frames: Optional[int] = None, allow_reorder: bool = True,
           timing: Optional[SourceTiming] = None) -> RetimeResult:
    """Write `dst`: the copy's video packets, unchanged, on the source's clock.

    Packet n in presentation order gets the pts of the source's n-th frame,
    in the source's time base; decode times follow from the copy's own
    reordering, so a copy without B-frames gets dts equal to pts. The
    source's audio goes in beside it when `audio_from` names a file, packets
    copied with their own timestamps, so sound and picture keep the source's
    alignment. The file is written with its index first (faststart) and
    takes `dst` only when the check below passed.

    The copy must hold exactly as many frames as the source. `frames` relaxes
    that for a source whose damaged last frames could not be decoded: the
    copy then holds `frames` and the source at least that many, and the
    first `frames` of the source's timestamps are used.

    `allow_reorder` False refuses a copy whose decode order is not its
    presentation order.
    """
    av = _av()
    copy, source, dst = Path(copy), Path(source), Path(dst)
    ref = timing or source_timing(source)
    layout = copy_layout(copy)
    n = layout.frames
    if layout.reorder and not allow_reorder:
        raise TimingError(f"{copy.name}: decode order is not presentation order "
                          f"(reorder delay {layout.reorder}); refusing to retime it.")
    if frames is not None:
        if n != frames or len(ref) < frames:
            raise TimingError(f"{copy.name}: {n} frames, source {len(ref)}, expected {frames}.")
    elif n != len(ref):
        raise TimingError(f"{copy.name}: {n} frames, the source has {len(ref)}; "
                          "frame counts differ, refusing to retime.")
    pts = list(ref.pts[:n])
    durations = [b - a for a, b in zip(pts, pts[1:])] + [ref.last_duration]
    if n > len(pts) or n == 0:
        raise TimingError(f"{copy.name}: no frames.")
    first_step = durations[0] if n > 1 else ref.last_duration
    # Decode times for a copy that reorders: the frame decoded at position i
    # shows no earlier than the (i - delay)th, so that frame's time is a
    # valid dts, and the first `delay` packets count back from the start.
    d = layout.reorder

    def dts_of(i: int) -> int:
        return pts[i - d] if i >= d else pts[0] - (d - i) * first_step

    # Before: the copy's own clock against the source's.
    copy_timing = source_timing(copy)
    before = _max_error_ms(copy_timing.seconds()[:n], [float(p * ref.time_base) for p in pts])

    part = part_path(dst)
    part.unlink(missing_ok=True)
    tb = ref.time_base
    digest = hashlib.sha256()
    audio_packets = 0
    try:
        with av.open(str(copy)) as cin, \
                av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            vin = cin.streams.video[0]
            vout = out.add_stream_from_template(vin)
            vout.time_base = tb
            ain = aout = None
            aud = None
            if audio_from is not None:
                aud = av.open(str(audio_from))
                if aud.streams.audio:
                    ain = aud.streams.audio[0]
                    aout = out.add_stream_from_template(ain)
                    aout.time_base = ain.time_base

            def video() -> Iterator:
                i = 0
                for p in cin.demux(vin):
                    if p.size == 0:
                        continue
                    if i >= n:
                        raise TimingError(f"{copy.name}: more packets than counted.")
                    digest.update(bytes(p))
                    r = layout.ranks[i]
                    p.pts, p.dts, p.duration = pts[r], dts_of(i), durations[r]
                    p.time_base = tb
                    p.stream = vout
                    yield (float(p.dts * tb), 0, i, p)
                    i += 1

            def audio() -> Iterator:
                k = 0
                for p in aud.demux(ain):
                    if p.size == 0 or p.dts is None:
                        continue
                    p.stream = aout
                    yield (float(p.dts * ain.time_base), 1, k, p)
                    k += 1

            streams = [video()] + ([audio()] if aout is not None else [])
            for _, kind, _, p in heapq.merge(*streams, key=lambda t: (t[0], t[1], t[2])):
                out.mux(p)
                audio_packets += kind
            if aud is not None:
                aud.close()
        result = RetimeResult(frames=n, time_base=f"{tb.numerator}/{tb.denominator}",
                              reorder=d, max_error_before_ms=round(before, 3),
                              max_error_after_ms=0.0, packets_sha256=digest.hexdigest(),
                              audio_packets=audio_packets)
        problem = check_retimed(part, pts, tb, result.packets_sha256)
        if problem:
            raise TimingError(f"{copy.name}: the retimed copy did not check: {problem}")
        got = source_timing(part)
        result.max_error_after_ms = round(
            _max_error_ms(got.seconds(), [float(p * tb) for p in pts]), 3)
        os.replace(part, dst)
        return result
    except av.error.FFmpegError as exc:
        part.unlink(missing_ok=True)
        raise TimingError(f"Could not retime {copy.name}: {exc}") from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def check_retimed(path: Path, pts: list, time_base: Fraction,
                  packets_sha256: Optional[str] = None) -> str:
    """Packet level check of a retimed file: as many frames as `pts`, each at
    exactly its time in `time_base`, and the video payloads the same bytes as
    the copy's (`packets_sha256`). Empty when it passes."""
    av = _av()
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            if Fraction(s.time_base.numerator, s.time_base.denominator) != time_base:
                return f"time base {s.time_base}, expected {time_base}"
            digest = hashlib.sha256()
            got = []
            for p in c.demux(s):
                if p.size == 0:
                    continue
                digest.update(bytes(p))
                got.append(p.pts)
    except av.error.FFmpegError as exc:
        return f"demux: {exc}"
    if len(got) != len(pts):
        return f"frame count {len(got)}, expected {len(pts)}"
    if sorted(got) != list(pts):
        wrong = sum(1 for a, b in zip(sorted(got), pts) if a != b)
        return f"{wrong} frames off the source's timestamps"
    if packets_sha256 is not None and digest.hexdigest() != packets_sha256:
        return "video packets changed"
    return ""


def decode_check(path: Path, frames: int, timeout: int = 6 * 3600) -> str:
    """Decode every frame with ffmpeg: no error, and `frames` frames. Empty
    when it passes."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostats", "-v", "error",
                        # Passthrough timestamps: on the encoder's default
                        # 1/fps time base two close frames of a variable rate
                        # video round to one tick and the null muxer complains
                        # (as in storage/checks.py).
                        "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
                        "-enc_time_base", "-1", "-f", "null",
                        "-progress", "pipe:1", "-"],
                       capture_output=True, timeout=timeout, **_no_window())
    err = r.stderr.decode(errors="replace").strip()
    if r.returncode != 0 or err:
        return f"decode: {err[:300] or r.returncode}"
    got = None
    for line in r.stdout.decode(errors="replace").splitlines():
        if line.startswith("frame="):
            try:
                got = int(line.split("=", 1)[1])
            except ValueError:
                pass
    if got != frames:
        return f"decoded {got} frames, expected {frames}"
    return ""


def same_timing(copy: Path, timing: SourceTiming) -> bool:
    """Does the copy already carry exactly these timestamps?"""
    got = source_timing(copy)
    return got.time_base == timing.time_base and got.pts == timing.pts
