"""Facts about a stored video, read from its packets with PyAV.

A converted file is checked against its source by frame count, size, rate and
timing. Reading packets decodes nothing, so this is cheap enough to run on
every file of a multi-hour recording, and it gives the same frame times as
`faceblur.segments.frame_times` without parsing ffmpeg's text output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av

from faceblur.video import VideoError


@dataclass(frozen=True)
class VideoFacts:
    codec: str
    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    size_bytes: int


def _open(path: Path):
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise VideoError(f"Could not read this video: {path}") from exc
    if not container.streams.video:
        container.close()
        raise VideoError(f"This file has no video stream: {path}")
    return container


def _video_packets(container):
    """The video stream's real packets; the empty flush packet at the end is skipped."""
    stream = container.streams.video[0]
    for packet in container.demux(stream):
        if packet.size and packet.pts is not None:
            yield packet


def probe(path: Path) -> VideoFacts:
    path = Path(path)
    with _open(path) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in _video_packets(container))
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = (container.duration or 0) / av.time_base
        rate = stream.average_rate
        return VideoFacts(
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=float(rate) if rate else 0.0,
            frames=frames,
            duration_s=duration,
            size_bytes=path.stat().st_size,
        )


def frame_pts_seconds(path: Path) -> list[float]:
    """Presentation time of every video frame, in seconds, sorted, decoding nothing."""
    with _open(Path(path)) as container:
        time_base = container.streams.video[0].time_base
        return sorted(float(p.pts * time_base) for p in _video_packets(container))
