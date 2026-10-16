"""Re-encode a delivered video to HEVC (or AV1) for hot storage.

The hot store keeps every recording, so each file is re-encoded to a codec
that is about half the size of the H.264 the pipeline writes. NVENC is used
when this GPU can run it, because it is many times faster than the software
encoders; libx265 and libaom-av1 are the fallback where it cannot.

Unlike the pipeline's own encode, B-frames are on: a stored file is never
joined to a stream-copied piece, so frame reordering costs nothing here and
saves size. A keyframe every GOP frames keeps seeking cheap.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from faceblur.video import VideoError, part_path
from storage.ffmpeg import nvenc_works, run

CODECS = ("hevc", "av1")
ENCODERS = ("auto", "nvenc", "software")
GOP = 60


# With `-b:v 0` and no `-maxrate`, the NVENC driver applies its own maximum
# bitrate (about 10 Mb/s here), so every cq below ~24 gave the same size. An
# explicit cap well above the H.264 sources' 25 Mb/s peak lets constant quality
# decide the size across the useful range.
NVENC_RATE_CAP = ["-maxrate", "60M", "-bufsize", "120M"]


def codec_args(codec: str, encoder: str, quality: int, gop: int = GOP) -> list[str]:
    """The `-c:v ...` arguments for a resolved encoder (`nvenc` or `software`)."""
    if codec == "hevc" and encoder == "nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", *NVENC_RATE_CAP, "-g", str(gop), "-bf", "3",
                "-b_ref_mode", "middle", "-spatial-aq", "1", "-temporal-aq", "1",
                "-tag:v", "hvc1"]
    if codec == "hevc" and encoder == "software":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", str(quality),
                "-g", str(gop), "-x265-params", f"keyint={gop}:min-keyint=1:bframes=4",
                "-tag:v", "hvc1", "-threads", "4"]
    if codec == "av1" and encoder == "nvenc":
        return ["-c:v", "av1_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0", *NVENC_RATE_CAP, "-g", str(gop), "-bf", "3"]
    if codec == "av1" and encoder == "software":
        return ["-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1", "-crf", str(quality),
                "-b:v", "0", "-g", str(gop), "-threads", "4"]
    raise ValueError(f"Unknown codec or encoder: {codec}, {encoder}")


def resolve_encoder(codec: str, encoder: str) -> str:
    """`auto` becomes `nvenc` when this GPU can run it, else `software`."""
    if encoder == "auto":
        return "nvenc" if nvenc_works(f"{codec}_nvenc") else "software"
    return encoder


@dataclass(frozen=True)
class EncodeResult:
    dst: Path
    codec: str
    encoder: str
    quality: int
    seconds: float
    size_bytes: int


def encode_file(src: Path, dst: Path, codec: str = "hevc", encoder: str = "auto",
                quality: int = 24, gop: int = GOP) -> EncodeResult:
    """Encode `src` to `dst`. The file takes its name only when ffmpeg succeeded."""
    src, dst = Path(src), Path(dst)
    resolved = resolve_encoder(codec, encoder)
    part = part_path(dst)
    args = ["-y", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a?",
            *codec_args(codec, resolved, quality, gop),
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            # Keep the source timestamps. Left alone, ffmpeg takes the encoder
            # time base from r_frame_rate (30/1 for the camera's 30.0085 fps),
            # so the copy drifts from the original and the data loaders would
            # see frame i at another time than in the original. 90000 matches
            # the camera files and faceblur/segments.py.
            "-fps_mode", "passthrough", "-enc_time_base", "-1",
            "-video_track_timescale", "90000",
            "-f", "mp4", str(part)]
    start = time.monotonic()
    try:
        r = run(args, timeout=6 * 3600)
    except (OSError, subprocess.SubprocessError) as exc:
        part.unlink(missing_ok=True)
        raise VideoError(f"Could not encode {src}: {exc}") from exc
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        tail = r.stderr.decode(errors="replace").strip()[-500:]
        raise VideoError(f"Could not encode {src}: {tail}")
    os.replace(part, dst)
    return EncodeResult(dst=dst, codec=codec, encoder=resolved, quality=quality,
                        seconds=time.monotonic() - start, size_bytes=dst.stat().st_size)
