"""The DF2 command line: `python -m faceblur.h5 pack a.mp4 [b.mp4 ...] --out file.h5`.

`pack` builds a DF2 file from camera mp4s, because the partner has not sent a
real one yet and the module needs something to read. Besides the cameras it
writes small deterministic datasets in every other group of the tree, so the
pass-through copy has something to carry and the tests something to compare.
Later tasks add the `run`, `verify` and `selfcheck` subcommands here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from . import df2, h5io

FIXTURE_VERSION = "fixture-2026-11"


def _dummy_passthrough(f: h5py.File, frames: int) -> None:
    """One small dataset in each pass-through group, the same every time."""
    for i, path in enumerate(df2.PASSTHROUGH_GROUPS):
        group = f.require_group(path)
        if path.startswith(df2.META_GROUP):
            data = group.create_dataset("text", data=np.bytes_(f"fixture text for {path}"))
        elif "/imu/" in path or "/emg/" in path:
            # Wearable sensors sample faster than the cameras.
            values = np.arange(frames * 4 * 6, dtype=np.float32).reshape(frames * 4, 6) + i
            data = group.create_dataset("samples", data=values)
        else:
            values = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) + i
            data = group.create_dataset("values", data=values)
        data.attrs["source"] = "faceblur.h5 pack"


def pack(mp4s: list[Path], cameras: list[str], audio: Path | None, out: Path) -> list[str]:
    """Write a DF2 file. Returns the groups written, for the report."""
    written = []
    with h5py.File(out, "w") as f:
        f.attrs["df2_version"] = FIXTURE_VERSION
        first_timestamps = None
        for mp4, name in zip(mp4s, cameras):
            path = df2.camera_path(name)
            h5io.write_video_group(f.require_group(path), mp4)
            if first_timestamps is None:
                first_timestamps = f[path][df2.TIMESTAMPS_NS][:]
            written.append(path)
        if audio is not None:
            h5io.write_audio_group(f, audio)
            written.append(df2.AUDIO_GROUP)
        timeline = f.require_group(df2.TIMELINE_GROUP)
        timeline.create_dataset(df2.TIMELINE_TIMESTAMPS_NS, data=first_timestamps)
        written.append(df2.TIMELINE_GROUP)
        _dummy_passthrough(f, len(first_timestamps))
        written += list(df2.PASSTHROUGH_GROUPS)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m faceblur.h5", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="build a DF2 file from camera mp4s")
    p.add_argument("mp4", nargs="+", type=Path)
    p.add_argument("--cameras", help="comma-separated camera names, one per mp4")
    p.add_argument("--audio", type=Path, help="a sound file for the audio group")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    cameras = args.cameras.split(",") if args.cameras else list(df2.PACK_ORDER)
    if len(cameras) < len(args.mp4):
        ap.error(f"{len(args.mp4)} mp4s but only {len(cameras)} camera names")
    for path in pack(args.mp4, cameras, args.audio, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
