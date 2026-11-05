"""The DF2 command line: `python -m faceblur.h5 pack ...` and `python -m faceblur.h5 run ...`.

`pack` builds a DF2 file from camera mp4s, because the partner has not sent a
real one yet and the module needs something to read. Besides the cameras it
writes small deterministic datasets in every other group of the tree, so the
pass-through copy has something to carry and the tests something to compare.

`run` is the privacy module itself. Each camera's packets are remuxed to an
mp4, put through the same file pipeline the desktop app uses (batch.run_video),
and the blurred mp4 is stored back as packets. The input is never modified.
The default output is a release file with no audio and no raw video; with
--spec-layout it is instead a copy of the input with the blurred cameras added
beside the raw ones, the single-file form the partner's spec describes.
Either way it is written under a temporary name and renamed only when every
camera finished, so a failed run leaves no episode.h5 that looks complete.
Later tasks add the `verify` and `selfcheck` subcommands here.
"""
from __future__ import annotations

import argparse
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from . import batch, df2, h5io
from .classes import UnknownKind
from .classes import parse as parse_mask
from .pipeline import STATUS_DONE, STATUS_FAILED, AuditRecord
from .settings import Settings

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


class RunFailed(Exception):
    """A run that cannot finish. `code` is the exit status the CLI returns."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


# What write_video_group makes from the blurred mp4. Everything else in an
# input camera group (calibration, whatever the partner adds) is copied over.
_REWRITTEN = (df2.VIDEO, df2.PTS, df2.DTS, df2.KEYFRAME, df2.TIMESTAMPS_NS)
_REWRITTEN_ATTRS = (df2.ATTR_CODEC, df2.ATTR_WIDTH, df2.ATTR_HEIGHT, df2.ATTR_FPS,
                    df2.ATTR_TIME_BASE_NUM, df2.ATTR_TIME_BASE_DEN, df2.ATTR_PIX_FMT,
                    df2.ATTR_EXTRADATA, df2.ATTR_FRAME_COUNT)


def _store_camera(src: h5py.Group, dst: h5py.Group, blurred: Path) -> None:
    """The blurred mp4 as packets in `dst`, with the rest of `src` beside them."""
    h5io.write_video_group(dst, blurred)
    for name in src:
        if name in _REWRITTEN:
            continue
        # write_video_group puts placeholder calibration in; the input's wins.
        if name in dst:
            del dst[name]
        src.copy(src[name], dst, name=name)
    for key, value in src.attrs.items():
        if key not in _REWRITTEN_ATTRS:
            dst.attrs[key] = value


def _blur_camera(f: h5py.File, camera: str, work: Path, settings: Settings,
                 submit, workers: int) -> dict:
    """Run one camera through the file pipeline. Returns its result dict."""
    raw = work / f"{camera}.mp4"
    blurred = work / f"{camera}_blurred.mp4"
    group = f[df2.camera_path(camera)]
    h5io.read_video_group(group, raw)
    per_frame: list = []
    started = time.time()
    try:
        record = batch.run_video(raw, blurred, settings, submit, workers=workers,
                                 write_sidecar=False, per_frame_out=per_frame)
    except batch.WorkerLost as exc:
        record = AuditRecord(source=raw.name, output=blurred.name, status=STATUS_FAILED,
                             error=str(exc))
    if record.status != STATUS_DONE:
        raise RunFailed(f"{camera}: {record.status}: {record.error}", 1)
    return {"camera": camera, "record": record, "per_frame": per_frame,
            "frames": len(per_frame), "seconds": round(time.time() - started, 2),
            "timestamps_ns": group[df2.TIMESTAMPS_NS][:], "mp4": blurred}


def _write_output(input_path: Path, part: Path, results: list[dict], spec_layout: bool) -> None:
    """Write `part`: the release file, or the input copy with a sibling group."""
    if spec_layout:
        # The raw cameras and the audio stay, by the spec's design.
        shutil.copyfile(input_path, part)
    with h5py.File(input_path, "r") as src, h5py.File(part, "r+" if spec_layout else "w") as out:
        if spec_layout:
            parent = df2.RELEASE_IMAGE_GROUP
        else:
            parent = df2.IMAGE_GROUP
            h5io.copy_passthrough(src, out)
        for result in results:
            camera = result["camera"]
            _store_camera(src[df2.camera_path(camera)],
                          out.require_group(f"{parent}/{camera}"), result["mp4"])
        out.attrs["faceblur_release"] = True
        out.attrs["faceblur_version"] = AuditRecord.faceblur_version
        out.attrs["faceblur_layout"] = "spec" if spec_layout else "two-file"


def run(input_path: Path, out_dir: Path, settings: Settings, workers: int = 2,
        spec_layout: bool = False) -> list[dict]:
    """De-identify every camera of `input_path` into `out_dir/episode.h5`.

    Returns one dict per camera: camera, record, per_frame, frames, seconds,
    timestamps_ns. Raises RunFailed when there is no camera (code 2) or a
    camera did not finish (code 1); then no episode.h5 is written.
    """
    input_path, out_dir = Path(input_path), Path(out_dir)
    with h5py.File(input_path, "r") as f:
        cameras = h5io.camera_groups(f)
    if not cameras:
        raise RunFailed(f"{input_path} has no camera group under {df2.IMAGE_GROUP}.", 2)
    work = out_dir / ".work"
    part = out_dir / "episode.h5.part"
    work.mkdir(parents=True, exist_ok=True)
    pool = None
    try:
        if workers == 1:
            submit = batch.serial_submit
        else:
            # Built the way cli.run_parallel builds it, one pool for the whole run.
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(workers, initializer=batch.init_pool_worker,
                                initargs=(workers,))
            submit = batch.PoolSubmit(pool)
        with h5py.File(input_path, "r") as f:
            results = [_blur_camera(f, camera, work, settings, submit, workers)
                       for camera in cameras]
        _write_output(input_path, part, results, spec_layout)
        part.replace(out_dir / "episode.h5")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        part.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    for result in results:
        del result["mp4"]  # gone with the work folder
    return results


def _describe(result: dict) -> str:
    """One line per camera for the CLI."""
    frames = result["frames"]
    ms = 1000 * result["seconds"] / frames if frames else 0.0
    return (f"{result['camera']}: {frames} frames, {result['seconds']:.1f} s, "
            f"{ms:.0f} ms/frame, masked {100 * result['record'].masked_mean:.2f}% of the picture")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m faceblur.h5", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="build a DF2 file from camera mp4s")
    p.add_argument("mp4", nargs="+", type=Path)
    p.add_argument("--cameras", help="comma-separated camera names, one per mp4")
    p.add_argument("--audio", type=Path, help="a sound file for the audio group")
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="write a de-identified episode.h5 from a DF2 file")
    r.add_argument("input", type=Path)
    r.add_argument("--out", type=Path, required=True, help="folder for episode.h5")
    r.add_argument("--spec-layout", action="store_true",
                   help="copy the input and add the blurred cameras beside the raw ones")
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--mask", default="face,text,plate,screen")
    r.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    r.add_argument("--check-output", action="store_true")
    args = ap.parse_args(argv)

    if args.command == "run":
        try:
            mask = parse_mask(args.mask)
        except UnknownKind as exc:
            ap.error(str(exc))
        settings = Settings(mask=mask, device=args.device, check_output=args.check_output)
        try:
            results = run(args.input, args.out, settings, args.workers, args.spec_layout)
        except RunFailed as exc:
            print(exc, file=sys.stderr)
            return exc.code
        for result in results:
            print(_describe(result))
        print(args.out / "episode.h5")
        return 0

    cameras = args.cameras.split(",") if args.cameras else list(df2.PACK_ORDER)
    if len(cameras) < len(args.mp4):
        ap.error(f"{len(args.mp4)} mp4s but only {len(cameras)} camera names")
    for path in pack(args.mp4, cameras, args.audio, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
