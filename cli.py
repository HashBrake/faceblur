#!/usr/bin/env python3
"""Command line front end over faceblur.pipeline.

    faceblur INPUT [-o OUTPUT] [--engine yunet|centerface|both] [--conf F]
             [--det-sizes 1280,1920] [--stride N] [--no-verify] [--max-face F]
             [--min-track N] [--max-gap N] [--tail N] [--pad F]
             [--mode blur|pixelate|solid] [--workers N] [--tta 0|1|2]
             [--no-third] [--no-camera] [--report PATH] [--no-progress] [--recursive]

INPUT is a video file or a folder of videos. OUTPUT defaults to a folder named
<input>_blurred next to the input. The exit code is 1 when any file failed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

from faceblur.pipeline import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_STOPPED,
    AuditRecord,
    output_path,
    process_video,
)
from faceblur.settings import VIDEO_EXT, Settings, SettingsError, parse_det_sizes

# Worker processes run the phases of one video at a time: detection in frame
# ranges, then segment encodes. faceblur.batch holds the pool helpers.
from faceblur.batch import PoolSubmit, init_pool_worker, run_video, serial_submit  # noqa: E402


def find_videos(src: Path, recursive: bool) -> list[Path]:
    if src.is_file():
        return [src]
    walker = src.rglob("*") if recursive else src.glob("*")
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def default_output_dir(src: Path) -> Path:
    """Where copies go when the user names no output folder.

    A folder gets a sibling folder. A single file gets its copy next to itself.
    """
    if src.is_dir():
        return src.parent / f"{src.name}_blurred"
    return src.parent


def drop_own_output(files: list[Path], out_dir: Path, root: Path,
                    suffix: str) -> list[Path]:
    """Skip copies this tool made, when the copies land beside the sources.

    Without this, a second run over the same folder blurs a_blurred.mp4 again
    and writes a_blurred_blurred.mp4.
    """
    try:
        same_folder = out_dir.resolve() == root.resolve()
    except OSError:
        same_folder = False
    if not same_folder:
        return files
    return [f for f in files if not f.stem.endswith(suffix)]


def destination(src_file: Path, root: Path, out_dir: Path, recursive: bool,
                suffix: str) -> Path:
    """Where one blurred copy goes.

    A recursive run mirrors the folder structure under the output folder. A flat
    run puts every copy straight in the output folder.
    """
    if recursive and root.is_dir():
        relative = src_file.parent.relative_to(root)
        return output_path(src_file, out_dir / relative, suffix)
    return output_path(src_file, out_dir, suffix)


class Reporter:
    """Prints progress. The library itself prints nothing."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stderr.isatty()
        self._width = 0

    def line(self, text: str) -> None:
        self.clear()
        print(text, flush=True)

    def status(self, text: str) -> None:
        if not self.enabled:
            return
        self._width = max(self._width, len(text))
        print("\r" + text.ljust(self._width), end="", file=sys.stderr, flush=True)

    def clear(self) -> None:
        if self.enabled and self._width:
            print("\r" + " " * self._width + "\r", end="", file=sys.stderr, flush=True)
            self._width = 0


def describe(record: AuditRecord | dict, dst: Path) -> str:
    r = record if isinstance(record, dict) else record.to_dict()
    if r["status"] == STATUS_DONE:
        return (f"  -> {dst}  ({r['tracks']} faces tracked, "
                f"{r['frames_with_mask']}/{r['frames']} frames masked, "
                f"{100 * r['masked_mean']:.2f}% of the frame on average, "
                f"{r['frames_over_budget']} frames over budget, "
                f"{r.get('frames_copied', 0)} frames copied untouched, "
                f"{r['detect_seconds']}s detect, {r['encode_seconds']}s write, "
                f"{'/'.join(sorted(set(r.get('compute', {}).values())) or ['cpu'])})")
    if r["status"] == STATUS_SKIPPED:
        return f"  Already done: {dst}"
    if r["status"] == STATUS_STOPPED:
        return "  Stopped"
    return f"  Failed: {r['error']}"


def run_serial(jobs, settings, reporter) -> list[dict]:
    records = []
    for number, (src, dst) in enumerate(jobs, 1):
        reporter.line(f"[{number}/{len(jobs)}] {src.name}")

        def on_progress(stage, done, total, name=src.name):
            if total:
                reporter.status(f"  {name}: {stage} {100 * done / total:5.1f}%")
            else:
                reporter.status(f"  {name}: {stage} frame {done}")

        record = run_video(src, dst, settings, serial_submit, on_progress=on_progress)
        reporter.clear()
        reporter.line(describe(record, dst))
        records.append(record.to_dict())
    return records


def run_parallel(jobs, settings, workers, reporter) -> list[dict]:
    """One video at a time, its phases spread over the pool."""
    records: list[dict] = []
    context = multiprocessing.get_context("spawn")
    reporter.line(f"Running with {workers} workers.")
    with context.Pool(workers, initializer=init_pool_worker, initargs=(workers,)) as pool:
        submit = PoolSubmit(pool)
        for number, (src, dst) in enumerate(jobs, 1):
            reporter.line(f"[{number}/{len(jobs)}] {src.name}")

            def on_progress(stage, done, total, name=src.name):
                if total:
                    reporter.status(f"  {name}: {stage} {100 * done / total:5.1f}%")

            record = run_video(src, dst, settings, submit, on_progress=on_progress,
                               workers=workers)
            reporter.clear()
            reporter.line(describe(record, dst))
            records.append(record.to_dict())
    return records


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="faceblur",
        description="Blur every detected face in a video. Runs on this PC.",
    )
    parser.add_argument("input", help="video file or folder of videos")
    parser.add_argument("-o", "--output", default=None,
                        help="output folder (default: a folder named <input>_blurred)")
    parser.add_argument("--engine", choices=["yunet", "centerface", "both"],
                        default=defaults.engine,
                        help="which detector to run (default: %(default)s). "
                             "both runs two detectors and finds more faces")
    parser.add_argument("--conf", type=float, default=defaults.conf,
                        help="detection threshold (default: %(default)s). "
                             "Lower catches more faces and blurs more non-faces")
    parser.add_argument("--det-sizes", default="1280,1920",
                        help="detection sizes as a long side in pixels "
                             "(default: %(default)s)")
    parser.add_argument("--stride", type=int, default=defaults.stride,
                        help="detect faces on every Nth frame (default: %(default)s). "
                             "1 is safest, higher is faster")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the second detector's confirmation of each face")
    parser.add_argument("--tta", type=int, choices=[0, 1, 2], default=defaults.confirm_tta,
                        help="views the confirmation looks at: 0 the crop, 1 also its "
                             "mirror, 2 also a tighter crop (default: %(default)s)")
    parser.add_argument("--no-third", dest="third_opinion", action="store_false",
                        help="do not ask the third detector when the second is unsure")
    parser.add_argument("--no-camera", dest="camera_comp", action="store_false",
                        help="ignore the camera's own movement in the tracker")
    parser.add_argument("--max-face", type=float, default=defaults.max_face_frac,
                        help="largest face as a share of the frame's long side "
                             "(default: %(default)s)")
    parser.add_argument("--min-track", type=int, default=defaults.min_track,
                        help="detections a face needs before it is masked "
                             "(default: %(default)s)")
    parser.add_argument("--max-gap", type=int, default=defaults.max_gap,
                        help="frames a face may go undetected inside a track "
                             "(default: %(default)s)")
    parser.add_argument("--tail", type=int, default=defaults.tail,
                        help="frames the mask extends past a track's ends "
                             "(default: %(default)s)")
    parser.add_argument("--pad", type=float, default=defaults.pad,
                        help="ellipse pad for a box that has no landmarks "
                             "(default: %(default)s)")
    parser.add_argument("--mode", choices=["blur", "pixelate", "solid"],
                        default=defaults.mode,
                        help="how to destroy the face pixels (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes for detection ranges and segment "
                             "encodes (default: half the CPU count)")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default=defaults.device,
                        help="run the detectors on the GPU when one is available "
                             "(default: %(default)s)")
    parser.add_argument("--encoder", choices=["auto", "nvenc", "x264"],
                        default=defaults.encoder,
                        help="video encoder (default: %(default)s, NVENC when available)")
    parser.add_argument("--chunk-seconds", type=float, default=defaults.chunk_seconds,
                        help="detection runs in ranges of this length, in parallel "
                             "(default: %(default)s)")
    parser.add_argument("--no-copy", dest="copy_clean", action="store_false",
                        help="re-encode every frame instead of copying face free stretches")
    parser.add_argument("--hwaccel", choices=["none", "cuda"], default=defaults.hwaccel,
                        help="hardware decode (default: %(default)s)")
    parser.add_argument("--report", default=None,
                        help="write one JSON file holding every audit record here")
    parser.add_argument("--recursive", action="store_true",
                        help="walk subfolders and mirror them in the output folder")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="do not print the progress line")
    return parser


def default_workers() -> int:
    from faceblur.detect import default_workers as _default
    return _default()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = Settings(
            engine=args.engine,
            conf=args.conf,
            # Continuation can never be looser than the main threshold.
            conf_weak=min(Settings.conf_weak, args.conf),
            det_sizes=parse_det_sizes(args.det_sizes),
            stride=args.stride,
            verify=args.verify,
            confirm_tta=args.tta,
            third_opinion=args.third_opinion,
            camera_comp=args.camera_comp,
            max_face_frac=args.max_face,
            min_track=args.min_track,
            max_gap=args.max_gap,
            tail=args.tail,
            pad=args.pad,
            mode=args.mode,
            device=args.device,
            encoder=args.encoder,
            chunk_seconds=args.chunk_seconds,
            copy_clean=args.copy_clean,
            hwaccel=args.hwaccel,
        )
    except SettingsError as exc:
        print(f"faceblur: {exc}", file=sys.stderr)
        return 2

    src = Path(args.input)
    if not src.exists():
        print(f"faceblur: could not find {src}", file=sys.stderr)
        return 2

    files = find_videos(src, args.recursive)
    if not files:
        print(f"faceblur: found no videos in {src}", file=sys.stderr)
        return 2

    out_dir = Path(args.output) if args.output else default_output_dir(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = src if src.is_dir() else src.parent
    files = drop_own_output(files, out_dir, root, settings.suffix)
    if not files:
        print(f"faceblur: found no videos in {src}", file=sys.stderr)
        return 2
    jobs = [(f, destination(f, root, out_dir, args.recursive, settings.suffix))
            for f in files]

    workers = args.workers if args.workers is not None else default_workers()
    workers = max(1, workers)

    reporter = Reporter(args.progress)
    reporter.line(f"{len(jobs)} videos. Output folder: {out_dir}")
    started = time.time()

    try:
        if workers == 1:
            records = run_serial(jobs, settings, reporter)
        else:
            records = run_parallel(jobs, settings, workers, reporter)
    except KeyboardInterrupt:
        reporter.clear()
        print("Stopped.", file=sys.stderr)
        return 130

    done = sum(1 for r in records if r["status"] == STATUS_DONE)
    failed = sum(1 for r in records if r["status"] == STATUS_FAILED)
    skipped = sum(1 for r in records if r["status"] == STATUS_SKIPPED)

    summary = f"{done} videos done. {failed} failed."
    if skipped:
        summary += f" {skipped} already done."
    reporter.line(f"{summary} Total time {time.time() - started:.1f}s.")

    if args.report:
        Path(args.report).write_text(json.dumps(records, indent=2), encoding="utf-8")
        reporter.line(f"Report: {args.report}")

    return 1 if failed else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
