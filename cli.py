#!/usr/bin/env python3
"""Command line front end over faceblur.pipeline.

    faceblur INPUT [-o OUTPUT] [--engine yunet|centerface|both] [--conf F]
             [--det-sizes 640,1280] [--stride N] [--persist N] [--pad F]
             [--mode blur|pixelate|solid] [--workers N] [--report PATH]
             [--no-progress] [--recursive]

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

# One detector bank per worker process, built once and reused across that
# worker's files. A bank holds an onnxruntime session, which cannot be pickled,
# so it is never passed between processes.
_BANK = None
_SETTINGS: Settings | None = None


def _init_worker(settings: Settings) -> None:
    global _BANK, _SETTINGS
    from faceblur.detect import DetectorBank

    _SETTINGS = settings
    _BANK = DetectorBank(settings)


def _run_one(job: tuple[str, str]) -> dict:
    src, dst = job
    record = process_video(Path(src), Path(dst), _SETTINGS, bank=_BANK)
    return record.to_dict()


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
        return (f"  -> {dst}  ({r['detections']} detections over {r['frames']} frames, "
                f"{r['frames_covered_after_propagation']}/{r['frames']} frames redacted, "
                f"{r['detect_seconds']}s detect, {r['encode_seconds']}s write)")
    if r["status"] == STATUS_SKIPPED:
        return f"  Already done: {dst}"
    if r["status"] == STATUS_STOPPED:
        return "  Stopped"
    return f"  Failed: {r['error']}"


def run_serial(jobs, settings, reporter) -> list[dict]:
    from faceblur.detect import DetectorBank

    bank = DetectorBank(settings)
    records = []
    for number, (src, dst) in enumerate(jobs, 1):
        reporter.line(f"[{number}/{len(jobs)}] {src.name}")

        def on_progress(stage, done, total, name=src.name):
            if total:
                reporter.status(f"  {name}: {stage} {100 * done / total:5.1f}%")
            else:
                reporter.status(f"  {name}: {stage} frame {done}")

        record = process_video(src, dst, settings, on_progress=on_progress, bank=bank)
        reporter.clear()
        reporter.line(describe(record, dst))
        records.append(record.to_dict())
    return records


def run_parallel(jobs, settings, workers, reporter) -> list[dict]:
    records: list[dict] = []
    context = multiprocessing.get_context("spawn")
    reporter.line(f"Running {workers} videos at a time.")
    with context.Pool(workers, initializer=_init_worker, initargs=(settings,)) as pool:
        payload = [(str(src), str(dst)) for src, dst in jobs]
        for number, record in enumerate(pool.imap(_run_one, payload), 1):
            src, dst = jobs[number - 1]
            reporter.line(f"[{number}/{len(jobs)}] {src.name}")
            reporter.line(describe(record, dst))
            records.append(record)
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
    parser.add_argument("--det-sizes", default="640,1280",
                        help="detection sizes as a long side in pixels "
                             "(default: %(default)s). Add 1920 for small faces")
    parser.add_argument("--stride", type=int, default=defaults.stride,
                        help="detect faces on every Nth frame (default: %(default)s). "
                             "1 is safest, higher is faster")
    parser.add_argument("--persist", type=int, default=defaults.persist,
                        help="frames to carry a box forward and backward "
                             "(default: %(default)s)")
    parser.add_argument("--pad", type=float, default=defaults.pad,
                        help="mask padding as a fraction of face size "
                             "(default: %(default)s)")
    parser.add_argument("--mode", choices=["blur", "pixelate", "solid"],
                        default=defaults.mode,
                        help="how to destroy the face pixels (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=None,
                        help="videos to process at the same time "
                             "(default: half the CPU count)")
    parser.add_argument("--report", default=None,
                        help="write one JSON file holding every audit record here")
    parser.add_argument("--recursive", action="store_true",
                        help="walk subfolders and mirror them in the output folder")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="do not print the progress line")
    return parser


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = Settings(
            engine=args.engine,
            conf=args.conf,
            det_sizes=parse_det_sizes(args.det_sizes),
            stride=args.stride,
            persist=args.persist,
            pad=args.pad,
            mode=args.mode,
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
    workers = max(1, min(workers, len(jobs)))

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
