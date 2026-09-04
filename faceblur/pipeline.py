"""The one function the CLI and the UI both call.

Two passes over each video. Pass one detects and propagates. Pass two redacts and
encodes. Progress goes through a callback, not through stdout. Cancellation goes
through a threading.Event that the pipeline checks between frames.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from .detect import DetectorBank
from .redact import redact
from .settings import Settings
from .track import propagate
from .video import Encoder, VideoError, probe, read_frames

# stage is "detecting" or "writing". done counts frames. total can be 0 when the
# container does not report a frame count.
ProgressCallback = Callable[[str, int, int], None]

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
STATUS_SKIPPED = "skipped"


@dataclass
class AuditRecord:
    """The evidence a compliance reviewer asks for, one per output file."""

    source: str
    output: str
    status: str
    frames: int = 0
    resolution: str = ""
    fps: float = 0.0
    frames_with_detection: int = 0
    detections: int = 0
    frames_covered_after_propagation: int = 0
    engine: str = ""
    model_sha256: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    detect_seconds: float = 0.0
    encode_seconds: float = 0.0
    wall_seconds: float = 0.0
    faceblur_version: str = "0.1.0"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def sidecar_path(dst: Path) -> Path:
    """The audit record sits next to its output file."""
    return Path(str(dst) + ".json")


def _cancelled(cancel: Optional[threading.Event]) -> bool:
    return cancel is not None and cancel.is_set()


def process_video(
    src: Path,
    dst: Path,
    settings: Optional[Settings] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
    bank: Optional[DetectorBank] = None,
    write_sidecar: bool = True,
) -> AuditRecord:
    """Blur every detected face in `src` and write the result to `dst`.

    Returns an AuditRecord. Never raises for an unreadable or unwritable video:
    the failure lands in the record's status and error fields so that a batch can
    carry on with the next file.
    """
    src, dst = Path(src), Path(dst)
    settings = settings or Settings()
    record = AuditRecord(source=src.name, output=dst.name, status=STATUS_FAILED,
                         engine=settings.engine, settings=settings.to_dict())
    started = time.time()

    if dst.exists() and not settings.replace_existing:
        record.status = STATUS_SKIPPED
        return record

    def report(stage: str, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(stage, done, total)

    try:
        info = probe(src)
    except VideoError as exc:
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record

    record.resolution = f"{info.width}x{info.height}"
    record.fps = round(info.fps, 3)

    if bank is None:
        bank = DetectorBank(settings)
    record.model_sha256 = bank.model_hashes

    # Pass one: detect.
    t0 = time.time()
    per_frame: list[list[list[float]]] = []
    last: list[list[float]] = []
    try:
        for index, frame in enumerate(read_frames(src)):
            if _cancelled(cancel):
                record.status = STATUS_STOPPED
                record.wall_seconds = round(time.time() - started, 2)
                return record
            if index % settings.stride == 0:
                last = bank.detect(frame)
                per_frame.append(last)
            else:
                # Frames between detections start empty. Propagation fills them.
                per_frame.append([])
            report("detecting", index + 1, info.frame_count)
    except VideoError as exc:
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record

    if not per_frame:
        record.error = ("Could not read this video. Check that the file is not "
                        "open in another program.")
        record.wall_seconds = round(time.time() - started, 2)
        return record

    record.frames_with_detection = sum(1 for f in per_frame if f)
    record.detections = sum(len(f) for f in per_frame)

    per_frame = propagate(per_frame, settings.window, settings.grow,
                          settings.nms_propagate)
    record.frames_covered_after_propagation = sum(1 for f in per_frame if f)
    record.detect_seconds = round(time.time() - t0, 2)

    # Pass two: redact and encode.
    t1 = time.time()
    written = 0
    encoder = Encoder(src, dst, info, settings.crf, settings.preset)
    try:
        for index, frame in enumerate(read_frames(src)):
            if _cancelled(cancel):
                encoder.abort()
                sidecar_path(dst).unlink(missing_ok=True)
                record.status = STATUS_STOPPED
                record.wall_seconds = round(time.time() - started, 2)
                return record
            boxes = per_frame[index] if index < len(per_frame) else []
            encoder.write(redact(frame, boxes, settings.pad, settings.mode,
                                 settings.strength))
            written += 1
            report("writing", written, info.frame_count)
    except VideoError as exc:
        encoder.abort()
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record
    except BaseException:
        # A KeyboardInterrupt must not leave half a video in the output folder.
        encoder.abort()
        raise

    error = encoder.close()
    if error:
        dst.unlink(missing_ok=True)
        record.error = ("Could not write the blurred copy. See the log file for "
                        f"details. ffmpeg said: {error[:400]}")
        record.wall_seconds = round(time.time() - started, 2)
        return record

    record.frames = written
    record.encode_seconds = round(time.time() - t1, 2)
    record.wall_seconds = round(time.time() - started, 2)
    record.status = STATUS_DONE

    if write_sidecar:
        sidecar_path(dst).write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )
    return record


def output_path(src: Path, out_dir: Path, suffix: str = "_blurred") -> Path:
    """Where the blurred copy of `src` goes. Always .mp4, because the encoder is x264."""
    return Path(out_dir) / f"{Path(src).stem}{suffix}.mp4"
