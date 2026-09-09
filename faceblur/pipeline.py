"""The one function the CLI and the UI both call.

Two passes over each video. Pass one detects on every frame and links the
detections into tracks. Pass two destroys the face pixels and encodes.
Progress goes through a callback. Cancellation goes through a threading.Event.

`plan` is pass one on its own. The evaluation harness calls it to get the
masks a video would receive without encoding anything.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .detect import Detection, DetectorBank, filter_candidates, weak_candidates
from .motion import downscale, estimate_shift
from .redact import redact
from .settings import Settings
from .track import Track, Tracker
from .video import Encoder, VideoError, VideoInfo, probe, read_frames

ProgressCallback = Callable[[str, int, int], None]

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
STATUS_SKIPPED = "skipped"

UNREADABLE = ("Could not read this video. Check that the file is not open in "
              "another program.")


@dataclass
class AuditRecord:
    """The evidence a compliance reviewer asks for, one per output file."""

    source: str
    output: str
    status: str
    frames: int = 0
    resolution: str = ""
    fps: float = 0.0
    # Detection and tracking
    frames_with_detection: int = 0
    detections: int = 0
    verified_detections: int = 0
    tracks: int = 0
    track_length_min: int = 0
    track_length_median: float = 0.0
    track_length_max: int = 0
    frames_with_mask: int = 0
    # Quality: how much of each frame was destroyed
    masked_mean: float = 0.0
    masked_p95: float = 0.0
    masked_max: float = 0.0
    frames_over_budget: int = 0
    flagged_frames: list = field(default_factory=list)
    engine: str = ""
    model_sha256: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)   # provider each model ran on
    frames_copied: int = 0        # frames copied from the source untouched
    join_attempts: int = 0
    reorder_delay: int = 0        # source B-frame delay in frames; copies need 0
    camera_shift_p95: float = 0.0     # source pixels per frame
    camera_shift_max: float = 0.0
    # Stretches where the primary detector saw a box nothing confirmed and no
    # mask covers: hands and objects mostly, faces sometimes. See batch.unconfirmed_runs.
    unconfirmed_runs: list = field(default_factory=list)
    unconfirmed_frames: int = 0
    # What the finished copy still shows, from a second detection pass over the
    # output; empty unless the run was asked for it. See faceblur/verify.py.
    checked_frames: int = 0
    residual_faces: int = 0
    # Boxes the check found that MediaPipe's hand models say are the wearer's
    # own hand. They are in residual_list, marked, and in no other count.
    residual_hands: int = 0
    # The largest face still visible, over every row, not just the 200 the
    # record carries. This is what the quarantine gate reads.
    residual_max_px: int = 0
    residual_frames: int = 0
    residual_by_size: dict = field(default_factory=dict)
    residual_runs: list = field(default_factory=list)
    residual_list: list = field(default_factory=list)
    flat_boxes: int = 0
    quarantined: bool = False
    check_seconds: float = 0.0
    settings: dict = field(default_factory=dict)
    detect_seconds: float = 0.0
    encode_seconds: float = 0.0
    wall_seconds: float = 0.0
    faceblur_version: str = "0.2.0"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    info: VideoInfo
    per_frame: list[list[Detection]]
    tracks: list[Track]
    raw_hits: int
    raw_boxes: int
    verified: int
    detect_seconds: float
    stopped: bool = False


def sidecar_path(dst: Path) -> Path:
    return Path(str(dst) + ".json")


def output_path(src: Path, out_dir: Path, suffix: str = "_blurred") -> Path:
    return Path(out_dir) / f"{Path(src).stem}{suffix}.mp4"


def _cancelled(cancel: Optional[threading.Event]) -> bool:
    return cancel is not None and cancel.is_set()


def tracker_for(settings: Settings) -> Tracker:
    # With a stride, consecutive detections are stride frames apart, so the
    # allowed gap must cover that.
    return Tracker(settings.min_track, max(settings.max_gap, settings.stride - 1),
                   settings.tail, settings.track_iou, tail_before=settings.tail_before,
                   tail_grow=settings.tail_grow, link_dist=settings.link_dist,
                   fast_shift=settings.fast_shift if settings.camera_comp else 0.0,
                   max_gap_fast=settings.max_gap_fast,
                   tail_before_max=settings.tail_before_max, tail_trend=settings.tail_trend,
                   conf_weak=settings.conf_weak, conf_weak_long=settings.conf_weak_long,
                   established_after=settings.established_after, tail_long=settings.tail_long,
                   continue_after=settings.continue_after, weak_run=settings.weak_run,
                   sure_conf=settings.track_sure_conf,
                   stitch_gap=settings.stitch_gap, stitch_slack=settings.stitch_slack,
                   gap_grow=settings.gap_grow,
                   blur_shift=settings.blur_shift if settings.camera_comp else 0.0)


def plan(src: Path, settings: Settings, bank: DetectorBank,
         on_progress: Optional[ProgressCallback] = None,
         cancel: Optional[threading.Event] = None,
         raw_cache: Optional[dict] = None) -> Plan:
    """Pass one: detect every frame, link tracks. Raises VideoError.

    `raw_cache` maps frame index to raw candidates. When given, detection is
    skipped for frames it holds and new results are added to it.
    """
    info = probe(src)
    t0 = time.time()
    detected: list[list[Detection]] = []
    weak: list[list[Detection]] = []
    shifts: list[tuple[float, float]] = []
    last: list[Detection] = []
    verified = 0
    prev_small = None
    shape = None
    for index, frame in enumerate(read_frames(src)):
        if _cancelled(cancel):
            return Plan(info, [], [], 0, 0, 0, time.time() - t0, stopped=True)
        shape = frame.shape[:2]
        if settings.camera_comp:
            small = downscale(frame)
            shifts.append(estimate_shift(prev_small, small, max(shape)))
            prev_small = small
        if index % settings.stride == 0:
            if raw_cache is not None and index in raw_cache:
                raw = raw_cache[index]
            else:
                raw = bank.detect_raw(frame)
                if raw_cache is not None:
                    raw_cache[index] = raw
            last = filter_candidates(raw, settings, frame.shape[:2])
            detected.append(last)
            weak.append(weak_candidates(raw, settings, frame.shape[:2]))
            verified += sum(1 for d in last if d.verified)
        else:
            detected.append([])
            weak.append([])
        if on_progress is not None:
            on_progress("detecting", index + 1, info.frame_count)
    if not detected:
        raise VideoError(UNREADABLE)

    per_frame, tracks = tracker_for(settings).run(
        detected, weak, shifts if settings.camera_comp else None, shape)
    return Plan(info, per_frame, tracks,
                raw_hits=sum(1 for f in detected if f),
                raw_boxes=sum(len(f) for f in detected),
                verified=verified, detect_seconds=time.time() - t0)


def process_video(
    src: Path,
    dst: Path,
    settings: Optional[Settings] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
    bank: Optional[DetectorBank] = None,
    write_sidecar: bool = True,
) -> AuditRecord:
    """Blur every confirmed face in `src` and write the result to `dst`.

    Never raises for an unreadable or unwritable video: the failure lands in
    the record so a batch carries on with the next file.
    """
    src, dst = Path(src), Path(dst)
    settings = settings or Settings()
    record = AuditRecord(source=src.name, output=dst.name, status=STATUS_FAILED,
                         engine=settings.engine, settings=settings.to_dict())
    started = time.time()

    if dst.exists() and not settings.replace_existing:
        record.status = STATUS_SKIPPED
        return record

    if bank is None:
        bank = DetectorBank(settings)
    record.model_sha256 = bank.model_hashes
    record.compute = bank.compute

    try:
        p = plan(src, settings, bank, on_progress, cancel)
    except VideoError as exc:
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record
    if p.stopped:
        record.status = STATUS_STOPPED
        record.wall_seconds = round(time.time() - started, 2)
        return record

    record.resolution = f"{p.info.width}x{p.info.height}"
    record.fps = round(p.info.fps, 3)
    record.frames_with_detection = p.raw_hits
    record.detections = p.raw_boxes
    record.verified_detections = p.verified
    record.tracks = len(p.tracks)
    if p.tracks:
        lengths = [len(t) for t in p.tracks]
        record.track_length_min = min(lengths)
        record.track_length_median = float(statistics.median(lengths))
        record.track_length_max = max(lengths)
    record.frames_with_mask = sum(1 for f in p.per_frame if f)
    record.detect_seconds = round(p.detect_seconds, 2)

    t1 = time.time()
    written = 0
    masked: list[float] = []
    encoder = Encoder(src, dst, p.info, settings.crf, settings.preset,
                      settings.encoder, settings.nvenc_cq)
    try:
        for index, frame in enumerate(read_frames(src)):
            if _cancelled(cancel):
                encoder.abort()
                sidecar_path(dst).unlink(missing_ok=True)
                record.status = STATUS_STOPPED
                record.wall_seconds = round(time.time() - started, 2)
                return record
            dets = p.per_frame[index] if index < len(p.per_frame) else []
            out, alpha = redact(frame, dets, settings)
            masked.append(float((alpha >= 0.5).mean()) if dets else 0.0)
            encoder.write(out)
            written += 1
            if on_progress is not None:
                on_progress("writing", written, p.info.frame_count)
    except VideoError as exc:
        encoder.abort()
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record
    except BaseException:
        encoder.abort()
        raise

    error = encoder.close()
    if error:
        dst.unlink(missing_ok=True)
        record.error = ("Could not write the blurred copy. See the log file for "
                        f"details. ffmpeg said: {error[:400]}")
        record.wall_seconds = round(time.time() - started, 2)
        return record

    m = np.asarray(masked) if masked else np.zeros(1)
    record.frames = written
    record.masked_mean = round(float(m.mean()), 5)
    record.masked_p95 = round(float(np.percentile(m, 95)), 5)
    record.masked_max = round(float(m.max()), 5)
    over = [i for i, v in enumerate(masked) if v > settings.mask_budget]
    record.frames_over_budget = len(over)
    record.flagged_frames = over[:200]
    record.encode_seconds = round(time.time() - t1, 2)
    record.wall_seconds = round(time.time() - started, 2)
    record.status = STATUS_DONE

    if write_sidecar:
        sidecar_path(dst).write_text(json.dumps(record.to_dict(), indent=2),
                                     encoding="utf-8")
    return record
