"""Blur one video in phases, each of which can run across worker processes.

    1. detect      frame ranges, in parallel, no overlap needed
    2. track       once, over the whole timeline, in the parent
    3. segment     copy or encode, decided per group of pictures
    4. write       encode masked segments in parallel, copy clean ones
    5. join        concat, source audio, verify; fall back to encoding
                   everything if a mixed join does not verify

`run_video` takes a `submit(func, jobs, limit)` callable that maps jobs to
results. The CLI hands it a process pool; the window hands it a serial mapper
inside its own worker; a cloud runner can hand it anything. `process_video` in
pipeline.py stays as the one process, one video path.
"""
from __future__ import annotations

import shutil
import statistics
from dataclasses import replace
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .detect import Detection, DetectorBank, filter_candidates, weak_candidates
from .pipeline import (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_STOPPED,
                       AuditRecord, ProgressCallback, sidecar_path, tracker_for)
from .redact import redact
from .segments import (Segment, SegmentEncoder, concat, cut_copy, decode_range, frame_times,
                       keyframes, plan_segments, split_long, verify)
from .settings import Settings
from .video import VideoError, VideoInfo, part_path, probe

Submit = Callable[[Callable, list, int], list]


def serial_submit(func: Callable, jobs: list, limit: int = 1) -> list:
    return [func(job) for job in jobs]


# ------------------------------------------------------------- worker side

_BANKS: dict = {}
_THREADS = None      # a pool initialiser may cap the threads per worker


def _bank(settings: Settings) -> DetectorBank:
    key = (settings.engine, settings.det_sizes, settings.verify, settings.device,
           settings.confirm_on_crops, settings.crop_size, settings.crop_scale,
           settings.max_face_frac, settings.nms_detect, settings.nms_yunet)
    bank = _BANKS.get(key)
    if bank is None:
        bank = DetectorBank(settings, threads=_THREADS)
        _BANKS.clear()
        _BANKS[key] = bank
    return bank


def detect_job(job: dict) -> dict:
    """Detect on frames start to end. Returns per frame strong and weak lists."""
    src, info, times = Path(job["src"]), job["info"], job["times"]
    settings: Settings = job["settings"]
    start, end = job["start"], job["end"]
    bank = _bank(settings)
    strong, weak = {}, {}
    shape = (info.height, info.width)
    for offset, frame in enumerate(decode_range(src, times, start, end, info)):
        i = start + offset
        if i % settings.stride:
            continue
        raw = bank.detect_raw(frame)
        strong[i] = filter_candidates(raw, settings, shape)
        weak[i] = weak_candidates(raw, settings, shape)
    return {"start": start, "end": end, "strong": strong, "weak": weak,
            "compute": bank.compute, "hashes": bank.model_hashes}


def encode_job(job: dict) -> dict:
    """Decode, redact and encode one segment to its own file."""
    src, info, times = Path(job["src"]), job["info"], job["times"]
    settings: Settings = job["settings"]
    seg: Segment = job["segment"]
    dets: dict[int, list[Detection]] = job["dets"]
    out = Path(job["out"])
    encoder = SegmentEncoder(out, info, settings.crf, settings.preset,
                             settings.encoder, settings.nvenc_cq)
    masked = []
    try:
        for offset, frame in enumerate(decode_range(src, times, seg.start, seg.end, info)):
            d = dets.get(seg.start + offset, [])
            frame_out, alpha = redact(frame, d, settings)
            masked.append(float((alpha >= 0.5).mean()) if d else 0.0)
            encoder.write(frame_out)
    except BaseException:
        encoder.abort()
        raise
    error = encoder.close()
    if error:
        raise VideoError(f"Could not write the blurred copy. ffmpeg said: {error[:300]}")
    if len(masked) != seg.frames:
        raise VideoError(f"Segment {seg.start}-{seg.end} decoded {len(masked)} frames, "
                         f"expected {seg.frames}")
    return {"segment": seg, "masked": masked, "out": str(out)}


def copy_job(job: dict) -> dict:
    seg: Segment = job["segment"]
    cut_copy(Path(job["src"]), job["times"], seg, Path(job["out"]))
    return {"segment": seg, "masked": [0.0] * seg.frames, "out": job["out"]}


# ------------------------------------------------------------- parent side

def _chunks(n: int, size: int) -> list[tuple[int, int]]:
    size = max(1, size)
    return [(a, min(n, a + size)) for a in range(0, n, size)]


def run_video(src: Path, dst: Path, settings: Settings,
              submit: Submit = serial_submit,
              on_progress: Optional[ProgressCallback] = None,
              cancel: Optional[threading.Event] = None,
              workers: int = 1,
              write_sidecar: bool = True) -> AuditRecord:
    """Blur every confirmed face in `src`, in phases, and write `dst`."""
    src, dst = Path(src), Path(dst)
    record = AuditRecord(source=src.name, output=dst.name, status=STATUS_FAILED,
                         engine=settings.engine, settings=settings.to_dict())
    started = time.time()

    def report(stage, done, total):
        if on_progress is not None:
            on_progress(stage, done, total)

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    if dst.exists() and not settings.replace_existing:
        record.status = STATUS_SKIPPED
        return record

    try:
        info = probe(src)
        times = frame_times(src)
    except VideoError as exc:
        record.error = str(exc)
        record.wall_seconds = round(time.time() - started, 2)
        return record
    n = len(times)
    if n == 0:
        record.error = ("Could not read this video. Check that the file is not open in "
                        "another program.")
        return record
    # Encoded segments get the rate the timestamps actually show, not the
    # nominal one from the container, so their length matches the copied
    # segments and the audio stays in step.
    if n > 1 and times[-1] > times[0]:
        info = replace(info, fps=(n - 1) / (times[-1] - times[0]))
    record.resolution = f"{info.width}x{info.height}"
    record.fps = round(info.fps, 3)

    # ---- 1. detect, in frame ranges
    t0 = time.time()
    chunk = max(1, int(round(settings.chunk_seconds * info.fps))) if settings.chunk_seconds else n
    jobs = [{"src": str(src), "info": info, "times": times, "settings": settings,
             "start": a, "end": b} for a, b in _chunks(n, chunk)]
    strong: list[list[Detection]] = [[] for _ in range(n)]
    weak: list[list[Detection]] = [[] for _ in range(n)]
    done_frames = 0
    compute, hashes = {}, {}

    def collect_detect(result):
        nonlocal done_frames, compute, hashes
        for i, d in result["strong"].items():
            strong[i] = d
        for i, d in result["weak"].items():
            weak[i] = d
        compute, hashes = result["compute"], result["hashes"]
        done_frames += result["end"] - result["start"]
        report("detecting", done_frames, n)

    try:
        for result in submit(detect_job, jobs, workers):
            if cancelled():
                record.status = STATUS_STOPPED
                return record
            collect_detect(result)
    except VideoError as exc:
        record.error = str(exc)
        return record
    record.detect_seconds = round(time.time() - t0, 2)
    record.compute = compute
    record.model_sha256 = hashes
    record.frames_with_detection = sum(1 for f in strong if f)
    record.detections = sum(len(f) for f in strong)
    record.verified_detections = sum(1 for f in strong for d in f if d.verified)

    # ---- 2. track
    per_frame, tracks = tracker_for(settings).run(strong, weak)
    record.tracks = len(tracks)
    if tracks:
        lengths = [len(t) for t in tracks]
        record.track_length_min = min(lengths)
        record.track_length_median = float(statistics.median(lengths))
        record.track_length_max = max(lengths)
    record.frames_with_mask = sum(1 for f in per_frame if f)
    if cancelled():
        record.status = STATUS_STOPPED
        return record

    # ---- 3 to 5. segment, write, join. Once with copies, once more without
    # if the mixed join does not verify.
    t1 = time.time()
    keys = keyframes(src, times)
    masked_flags = [bool(f) for f in per_frame]
    min_copy = max(1, int(round(settings.min_copy_seconds * info.fps)))
    for attempt, copy_clean in enumerate((settings.copy_clean, False)):
        if attempt and not settings.copy_clean:
            break
        segments = plan_segments(masked_flags, keys, n, min_copy, copy_clean)
        segments = split_long(segments, keys, chunk)
        work = tempfile.mkdtemp(prefix="faceblur_", dir=str(dst.parent))
        try:
            outcome = _write_and_join(src, dst, info, times, settings, segments, per_frame,
                                      submit, workers, Path(work), report, cancelled)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if outcome is None:
            record.status = STATUS_STOPPED
            dst.unlink(missing_ok=True)
            return record
        masked, problem, copied = outcome
        if not problem:
            record.frames_copied = copied
            record.join_attempts = attempt + 1
            break
        record.error = problem
        dst.unlink(missing_ok=True)
    if record.error:
        record.wall_seconds = round(time.time() - started, 2)
        return record

    m = np.asarray(masked) if masked else np.zeros(1)
    record.frames = n
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
        import json
        sidecar_path(dst).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record


def _write_and_join(src, dst, info, times, settings, segments, per_frame, submit, workers,
                    work: Path, report, cancelled):
    """Returns (masked per frame, problem text, frames copied) or None if stopped."""
    parts = [work / f"seg{k:05d}.mp4" for k in range(len(segments))]
    encode_jobs, copy_jobs = [], []
    for seg, part in zip(segments, parts):
        if seg.copy:
            copy_jobs.append({"src": str(src), "times": times, "segment": seg, "out": str(part)})
        else:
            dets = {i: per_frame[i] for i in range(seg.start, seg.end) if per_frame[i]}
            encode_jobs.append({"src": str(src), "info": info, "times": times,
                                "settings": settings, "segment": seg, "dets": dets,
                                "out": str(part)})
    masked = [0.0] * len(per_frame)
    done = 0
    total = sum(s.frames for s in segments)
    # NVENC allows a handful of concurrent sessions on a consumer GPU.
    encode_limit = min(workers, 3) if settings.encoder != "x264" else workers
    try:
        for result in submit(encode_job, encode_jobs, encode_limit):
            if cancelled():
                return None
            seg = result["segment"]
            masked[seg.start:seg.end] = result["masked"]
            done += seg.frames
            report("writing", done, total)
        for result in submit(copy_job, copy_jobs, workers):
            if cancelled():
                return None
            done += result["segment"].frames
            report("writing", done, total)
    except VideoError as exc:
        return masked, str(exc), 0

    part = part_path(dst)
    error = concat(parts, src, part, work / "list.txt")
    if error:
        part.unlink(missing_ok=True)
        return masked, f"Could not write the blurred copy. ffmpeg said: {error[:300]}", 0
    problem = verify(part, len(per_frame), times[-1] - times[0])
    if problem:
        part.unlink(missing_ok=True)
        return masked, f"The joined file did not verify: {problem}", 0
    import os
    os.replace(part, dst)
    return masked, "", sum(s.frames for s in segments if s.copy)
