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

from .detect import Detection, DetectorBank, filter_candidates, iou, weak_candidates
from .motion import downscale, estimate_shift
from .pipeline import (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_STOPPED,
                       AuditRecord, ProgressCallback, sidecar_path, tracker_for)
from .redact import redact
from .segments import (Segment, SegmentEncoder, concat, cut_copy, decode_range, frame_times,
                       keyframes, plan_segments, reorder_delay, split_long, verify)
from .settings import Settings
from .verify import check
from .video import VideoError, VideoInfo, part_path, probe

Submit = Callable[[Callable, list, int], list]


def serial_submit(func: Callable, jobs: list, limit: int = 1) -> list:
    return [func(job) for job in jobs]


class PoolSubmit:
    """Maps phase jobs over a multiprocessing pool, at most `limit` in flight."""

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, func, jobs, limit):
        limit = max(1, limit)
        pending, results = [], []
        for job in jobs:
            pending.append(self.pool.apply_async(func, (job,)))
            if len(pending) >= limit:
                results.append(pending.pop(0).get())
        while pending:
            results.append(pending.pop(0).get())
        return results


def init_pool_worker(workers: int = 1) -> None:
    """Pool initialiser: split the cores between workers."""
    import os

    global _THREADS
    _THREADS = None if workers <= 1 else max(1, (os.cpu_count() or 2) // workers)
    if _THREADS:
        import cv2
        cv2.setNumThreads(_THREADS)


# ------------------------------------------------------------- worker side

_BANKS: dict = {}
_THREADS = None      # a pool initialiser may cap the threads per worker


def _bank(settings: Settings) -> DetectorBank:
    key = (settings.engine, settings.det_sizes, settings.verify, settings.device,
           settings.confirm_on_crops, settings.crop_size, settings.crop_scale,
           settings.max_face_frac, settings.grow_face_frac, settings.third_opinion,
           settings.nms_detect, settings.nms_yunet)
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
    strong, weak, shifts = {}, {}, {}
    shape = (info.height, info.width)
    progress = job.get("progress")          # only when the job runs in-process
    prev_small = None
    for offset, frame in enumerate(decode_range(src, times, start, end, info, settings.hwaccel)):
        i = start + offset
        if progress is not None and offset % 5 == 0:
            progress(offset)
        if settings.camera_comp:
            # The first frame of a range has no frame before it here; its
            # shift stays zero, one frame in fifteen seconds.
            small = downscale(frame)
            shifts[i] = estimate_shift(prev_small, small, max(info.width, info.height))
            prev_small = small
        if i % settings.stride:
            continue
        raw = bank.detect_raw(frame)
        strong[i] = filter_candidates(raw, settings, shape)
        weak[i] = weak_candidates(raw, settings, shape)
    return {"start": start, "end": end, "strong": strong, "weak": weak, "shifts": shifts,
            "compute": bank.compute, "hashes": bank.model_hashes}


def encode_job(job: dict) -> dict:
    """Decode, redact and encode one segment to its own file."""
    src, info, times = Path(job["src"]), job["info"], job["times"]
    settings: Settings = job["settings"]
    seg: Segment = job["segment"]
    dets: dict[int, list[Detection]] = job["dets"]
    out = Path(job["out"])
    progress = job.get("progress")
    encoders = [settings.encoder]
    if settings.encoder != "x264":
        encoders.append("x264")      # if the GPU encoder refuses, the CPU one will not
    last_error = ""
    for encoder_name in encoders:
        encoder = SegmentEncoder(out, info, settings.crf, settings.preset,
                                 encoder_name, settings.nvenc_cq)
        masked = []
        try:
            for offset, frame in enumerate(decode_range(src, times, seg.start, seg.end,
                                                        info, settings.hwaccel)):
                if progress is not None and offset % 5 == 0:
                    progress(offset)
                d = dets.get(seg.start + offset, [])
                frame_out, alpha = redact(frame, d, settings)
                masked.append(float((alpha >= 0.5).mean()) if d else 0.0)
                encoder.write(frame_out)
            error = encoder.close()
        except VideoError as exc:
            encoder.abort()
            error = str(exc)
        except BaseException:
            encoder.abort()
            raise
        if not error and len(masked) == seg.frames:
            return {"segment": seg, "masked": masked, "out": str(out), "encoder": encoder_name}
        last_error = error or (f"decoded {len(masked)} frames, expected {seg.frames}")
        out.unlink(missing_ok=True)
    raise VideoError(f"Could not write the blurred copy. Segment {seg.start}-{seg.end}: "
                     f"{last_error[:300]}")


def copy_job(job: dict) -> dict:
    seg: Segment = job["segment"]
    cut_copy(Path(job["src"]), job["times"], seg, Path(job["out"]))
    return {"segment": seg, "masked": [0.0] * seg.frames, "out": job["out"]}


# ------------------------------------------------------------- parent side

def unconfirmed_runs(weak, per_frame, conf: float, min_run: int = 3,
                     limit: int = 100) -> tuple[list[list[int]], int]:
    """Stretches of at least `min_run` frames in which the primary detector
    saw a box at full threshold that nothing confirmed and no mask covers.

    Most are hands and objects, some are faces the confirmers did not
    recognise. The audit record lists them so a batch can be spot checked
    where it matters, and the evaluation harness counts them before and
    after a change. Returns ([start, end] pairs, flagged frame count).
    """
    flagged = []
    for i, boxes in enumerate(weak):
        hit = any(d.score >= conf and not any(iou(d, m) >= 0.3 for m in per_frame[i])
                  for d in boxes)
        flagged.append(hit)
    runs: list[list[int]] = []
    start = None
    for i, f in enumerate(flagged + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_run:
                runs.append([start, i - 1])
            start = None
    return runs[:limit], sum(flagged)


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
    shifts: list[tuple[float, float]] = [(0.0, 0.0)] * n
    done_frames = 0
    compute, hashes = {}, {}
    in_process = submit is serial_submit
    if in_process:
        # Jobs run here, so they can report every few frames rather than only
        # when a whole range finishes. A pool cannot carry the callable.
        for job in jobs:
            job["progress"] = (lambda k, base=job["start"]: report("detecting", done_frames + k, n))
    report("detecting", 0, n)

    def collect_detect(result):
        nonlocal done_frames, compute, hashes
        for i, d in result["strong"].items():
            strong[i] = d
        for i, d in result["weak"].items():
            weak[i] = d
        for i, sh in result.get("shifts", {}).items():
            shifts[i] = sh
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
    per_frame, tracks = tracker_for(settings).run(
        strong, weak, shifts if settings.camera_comp else None, (info.height, info.width))
    record.tracks = len(tracks)
    if settings.camera_comp:
        mag = np.hypot(*np.asarray(shifts, dtype=np.float64).T) if n else np.zeros(1)
        record.camera_shift_p95 = round(float(np.percentile(mag, 95)), 2)
        record.camera_shift_max = round(float(mag.max()), 2)
    record.unconfirmed_runs, record.unconfirmed_frames = unconfirmed_runs(
        weak, per_frame, settings.conf)
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
    # Copied pieces must carry the same reordering delay as the encoded ones,
    # which have none. A source with B-frames is encoded whole.
    record.reorder_delay = reorder_delay(src)
    want_copy = settings.copy_clean and record.reorder_delay == 0
    for attempt, copy_clean in enumerate((want_copy, False)):
        if attempt and not want_copy:
            break
        segments = plan_segments(masked_flags, keys, n, min_copy, copy_clean)
        piece = max(1, int(round(settings.encode_seconds * info.fps))) if settings.encode_seconds else chunk
        segments = split_long(segments, keys, piece)
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
        record.join_attempts = attempt + 1
        record.error = problem
        if not problem:
            record.frames_copied = copied
            break
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
    record.status = STATUS_DONE

    # ---- 6. check the copy, and hold it back if it still shows a face
    if settings.check_output or settings.quarantine:
        t2 = time.time()
        report("checking", 0, n)
        try:
            found = check(src, dst, settings, settings.check_stride, submit, workers, chunk)
        except VideoError as exc:
            record.error = str(exc)
            record.wall_seconds = round(time.time() - started, 2)
            return record
        record.checked_frames = found["frames_checked"]
        record.residual_faces = found["residual_faces"]
        record.residual_frames = found["residual_frames"]
        record.residual_by_size = found["residual_by_size"]
        record.residual_runs = found["residual_runs"]
        record.residual_list = found["residual_list"]
        record.flat_boxes = found["flat_boxes"]
        record.check_seconds = round(time.time() - t2, 2)
        report("checking", n, n)
        big = [row for row in found["residual_list"]
               if row["px"] >= settings.quarantine_min_px]
        if settings.quarantine and big:
            dst = quarantine_output(dst)
            record.output = dst.name
            record.quarantined = True

    record.wall_seconds = round(time.time() - started, 2)
    if write_sidecar:
        import json
        sidecar_path(dst).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record


def quarantine_output(dst: Path) -> Path:
    """Move a copy that still shows a face out of the delivery folder.

    It is not deleted: it is the only blurred copy of that video, and the
    audit record beside it says which frames stopped it. Whoever picks it up
    decides whether to cut those frames, run it again with other settings, or
    look at it.
    """
    held = dst.parent / "quarantine"
    held.mkdir(parents=True, exist_ok=True)
    target = held / dst.name
    target.unlink(missing_ok=True)
    shutil.move(str(dst), str(target))
    sidecar_path(dst).unlink(missing_ok=True)
    return target


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
    if submit is serial_submit:
        for job in encode_jobs:
            job["progress"] = (lambda k: report("writing", done + k, total))
    report("writing", 0, total)
    # NVENC allows a handful of concurrent sessions on a consumer GPU.
    encode_limit = min(workers, settings.nvenc_sessions) if settings.encoder != "x264" else workers
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
