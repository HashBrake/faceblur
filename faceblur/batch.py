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
from .redact import build_alpha, masked_shares, redact
from .segments import (Segment, SegmentEncoder, concat, cut_copy, decode_range, frame_times,
                       keyframes, plan_segments, reorder_delay, split_long, verify)
from .settings import Settings
from .verify import check, held_for
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


_SCREENS: dict = {}


def _screens(settings: Settings):
    """The worker's cached screen detector, built the first time it is asked.

    Kept apart from `_bank` for the same reason the hand models are: most
    runs never ask for screens, and a model that is never used should not be
    loaded, still less loaded once per worker.
    """
    from .screens import ScreenDetector

    key = (settings.device, settings.screen_labels, settings.screen_conf,
           settings.screen_nms, settings.screen_pad)
    found = _SCREENS.get(key)
    if found is None:
        found = ScreenDetector(settings, threads=_THREADS)
        _SCREENS.clear()
        _SCREENS[key] = found
    return found


_ZONES: dict = {}


def _zone(settings: Settings):
    """The worker's cached hand finder for the handled zone.

    Its own cache rather than `_hand_rule`'s, because the two ask different
    questions of the same two models: the rule asks whether one flagged box is
    a hand, this asks where every hand in the frame is, and they are keyed by
    different settings.
    """
    from .zone import HandFinder

    key = (settings.zone_device, settings.zone_window, settings.zone_nms,
           settings.hand_conf, settings.hand_presence,
           settings.zone_min_hand_px, settings.zone_edge_frac)
    found = _ZONES.get(key)
    if found is None:
        found = HandFinder(settings, threads=_THREADS)
        _ZONES.clear()
        _ZONES[key] = found
    return found


_HAND_RULES: dict = {}


def _hand_rule(settings: Settings):
    """The worker's cached hand models, built the first time the check asks.

    Kept apart from `_bank` because most runs never build it: the models load
    only when the output-side check is on and the hand rule with it.
    """
    from .hands import HandRule

    key = (settings.hand_device, settings.hand_windows, settings.hand_conf)
    rule = _HAND_RULES.get(key)
    if rule is None:
        rule = HandRule(settings, threads=_THREADS)
        _HAND_RULES.clear()
        _HAND_RULES[key] = rule
    return rule


def detect_job(job: dict) -> dict:
    """Detect on frames start to end. Returns per frame strong and weak lists."""
    src, info, times = Path(job["src"]), job["info"], job["times"]
    settings: Settings = job["settings"]
    start, end = job["start"], job["end"]
    bank = _bank(settings)
    screens = _screens(settings) if settings.wants("screen") else None
    # The zone is not a kind and does not depend on what is being masked: it
    # is the region nothing is masked in, so it runs whenever it is on.
    finder = _zone(settings) if settings.zone else None
    strong, weak, shifts, found_screens, found_hands = {}, {}, {}, {}, {}
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
        if finder is not None and i % settings.zone_stride == 0:
            hands = finder.detect(frame)
            if hands:
                found_hands[i] = hands
        if i % settings.stride:
            continue
        if screens is not None:
            # The frame is already decoded for the face pass, and this is one
            # more inference on it at a sixteenth of the area.
            hits = screens.detect(frame)
            if hits:
                found_screens[i] = hits
        raw = bank.detect_raw(frame)
        strong[i] = filter_candidates(raw, settings, shape)
        weak[i] = weak_candidates(raw, settings, shape)
    compute = dict(bank.compute)
    hashes = dict(bank.model_hashes)
    if screens is not None:
        compute.update(screens.compute)
        hashes.update(screens.model_hashes)
    if finder is not None:
        compute.update(finder.compute)
        hashes.update(finder.model_hashes)
    return {"start": start, "end": end, "strong": strong, "weak": weak, "shifts": shifts,
            "screens": found_screens, "hands": found_hands,
            "compute": compute, "hashes": hashes}


def encode_job(job: dict) -> dict:
    """Decode, redact and encode one segment to its own file."""
    src, info, times = Path(job["src"]), job["info"], job["times"]
    settings: Settings = job["settings"]
    seg: Segment = job["segment"]
    dets: dict[int, list[Detection]] = job["dets"]
    zones: dict = job.get("zones") or {}
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
        # One share per frame per kind, beside the union share. A kind that
        # first appears part way through the segment is padded back to zero,
        # so every list is as long as `masked`.
        by_kind: dict[str, list[float]] = {}
        # What the zone took back out of the mask, per frame. This is the
        # number that says the zone is doing something rather than the
        # detectors finding nothing where the hands are.
        prevented: list[float] = []
        try:
            for offset, frame in enumerate(decode_range(src, times, seg.start, seg.end,
                                                        info, settings.hwaccel)):
                if progress is not None and offset % 5 == 0:
                    progress(offset)
                d = dets.get(seg.start + offset, [])
                z = zones.get(seg.start + offset)
                frame_out, alpha = redact(frame, d, settings, zone=z)
                masked.append(float((alpha >= 0.5).mean()) if d else 0.0)
                if d and z:
                    plain = build_alpha(frame.shape[:2], d, settings) >= 0.5
                    prevented.append(float((plain & ~(alpha >= 0.5)).mean()))
                else:
                    prevented.append(0.0)
                shares = (masked_shares(frame.shape[:2], d, settings, alpha, z)
                          if d else {})
                for name in set(by_kind) | set(shares):
                    row = by_kind.setdefault(name, [0.0] * (len(masked) - 1))
                    row.append(shares.get(name, 0.0))
                encoder.write(frame_out)
            error = encoder.close()
        except VideoError as exc:
            encoder.abort()
            error = str(exc)
        except BaseException:
            encoder.abort()
            raise
        if not error and len(masked) == seg.frames:
            return {"segment": seg, "masked": masked, "masked_by_kind": by_kind,
                    "prevented": prevented, "out": str(out), "encoder": encoder_name}
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
    screens_found: dict[int, list[Detection]] = {}
    hands_found: dict[int, list] = {}
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
        screens_found.update(result.get("screens") or {})
        hands_found.update(result.get("hands") or {})
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

    # ---- 2b. screens, which need no tracker: they are large, rigid and
    # found outright, so all they want is short gaps closed and a tail.
    if settings.wants("screen"):
        from .screens import hold

        held = hold(screens_found, settings, n)
        record.screens = sum(len(v) for v in screens_found.values())
        record.screen_frames = len(held)
        for i, dets in held.items():
            if 0 <= i < n:
                per_frame[i] = list(per_frame[i]) + list(dets)
    # ---- 2d. the handled zone, built once over the whole timeline because
    # its memory runs across frame range boundaries the way the tracker does.
    from .zone import build as build_zone, describe as describe_zone

    zones = build_zone(hands_found, shifts if settings.camera_comp else None,
                       settings, n, info.fps, (info.height, info.width))
    if settings.zone:
        for key, value in describe_zone(zones, (info.height, info.width)).items():
            setattr(record, key, value)
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
                                      submit, workers, Path(work), report, cancelled,
                                      zones)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if outcome is None:
            record.status = STATUS_STOPPED
            dst.unlink(missing_ok=True)
            return record
        masked, masked_by_kind, prevented, problem, copied = outcome
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
    # The union numbers above keep their names and their meaning. These split
    # them, because a screen mask is large by design and would otherwise make
    # the face numbers unreadable in any run that masks two kinds at once.
    record.masked_mean_by_kind = {k: round(float(np.mean(v)), 5)
                                  for k, v in sorted(masked_by_kind.items())}
    record.masked_max_by_kind = {k: round(float(np.max(v)), 5)
                                 for k, v in sorted(masked_by_kind.items())}
    record.frames_over_budget_by_kind = {
        k: sum(1 for x in v if x > settings.mask_budget)
        for k, v in sorted(masked_by_kind.items())}
    if prevented:
        record.masked_in_zone_prevented = round(float(np.mean(prevented)), 5)
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
        record.residual_hands = found["residual_hands"]
        record.residual_in_zone = found["residual_in_zone"]
        record.residual_max_px = found["residual_max_px"]
        record.model_sha256 = {**record.model_sha256, **found["hand_models"]}
        record.compute = {**record.compute, **found["hand_compute"]}
        record.residual_frames = found["residual_frames"]
        record.residual_by_size = found["residual_by_size"]
        record.residual_runs = found["residual_runs"]
        record.residual_list = found["residual_list"]
        record.flat_boxes = found["flat_boxes"]
        record.flat_by_kind = found["flat_by_kind"]
        record.residual_screens = found["residual_screens"]
        record.residual_screen_frames = found["residual_screen_frames"]
        record.residual_screen_max_px = found["residual_screen_max_px"]
        record.residual_screen_runs = found["residual_screen_runs"]
        record.zone_pixels_changed_max = found["zone_pixels_changed_max"]
        record.zone_frames_over = found["zone_frames_over"]
        record.zone_list = found["zone_list"]
        record.check_seconds = round(time.time() - t2, 2)
        report("checking", n, n)
        # Hands do not hold a copy back. They are in the record either way.
        # Which kinds do is `verify.held_for`, so that the gate and the report
        # of it cannot drift apart.
        held_back = held_for(found, settings)
        record.held_back_for = list(held_back)
        if settings.quarantine and held_back:
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
                    work: Path, report, cancelled, zones=None):
    """Returns (masked, by kind, prevented, problem text, frames copied), or None.

    `masked` is the union share of each frame and `masked by kind` splits it.
    `prevented` is what the handled zone took back out of the mask. Copied
    stretches carry no mask at all, so they are zero in all three.
    """
    parts = [work / f"seg{k:05d}.mp4" for k in range(len(segments))]
    encode_jobs, copy_jobs = [], []
    for seg, part in zip(segments, parts):
        if seg.copy:
            copy_jobs.append({"src": str(src), "times": times, "segment": seg, "out": str(part)})
        else:
            dets = {i: per_frame[i] for i in range(seg.start, seg.end) if per_frame[i]}
            in_zone = {i: zones[i] for i in range(seg.start, seg.end)
                       if zones is not None and i < len(zones) and zones[i]}
            encode_jobs.append({"src": str(src), "info": info, "times": times,
                                "settings": settings, "segment": seg, "dets": dets,
                                "zones": in_zone, "out": str(part)})
    masked = [0.0] * len(per_frame)
    by_kind: dict[str, list[float]] = {}
    prevented = [0.0] * len(per_frame)
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
            for name, values in (result.get("masked_by_kind") or {}).items():
                row = by_kind.setdefault(name, [0.0] * len(per_frame))
                row[seg.start:seg.end] = values
            if result.get("prevented"):
                prevented[seg.start:seg.end] = result["prevented"]
            done += seg.frames
            report("writing", done, total)
        for result in submit(copy_job, copy_jobs, workers):
            if cancelled():
                return None
            done += result["segment"].frames
            report("writing", done, total)
    except VideoError as exc:
        return masked, by_kind, prevented, str(exc), 0

    part = part_path(dst)
    error = concat(parts, src, part, work / "list.txt")
    if error:
        part.unlink(missing_ok=True)
        return (masked, by_kind, prevented,
                f"Could not write the blurred copy. ffmpeg said: {error[:300]}", 0)
    problem = verify(part, len(per_frame), times[-1] - times[0])
    if problem:
        part.unlink(missing_ok=True)
        return masked, by_kind, prevented, f"The joined file did not verify: {problem}", 0
    import os
    os.replace(part, dst)
    return masked, by_kind, prevented, "", sum(s.frames for s in segments if s.copy)
