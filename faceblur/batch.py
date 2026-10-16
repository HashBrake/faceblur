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

import multiprocessing
import os
import shutil
import statistics
from dataclasses import replace
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .detect import Detection, DetectorBank, filter_candidates, iou, weak_candidates
from .motion import downscale, estimate_shift
from .pipeline import (STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, STATUS_STOPPED,
                       AuditRecord, ProgressCallback, sidecar_path, tracker_for)
from .redact import redact_measured
from .segments import (Segment, SegmentEncoder, concat, cut_copy, decode_range, frame_times,
                       keyframes, plan_segments, reorder_delay, split_long, verify)
from .settings import Settings
from .stride import AdaptiveStride, shift_px
from .verify import (check, held_for, merge_checks, residual_job, second_chance_seeds,
                     summarise)
from .video import VideoError, VideoInfo, part_path, probe, unchecked_path

Submit = Callable[[Callable, list, int], list]


def serial_submit(func: Callable, jobs: list, limit: int = 1) -> list:
    return [func(job) for job in jobs]


def serial_stream(func: Callable, jobs, limit: int = 1):
    """`serial_submit` one result at a time, so that `jobs` may be a generator
    that yields its next job from what the caller made of the last result.
    None from `jobs` means "nothing to run until a result is back", and here
    every result is already back."""
    for job in jobs:
        if job is not None:
            yield func(job)


class WorkerLost(RuntimeError):
    """A worker process died with a job in hand.

    Not a VideoError: nothing about the video is wrong, and the pool that lost
    the worker cannot be trusted with the next job, so the caller has to see
    this and start a new pool rather than have run_video fold it into a record.
    """


WORKER_LOST = ("A worker process stopped before it finished, most likely because the "
               "PC ran out of memory. Run the video again with fewer workers.")


class PoolSubmit:
    """Maps phase jobs over a multiprocessing pool, at most `limit` in flight.

    `multiprocessing.Pool` replaces a worker that dies and never tells the
    job it was running, so a plain `get()` waits forever. That is what a run
    that "never finishes" was: the window at 10 workers ran the PC out of
    memory, one worker went, and the batch sat on its result. So the wait is
    in short steps, and between steps the workers are counted.
    """

    POLL_SECONDS = 2.0

    def __init__(self, pool):
        self.pool = pool

    def _workers(self) -> list:
        return list(getattr(self.pool, "_pool", None) or [])

    def _wait(self, result, watched: list):
        while True:
            try:
                return result.get(timeout=self.POLL_SECONDS)
            except multiprocessing.TimeoutError:
                pass
            if any(p.exitcode is not None for p in watched):
                raise WorkerLost(WORKER_LOST)
            # A replacement started since the last look is watched too, so a
            # second death is caught as well as the first.
            watched.extend(p for p in self._workers() if p not in watched)

    def __call__(self, func, jobs, limit):
        return list(self.stream(func, jobs, limit))

    def stream(self, func, jobs, limit):
        """Results one at a time, oldest job first, at most `limit` in flight.

        `jobs` may be a generator that decides its next job from the results
        the caller has seen so far (see `_overlapped`); None from it means
        "wait for the oldest job before asking me again". A caller that stops
        early (a cancelled run) still waits for the jobs in flight, so that
        none of them is left writing into a folder that is being removed."""
        limit = max(1, limit)
        watched = self._workers()
        pending = []
        try:
            for job in jobs:
                if job is None:
                    if pending:
                        yield self._wait(pending.pop(0), watched)
                    continue
                pending.append(self.pool.apply_async(func, (job,)))
                if len(pending) >= limit:
                    yield self._wait(pending.pop(0), watched)
            while pending:
                yield self._wait(pending.pop(0), watched)
        except GeneratorExit:
            for result in pending:
                try:
                    self._wait(result, watched)
                except Exception:
                    pass
            raise


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
    # Fresh per job: the policy's state never crosses a chunk, so the first
    # frame of every chunk is detected.
    adaptive = AdaptiveStride(settings)
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
        # Without camera_comp there is no shift, which counts as moving.
        if adaptive.skip(shift_px(shifts.get(i))):
            # Empty, as a frame off the plain stride is (and the tracker
            # reads a missing frame as empty).
            continue
        raw = bank.detect_raw(frame)
        strong[i] = filter_candidates(raw, settings, shape)
        weak[i] = weak_candidates(raw, settings, shape)
        adaptive.saw(strong[i], weak[i])
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
            "skipped_adaptive": adaptive.count,
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
                                 encoder_name, settings.nvenc_cq, settings.encode_gop_seconds)
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
                # Mask work on the box around the regions only; the numbers
                # are the full frame's, see `redact.redact_measured`.
                frame_out, share, shares, took_back = redact_measured(frame, d, settings, z)
                masked.append(share)
                prevented.append(took_back)
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


def write_or_check_job(job: dict) -> dict:
    """One pool takes both kinds when the check overlaps the write, and a
    submit maps one function. A check that cannot read its piece says so
    instead of failing the write: the whole-copy check then runs after the
    join and fails the run the way it always did."""
    if job.get("kind") != "check":
        return encode_job(job)
    try:
        return residual_job(job)
    except VideoError as exc:
        return {"found": None, "error": str(exc)}


def _stream_of(submit):
    """The one-result-at-a-time form of a submit this module knows, or None.

    The overlapped write feeds jobs from the results as they come back, which
    a submit that maps the whole list before returning cannot do. Any other
    submit keeps the old order."""
    if submit is serial_submit:
        return serial_stream
    if isinstance(submit, PoolSubmit):
        return submit.stream
    return None


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


def _mask_key(dets) -> Counter:
    """What a frame's masks are, for telling whether a round changed them.

    The fields the redaction reads, in any order. A score or a source that
    changed with the box where it was draws the same mask and is not a change.
    """
    def points(p):
        return None if p is None else tuple((round(float(a), 3), round(float(b), 3))
                                            for a, b in p)

    return Counter((d.kind, round(d.x, 3), round(d.y, 3), round(d.w, 3), round(d.h, 3),
                    points(d.landmarks), points(d.quad)) for d in dets)


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

    if not settings.replace_existing and (dst.exists() or _held_back_before(dst, settings)):
        record.status = STATUS_SKIPPED
        return record
    # A recursive run mirrors the source tree, so the folder may not exist yet,
    # and the work folder and the held copy are both made inside it.
    dst.parent.mkdir(parents=True, exist_ok=True)

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
    record.source_audio = bool(info.audio)
    record.source_tags_present = list(info.tags)

    # ---- 1. detect, in frame ranges
    t0 = time.time()
    chunk = max(1, int(round(settings.chunk_seconds * info.fps))) if settings.chunk_seconds else n
    jobs = [{"src": str(src), "info": info, "times": times, "settings": settings,
             "start": a, "end": b} for a, b in _chunks(n, chunk)]
    strong: list[list[Detection]] = [[] for _ in range(n)]
    weak: list[list[Detection]] = [[] for _ in range(n)]
    screens_found: dict[int, list[Detection]] = {}
    held_screens: dict[int, list[Detection]] = {}
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
        record.frames_skipped_adaptive += result.get("skipped_adaptive", 0)
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

        held_screens = hold(screens_found, settings, n)
        record.screens = sum(len(v) for v in screens_found.values())
        record.screen_frames = len(held_screens)
        for i, dets in held_screens.items():
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
    #
    # A function rather than a straight line because package F1 runs it
    # again: a second chance writes the file a second time from the source,
    # never by re-encoding the copy, so a face found late is blurred once and
    # not blurred twice. With `second_chance_scope` "changed" only the
    # stretches whose masks changed are encoded again, from the source; the
    # rest is stream copied from the first copy, packets unchanged.
    keys = keyframes(src, times)
    min_copy = max(1, int(round(settings.min_copy_seconds * info.fps)))
    piece = max(1, int(round(settings.encode_seconds * info.fps))) if settings.encode_seconds else chunk
    # Copied pieces must carry the same reordering delay as the encoded ones,
    # which have none. A source with B-frames is encoded whole.
    record.reorder_delay = reorder_delay(src)
    want_copy = settings.copy_clean and record.reorder_delay == 0
    # A copy the gate will judge is written to a holding name and only moved
    # once, to `dst` or to quarantine, after the check has spoken. Moving it
    # into place first and checking it there meant a process that died during
    # the check left a whole, ungated copy that looked delivered. Report 24.4.
    gated = settings.check_output or settings.quarantine
    out = unchecked_path(dst) if gated else dst
    # The hands the detection pass found go with the check when that pass
    # looked at every frame, so the check does not find them again.
    known = hands_found if settings.zone and settings.zone_stride == 1 else None
    # The check of each written piece runs while the next pieces encode. The
    # results wait in `overlapped` for `check_pass`, which then only sums.
    overlap = gated and settings.check_overlap and _stream_of(submit) is not None
    overlapped: Optional[list] = None

    def piece_checks() -> Optional[dict]:
        if not overlap:
            return None
        return {"chunk": chunk, "stride": settings.check_stride, "hands": known,
                "results": [], "complete": True}

    def write_pass() -> bool:
        """Write `out` from `src` with the masks in `per_frame`. Sets the
        record's quality numbers. False means it gave up and said why."""
        t1 = time.time()
        masked_flags = [bool(f) for f in per_frame]
        masked = masked_by_kind = prevented = None
        for attempt, copy_clean in enumerate((want_copy, False)):
            if attempt and not want_copy:
                break
            segments = split_long(plan_segments(masked_flags, keys, n, min_copy, copy_clean),
                                  keys, piece)
            work = tempfile.mkdtemp(prefix="faceblur_", dir=str(dst.parent))
            checks = piece_checks()
            try:
                outcome = _write_and_join(src, out, info, times, settings, segments, per_frame,
                                          submit, workers, Path(work), report, cancelled,
                                          zones, checks=checks)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            if outcome is None:
                record.status = STATUS_STOPPED
                out.unlink(missing_ok=True)
                return False
            masked, masked_by_kind, prevented, problem, copied = outcome
            record.join_attempts = attempt + 1
            record.error = problem
            if not problem:
                record.frames_copied = copied
                set_overlapped(checks)
                break
            out.unlink(missing_ok=True)
        if record.error:
            return False
        written.update(masked=masked, by_kind=masked_by_kind, prevented=prevented)
        set_quality(masked, masked_by_kind, prevented)
        record.encode_seconds = round(record.encode_seconds + time.time() - t1, 2)
        record.status = STATUS_DONE
        return True

    # What the copy in `out` was written with, frame by frame, so that a
    # scoped second chance can keep the numbers of the frames it copies.
    written: dict = {}
    copy_delay = None                   # reorder_delay(out), when first asked

    def set_overlapped(checks: Optional[dict]) -> None:
        nonlocal overlapped
        overlapped = checks["results"] if checks is not None and checks["complete"] else None
    # The scope "changed" keeps the check's per job results, so that a
    # round can re-check only what it re-encoded and merge the rest.
    scoped = bool(settings.second_chance) and settings.second_chance_scope == "changed"
    check_results: list = []

    def set_quality(masked, masked_by_kind, prevented) -> None:
        """The record's quality numbers, from per frame mask shares."""
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
        record.masked_in_zone_prevented = (round(float(np.mean(prevented)), 5)
                                           if prevented else 0.0)

    def changed_plan(previous):
        """The second write's segments for the scope "changed", or a string
        naming why the round has to write the whole file the way the scope
        "all" does (it goes in `second_chance_scope_fallback`).

        Encode segments are every keyframe interval of the first copy that
        holds a frame whose masks changed, give or take the margin; copy
        segments are the rest, cut from the first copy. A plan with no encode
        segment means nothing changed.
        """
        nonlocal copy_delay
        margin = int(round(settings.second_chance_margin_seconds * info.fps))
        changed = [False] * n
        for i in range(n):
            if _mask_key(previous[i]) != _mask_key(per_frame[i]):
                record.second_chance_changed_frames += 1
                a, b = max(0, i - margin), min(n, i + margin + 1)
                changed[a:b] = [True] * (b - a)
        if all(changed):
            return "all_changed"
        if not any(changed):
            return [[Segment(0, n, True)], None]
        try:
            # The first copy is what gets stream copied now, so its
            # reordering delay is the one that has to match the encoded
            # pieces, which have none. Our encoders write no B-frames, so it
            # is zero unless the copy holds pieces copied from a source that
            # has them, which `want_copy` already rules out. Checked anyway,
            # once per file: a mismatch would put copied frames a few places
            # off, and a whole rewrite is always right.
            if copy_delay is None:
                copy_delay = reorder_delay(out)
            if copy_delay != 0:
                return "reorder_delay"
            out_times = frame_times(out)
            out_keys = keyframes(out, out_times)
        except VideoError:
            return "probe_failed"
        # plan_segments snaps each changed stretch out to the keyframes
        # around it, merges what touches and folds clean stretches shorter
        # than min_copy into their neighbours, which is the same rule the
        # first write uses against the source.
        segments = split_long(plan_segments(changed, out_keys, n, min_copy, True),
                              out_keys, piece)
        if all(not s.copy for s in segments):
            return "no_copy_segment"
        return [segments, out_times]

    def scoped_write(segments, out_times):
        """Write the changed segments from the source and copy the rest from
        the first copy. Returns the re-encoded [start, end) ranges, None if
        the run stopped, or False if the join did not verify."""
        t1 = time.time()
        # A new name: the first copy is the copy source until the new file
        # verifies, and only then takes its place.
        again = Path(str(out) + ".again")
        work = tempfile.mkdtemp(prefix="faceblur_", dir=str(dst.parent))
        # Copied pieces come from the first copy here, and the first check's
        # verdict on them stands (`merge_checks`), so only encoded ones are
        # checked.
        checks = piece_checks()
        try:
            outcome = _write_and_join(src, again, info, times, settings, segments, per_frame,
                                      submit, workers, Path(work), report, cancelled, zones,
                                      copy_src=out, copy_times=out_times, checks=checks)
            if outcome is None:
                record.status = STATUS_STOPPED
                return None
            masked, masked_by_kind, prevented, problem, _ = outcome
            if problem:
                return False
            os.replace(again, out)
            set_overlapped(checks)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            again.unlink(missing_ok=True)
        ranges = []
        for seg in segments:
            if seg.copy:
                continue
            if ranges and ranges[-1][1] == seg.start:
                ranges[-1][1] = seg.end
            else:
                ranges.append([seg.start, seg.end])
        # Copied frames keep the numbers they were written with; the rest
        # take the new ones.
        old_kind = written["by_kind"] or {}
        merged_kind = {name: list(old_kind.get(name) or [0.0] * n)
                       for name in set(old_kind) | set(masked_by_kind)}
        for a, b in ranges:
            written["masked"][a:b] = masked[a:b]
            written["prevented"][a:b] = prevented[a:b]
            for name, row in merged_kind.items():
                new_row = masked_by_kind.get(name)
                row[a:b] = new_row[a:b] if new_row else [0.0] * (b - a)
        written["by_kind"] = merged_kind
        set_quality(written["masked"], merged_kind, written["prevented"])
        record.second_chance_reencoded_frames += sum(b - a for a, b in ranges)
        record.second_chance_ranges += [list(r) for r in ranges]
        record.encode_seconds = round(record.encode_seconds + time.time() - t1, 2)
        record.status = STATUS_DONE
        return ranges

    def check_pass(ranges=None):
        """Detect on the finished copy and put what it found in the record.

        Returns the check's own report, or None if it could not run. The
        quarantine decision is not made here: package F1 gets its rounds
        first, and a copy that is moved is a copy the next round cannot
        write to.

        `ranges` limits the check to those frames, for the scope "changed";
        the report is then the earlier check with those frames replaced.
        """
        nonlocal check_results, overlapped
        t2 = time.time()
        report("checking", 0, n)
        # With `check_overlap` every frame this check would read was read
        # piece by piece during the write, from the same encoded pixels, and
        # `check` only sums those results.
        pieces, overlapped = overlapped, None
        if pieces is not None:
            record.overlap_check_seconds = round(
                record.overlap_check_seconds + sum(r["seconds"] for r in pieces), 2)
        try:
            if not scoped:
                got = check(src, out, settings, settings.check_stride, submit, workers, chunk,
                            hands=known, done=pieces)
            else:
                fresh: list = []
                got = check(src, out, settings, settings.check_stride, submit, workers, chunk,
                            hands=known, ranges=ranges, results=fresh, done=pieces)
                if ranges is not None:
                    check_results = merge_checks(check_results, fresh, ranges)
                    got = summarise(check_results, settings)
                else:
                    check_results = fresh
        except VideoError as exc:
            # A copy the check could not look at is not delivered.
            record.error = str(exc)
            record.status = STATUS_FAILED
            return None
        record.checked_frames = got["frames_checked"]
        record.residual_faces = got["residual_faces"]
        record.residual_hands = got["residual_hands"]
        record.residual_in_zone = got["residual_in_zone"]
        record.residual_max_px = got["residual_max_px"]
        record.model_sha256 = {**record.model_sha256, **got["hand_models"]}
        record.compute = {**record.compute, **got["hand_compute"]}
        record.residual_frames = got["residual_frames"]
        record.residual_by_size = got["residual_by_size"]
        record.residual_runs = got["residual_runs"]
        record.residual_list = got["residual_list"]
        record.flat_boxes = got["flat_boxes"]
        record.flat_by_kind = got["flat_by_kind"]
        record.residual_screens = got["residual_screens"]
        record.residual_screen_frames = got["residual_screen_frames"]
        record.residual_screen_max_px = got["residual_screen_max_px"]
        record.residual_screen_runs = got["residual_screen_runs"]
        record.zone_pixels_changed_max = got["zone_pixels_changed_max"]
        record.zone_frames_over = got["zone_frames_over"]
        record.zone_list = got["zone_list"]
        record.check_seconds = round(record.check_seconds + time.time() - t2, 2)
        report("checking", n, n)
        return got

    if not write_pass():
        record.wall_seconds = round(time.time() - started, 2)
        return record

    try:
        # ---- 6. check the copy, and hold it back if it still shows a face
        found = None
        if settings.check_output or settings.quarantine:
            found = check_pass()
            if found is None:
                record.wall_seconds = round(time.time() - started, 2)
                return record

        # ---- 7. the second chance, package F1
        #
        # The check found faces in the copy on pixels nothing changed. They are
        # evidence: a detector saw them at a threshold this run already trusts,
        # on a frame the first pass looked at and passed over. So they go back in
        # as seeds and the tracker runs again, which is what turns a box on one
        # frame into a mask over the stretch the face is there for. Nothing is
        # lowered and nothing is guessed. Report section 20.
        if found is not None and settings.second_chance:
            t3 = time.time()
            record.residual_faces_first_pass = record.residual_faces
            for _ in range(settings.second_chance):
                seeds = second_chance_seeds(found, settings)
                if not seeds:
                    break
                record.second_chance_rounds += 1
                record.second_chance_seeds += sum(len(v) for v in seeds.values())
                previous = per_frame             # the masks `out` was written with
                for i, dets in seeds.items():
                    strong[i] = list(strong[i]) + list(dets)
                per_frame, tracks = tracker_for(settings).run(
                    strong, weak, shifts if settings.camera_comp else None,
                    (info.height, info.width))
                record.second_chance_tracks_added += len(tracks) - record.tracks
                record.tracks = len(tracks)
                if settings.wants("screen"):
                    # The tracker does not know about screens and the run above
                    # threw away what section 2b put in. Put it back.
                    for i, dets in held_screens.items():
                        if 0 <= i < n:
                            per_frame[i] = list(per_frame[i]) + list(dets)
                # A seed the tracker dropped goes back as a mask on its own frame.
                # `min_track` is there so that one spurious detection cannot become
                # a mask, and a seed is not one: it is a box a second detection
                # pass found in the finished copy, on pixels the first pass never
                # changed. Keeping the rule here throws away most of what the
                # check paid for, and it is measured: on 004100 eight seeds made
                # three tracks and five fell on the floor.
                #
                # With the run's own `tail` either side of the sighting, which is
                # what the tracker gives the last frame of a track and for the
                # same reason: a face seen on one frame was there just before and
                # just after, and the detector is what blinks. Measured, on
                # 004100: seeds alone left two faces visible at frames 402 and
                # 591, two and three frames from a seed, and the tail is what
                # closes them. The box is not grown along the tail, because
                # nothing here knows which way the face went.
                tail = max(0, settings.tail)
                for i, dets in seeds.items():
                    for d in dets:
                        if any(iou(d, other) >= 0.3 for other in per_frame[i]):
                            continue
                        record.second_chance_seeds_kept += 1
                        for k in range(i - tail, i + tail + 1):
                            if 0 <= k < n and not any(iou(d, o) >= 0.3 for o in per_frame[k]):
                                per_frame[k] = list(per_frame[k]) + [d]
                record.frames_with_mask = sum(1 for f in per_frame if f)
                plan = changed_plan(previous) if scoped else None
                ranges = None
                if isinstance(plan, str):
                    record.second_chance_scope_fallback = (
                        record.second_chance_scope_fallback or plan)
                elif plan is not None:
                    if all(s.copy for s in plan[0]):
                        # No mask changed, so the copy and the check stand
                        # as they are, and the next round would find the
                        # same seeds.
                        break
                    ranges = scoped_write(*plan)
                    if ranges is None:
                        record.second_chance_seconds = round(time.time() - t3, 2)
                        record.wall_seconds = round(time.time() - started, 2)
                        return record
                    if ranges is False:
                        record.second_chance_scope_fallback = (
                            record.second_chance_scope_fallback or "join_failed")
                if not ranges:
                    # The scope "all", or a scoped join that did not verify:
                    # the whole file, from the source, the way it always was.
                    ranges = None
                    if not write_pass():
                        record.second_chance_seconds = round(time.time() - t3, 2)
                        record.wall_seconds = round(time.time() - started, 2)
                        return record
                    record.second_chance_reencoded_frames += n - record.frames_copied
                found = check_pass(ranges)
                if found is None:
                    record.second_chance_seconds = round(time.time() - t3, 2)
                    record.wall_seconds = round(time.time() - started, 2)
                    return record
            record.second_chance_seconds = round(time.time() - t3, 2)

        # ---- 8. the gate, on whatever the last check found
        if found is not None:
            # Hands do not hold a copy back. They are in the record either way.
            # Which kinds do is `verify.held_for`, so that the gate and the report
            # of it cannot drift apart.
            held_back = held_for(found, settings)
            record.held_back_for = list(held_back)
            if settings.quarantine and held_back:
                dst = quarantine_output(dst, out)
                record.output = dst.name
                record.quarantined = True
        if gated and not record.quarantined:
            os.replace(out, dst)
    finally:
        # Whatever stopped the run between the write and the gate, the copy
        # that was never delivered does not stay behind under any name.
        if gated:
            out.unlink(missing_ok=True)

    record.wall_seconds = round(time.time() - started, 2)
    if write_sidecar:
        import json
        sidecar_path(dst).write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record


def _held_back_before(dst: Path, settings: Settings) -> bool:
    """An earlier run with the gate on already finished this video and held it
    back. Without this a restarted batch redoes every quarantined copy, and on
    footage where the gate holds most files that is most of the run.

    The record is written after the copy is moved, so a record in quarantine
    means the copy beside it is whole."""
    if not settings.quarantine:
        return False
    held = Path(dst).parent / "quarantine" / Path(dst).name
    return held.is_file() and sidecar_path(held).is_file()


def quarantine_output(dst: Path, copy: Optional[Path] = None) -> Path:
    """Move a copy that still shows a face out of the delivery folder.

    It is not deleted: it is the only blurred copy of that video, and the
    audit record beside it says which frames stopped it. Whoever picks it up
    decides whether to cut those frames, run it again with other settings, or
    look at it.

    `copy` is where the copy is now, `dst` when not given. Whatever sits at
    `dst` from an earlier run goes, so a held back copy never leaves an older
    one looking delivered.
    """
    copy = Path(copy) if copy is not None else Path(dst)
    held = dst.parent / "quarantine"
    held.mkdir(parents=True, exist_ok=True)
    target = held / dst.name
    target.unlink(missing_ok=True)
    shutil.move(str(copy), str(target))
    Path(dst).unlink(missing_ok=True)
    sidecar_path(dst).unlink(missing_ok=True)
    return target


def _write_and_join(src, dst, info, times, settings, segments, per_frame, submit, workers,
                    work: Path, report, cancelled, zones=None, copy_src=None, copy_times=None,
                    checks: Optional[dict] = None):
    """Returns (masked, by kind, prevented, problem text, frames copied), or None.

    `checks`, when given, has each piece checked as soon as it is written, on
    the same pool, while the next pieces encode: `chunk`, `stride` and
    `hands` say how, `residual_job` results go into its `results`, and its
    `complete` turns False if any piece could not be checked. Pieces copied
    from the source are checked from the source alone; pieces copied from
    `copy_src` are not checked at all, their verdict is the earlier check's.

    `masked` is the union share of each frame and `masked by kind` splits it.
    `prevented` is what the handled zone took back out of the mask. Copied
    stretches are zero in all three: copied from the source they carry no
    mask, and copied from an earlier copy their numbers are the caller's.

    Copy segments come from `copy_src` with its own `copy_times` when given,
    the source otherwise. Encode segments and the audio always come from the
    source.
    """
    copy_src = src if copy_src is None else copy_src
    copy_times = times if copy_times is None else copy_times
    parts = [work / f"seg{k:05d}.mp4" for k in range(len(segments))]
    encode_jobs, copy_jobs = [], []
    for seg, part in zip(segments, parts):
        if seg.copy:
            copy_jobs.append({"src": str(copy_src), "times": copy_times, "segment": seg,
                              "out": str(part)})
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
    if checks is not None:
        results, feed = _overlapped(submit, encode_jobs, encode_limit, workers, checks, src,
                                    info, times, settings,
                                    copy_jobs if copy_src is src else [], len(per_frame))
    else:
        results, feed = submit(encode_job, encode_jobs, encode_limit), None
    try:
        for result in results:
            if cancelled():
                return None
            if "found" in result:
                if result["found"] is None:
                    checks["complete"] = False
                else:
                    checks["results"].append(result)
                continue
            if feed is not None:
                feed(result)
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
    finally:
        if feed is not None:
            results.close()

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
    os.replace(part, dst)
    return masked, by_kind, prevented, "", sum(s.frames for s in segments if s.copy)


def _overlapped(submit, encode_jobs, encode_limit, workers, checks, src, info, times, settings,
                source_copies, n):
    """The encode jobs and their pieces' check jobs as one stream of results.

    Returns (results, feed): the caller hands every encode result to `feed`,
    which queues that piece's checks to go out next. Checks go out before the
    next encode, so the pool has check work while pieces are still being
    written; an encode goes out only while fewer than `encode_limit` are in
    flight, so `nvenc_sessions` holds. Checks of pieces copied from the
    source wait on nothing and go first.
    """
    queue: deque = deque()
    todo = deque(encode_jobs)
    in_flight = 0

    def check_jobs(seg: Segment, piece: Optional[Path]) -> list[dict]:
        base = {"kind": "check", "src": str(src), "settings": settings,
                "stride": checks["stride"], "info": info, "times": times}
        if piece is None:
            base["copied"] = True
        else:
            base.update(out=str(piece), out_info=probe(piece), out_times=frame_times(piece))
        step = max(1, checks["chunk"])
        jobs = []
        for k in range(seg.start, min(seg.end, n), step):
            job = dict(base, start=k, end=min(k + step, seg.end, n), out_start=k - seg.start)
            if checks["hands"] is not None:
                job["hands"] = {i: checks["hands"][i] for i in range(job["start"], job["end"])
                                if i in checks["hands"]}
            jobs.append(job)
        return jobs

    for job in source_copies:
        queue.extend(check_jobs(job["segment"], None))

    def jobs():
        nonlocal in_flight
        while queue or todo or in_flight:
            if queue:
                yield queue.popleft()
            elif todo and in_flight < encode_limit:
                in_flight += 1
                yield todo.popleft()
            else:
                yield None              # wait for an encode to come back

    def feed(result: dict) -> None:
        nonlocal in_flight
        in_flight -= 1
        try:
            queue.extend(check_jobs(result["segment"], Path(result["out"])))
        except VideoError:
            checks["complete"] = False

    return _stream_of(submit)(write_or_check_job, jobs(), workers), feed
