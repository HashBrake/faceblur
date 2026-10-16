"""Run one video in a child process and stop it when a stage overruns.

In the first cloud run one video sat for 12 hours inside the tracker, which is
pure Python in the process that runs the video. Nothing short of ending that
process stops it, so each video gets a process of its own. The child runs
`run_video`, with the detector pool when there is one, and tells the parent
over a pipe which stage it is in. The parent times each stage against
`max(watchdog_min_seconds, seconds per frame x frames)` and ends the child
when a stage runs past that.
"""
from __future__ import annotations

import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from .batch import PoolSubmit, WorkerLost, init_pool_worker, run_video, serial_submit
from .pipeline import STATUS_FAILED, AuditRecord, ProgressCallback
from .settings import Settings
from .video import part_path, unchecked_path

# How often the parent looks at the clock while the child is quiet, and how
# often the child may report progress within one stage. A stage change is
# always sent, so the parent starts timing the new stage on time.
POLL_SECONDS = 0.5
SEND_SECONDS = 0.5


def _seconds(spent: float) -> str:
    """Time spent, short but never rounded down to 0 s: one decimal below
    10 s, whole seconds above. The limit itself prints with :g, exactly as it
    was set, because a 0.3 s floor printed as "limit 0 s" read as no limit."""
    return f"{spent:.1f}" if spent < 10 else f"{spent:.0f}"


def new_pool(workers: int):
    """The detector pool, built the way the CLI builds it."""
    context = multiprocessing.get_context("spawn")
    return context.Pool(workers, initializer=init_pool_worker, initargs=(workers,))


def remove_leftovers(dst: Path) -> None:
    """What a video stopped part way leaves beside its output.

    The write goes to `.part` and is moved into place whole, so `dst` itself
    is either a finished copy or not there, and it is not touched.
    """
    for path in (part_path(dst), unchecked_path(dst), part_path(unchecked_path(dst))):
        path.unlink(missing_ok=True)
    for work in dst.parent.glob("faceblur_*"):
        if work.is_dir():
            shutil.rmtree(work, ignore_errors=True)


class _Sender:
    """The child's progress callback: forwards to the parent, throttled."""

    def __init__(self, conn):
        self.conn = conn
        self.stage = None
        self.last = 0.0

    def __call__(self, stage, done, total):
        now = time.monotonic()
        if stage == self.stage and now - self.last < SEND_SECONDS:
            return
        self.stage, self.last = stage, now
        self.conn.send(("progress", stage, done, total))


def video_entry(src: Path, dst: Path, settings: Settings, workers: int, conn) -> None:
    """The child's side: blur the video and send the record back."""
    send = _Sender(conn)
    try:
        if workers > 1:
            pool = new_pool(workers)
            try:
                record = run_video(src, dst, settings, PoolSubmit(pool),
                                   on_progress=send, workers=workers)
            except WorkerLost as exc:
                # The same as the CLI does: the lost worker's job left its
                # files behind, and the record says why there is no copy.
                part_path(dst).unlink(missing_ok=True)
                unchecked_path(dst).unlink(missing_ok=True)
                record = AuditRecord(source=src.name, output=dst.name,
                                     status=STATUS_FAILED, error=str(exc))
            finally:
                pool.terminate()
                pool.join()
        else:
            record = run_video(src, dst, settings, serial_submit, on_progress=send)
        conn.send(("record", record.to_dict()))
    finally:
        conn.close()


def _stop(child) -> None:
    child.terminate()
    child.join(10)
    if child.is_alive():
        child.kill()
        child.join()


def supervise(entry: Callable, src: Path, dst: Path, settings: Settings, workers: int = 1,
              on_progress: Optional[ProgressCallback] = None) -> AuditRecord:
    """Run `entry(src, dst, settings, workers, conn)` in a child and return its
    record, or a failed one if the child overran a stage or died."""
    src, dst = Path(src), Path(dst)
    started = time.time()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    # Not a daemon: the child starts the detector pool, and a daemon process
    # may not have children of its own.
    child = context.Process(target=entry, args=(src, dst, settings, workers, sender))
    child.start()
    # The parent's copy of the sending end is closed, so the pipe breaks when
    # the child goes and a read does not wait for a message that cannot come.
    sender.close()

    def failed(error: str, stage: str = "") -> AuditRecord:
        return AuditRecord(source=src.name, output=dst.name, status=STATUS_FAILED,
                           engine=settings.engine, settings=settings.to_dict(),
                           stopped_in_stage=stage, error=error,
                           wall_seconds=round(time.time() - started, 2))

    per_frame = dict(settings.watchdog_seconds_per_frame)
    stage, stage_started, frames = None, time.monotonic(), 0
    try:
        while True:
            try:
                message = receiver.recv() if receiver.poll(POLL_SECONDS) else None
            except (EOFError, OSError):
                break
            if message is not None and message[0] == "record":
                child.join()
                return AuditRecord(**message[1])
            if message is not None:
                _, now_in, done, total = message
                if now_in != stage:
                    stage, stage_started = now_in, time.monotonic()
                if not frames and total > 0:
                    frames = total
                if on_progress is not None:
                    on_progress(now_in, done, total)
            elif not child.is_alive():
                break
            spent = time.monotonic() - stage_started
            if stage is None:
                # Before the first stage the child is probing the file, and
                # ffmpeg on a damaged file can hang there as surely as the
                # tracker can. The floor alone is the limit; 0 leaves it open.
                limit = settings.watchdog_min_seconds
                if limit > 0 and spent > limit:
                    _stop(child)
                    remove_leftovers(dst)
                    return failed(f"Watchdog stopped the video in the starting stage after "
                                  f"{_seconds(spent)} s (limit {limit:g} s): no stage was "
                                  f"reported", "starting")
                continue
            # A stage nobody named has no limit.
            if not per_frame.get(stage):
                continue
            limit = max(settings.watchdog_min_seconds, per_frame[stage] * frames)
            if spent > limit:
                _stop(child)
                remove_leftovers(dst)
                return failed(f"Watchdog stopped the video in the {stage} stage after "
                              f"{_seconds(spent)} s (limit {limit:g} s)", stage)
    finally:
        receiver.close()
    # The pipe broke or the child ended without a record.
    child.join(10)
    if child.is_alive():
        _stop(child)
    remove_leftovers(dst)
    return failed(f"The worker process died before it finished the video "
                  f"(exit code {child.exitcode}).")
