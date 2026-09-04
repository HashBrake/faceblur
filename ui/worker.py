"""Batch execution in worker processes.

This module never imports Qt. Windows starts a worker process with spawn, which
re-imports the module that holds the worker function, so importing Qt here would
load the whole toolkit in every worker for nothing.

The pool sends progress back through a queue. The window drains that queue and
turns the messages into Qt signals.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from faceblur.batch import run_video, serial_submit
import faceblur.batch as batch
from faceblur.settings import Settings

# Per worker process. A detector bank holds an onnxruntime session, which cannot
# be pickled, so each worker builds its own once and reuses it.
_BANK = None
_SETTINGS: Settings | None = None
_QUEUE = None
_CANCEL = None

# Progress messages are throttled. A 4000 frame video would otherwise send 8000
# messages through the queue and swamp the window.
_MIN_SECONDS_BETWEEN_MESSAGES = 0.1


def init_worker(settings: Settings, queue, cancel, workers: int = 1) -> None:
    global _BANK, _SETTINGS, _QUEUE, _CANCEL
    _SETTINGS = settings
    _QUEUE = queue
    _CANCEL = cancel
    # One worker keeps the library defaults. A pool splits the cores.
    batch._THREADS = None if workers <= 1 else max(1, (os.cpu_count() or 2) // workers)
    _BANK = None


def run_one(job: tuple[int, str, str]) -> tuple[int, dict]:
    """Blur one video. `job` carries the row index the window uses to find its row."""
    index, src, dst = job
    last_sent = [0.0]
    last_stage = [""]

    def on_progress(stage: str, done: int, total: int) -> None:
        now = time.monotonic()
        if stage == last_stage[0] and now - last_sent[0] < _MIN_SECONDS_BETWEEN_MESSAGES:
            return
        last_sent[0] = now
        last_stage[0] = stage
        try:
            _QUEUE.put(("progress", index, stage, done, total))
        except Exception:
            # A closed queue means the run was stopped. Losing progress is fine.
            pass

    record = run_video(Path(src), Path(dst), _SETTINGS, serial_submit,
                       on_progress=on_progress, cancel=_CANCEL)
    return index, record.to_dict()
