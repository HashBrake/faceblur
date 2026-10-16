"""The watchdog: one child process per video, ended when a stage overruns.

The slow stage is injected through `supervise`'s `entry` parameter, so the
entries below live at module level where a spawned child can import them.
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import cli
from faceblur import watchdog
from faceblur.batch import init_pool_worker, run_video, serial_submit
from faceblur.pipeline import STATUS_DONE, STATUS_FAILED, AuditRecord
from faceblur.settings import Settings
from faceblur.video import part_path
from faceblur.watchdog import supervise
from faceblur.watchdog import video_entry as real_video_entry

FAST = ["--det-sizes", "320", "--no-verify", "--min-track", "1", "--no-progress"]

# No floor, so the start-up stays untimed: a child importing the pipeline on a
# busy machine can take longer than any limit short enough for a test. 100
# frames at 0.01 s is a 1 s limit on a named stage.
QUICK = Settings(device="cpu", watchdog_min_seconds=0.0,
                 watchdog_seconds_per_frame=(("detecting", 0.01), ("tracking", 0.01),
                                             ("writing", 0.01), ("checking", 0.01)))
# A 1 s floor, which is also the start-up's limit.
FLOOR = QUICK.with_changes(watchdog_min_seconds=1.0)


# ------------------------------------------------------------ child entries

def stalls_in_tracking(src, dst, settings, workers, conn):
    """Leaves the files a real run leaves mid-way, then hangs in tracking."""
    part_path(dst).write_bytes(b"half")
    (dst.parent / "faceblur_xyz").mkdir()
    (dst.parent / "faceblur_xyz" / "seg00000.mp4").write_bytes(b"x")
    (dst.parent / "pid.txt").write_text(str(os.getpid()))
    conn.send(("progress", "detecting", 100, 100))
    conn.send(("progress", "tracking", 0, 100))
    time.sleep(120)


def finishes(src, dst, settings, workers, conn):
    conn.send(("progress", "detecting", 0, 10))
    conn.send(("progress", "detecting", 10, 10))
    record = AuditRecord(source=src.name, output=dst.name, status=STATUS_DONE,
                         frames=10, error="", settings=settings.to_dict())
    conn.send(("record", record.to_dict()))
    conn.close()


def never_reports(src, dst, settings, workers, conn):
    """Hangs before the first stage, the way ffprobe can on a damaged file."""
    (dst.parent / "pid.txt").write_text(str(os.getpid()))
    time.sleep(120)


def finishes_slowly(src, dst, settings, workers, conn):
    """Quiet for longer than FLOOR's limit before every message."""
    time.sleep(1.5)
    conn.send(("progress", "tracking", 0, 10))
    time.sleep(1.5)
    finishes(src, dst, settings, workers, conn)


def dies(src, dst, settings, workers, conn):
    conn.send(("progress", "detecting", 0, 10))
    os._exit(3)


def slow_clip_stalls(src, dst, settings, workers, conn):
    """Stalls in tracking on any clip named slow, counting its attempts beside
    the output; blurs every other clip for real."""
    if "slow" not in src.name:
        # Bound when this module was imported, before any monkeypatch, so it
        # is the real entry in the parent and in the child alike.
        real_video_entry(src, dst, settings, workers, conn)
        return
    with open(dst.parent / "attempts.txt", "a", encoding="utf-8") as attempts:
        attempts.write(f"{os.getpid()}\n")
    conn.send(("progress", "detecting", 20, 20))
    conn.send(("progress", "tracking", 0, 20))
    time.sleep(120)


def _grandchild(folder):
    init_pool_worker(2)
    (Path(folder) / "grandchild.txt").write_text(str(os.getpid()))
    time.sleep(60)


def holds_a_worker(folder):
    worker = multiprocessing.get_context("spawn").Process(target=_grandchild, args=(folder,))
    worker.start()
    worker.join()


# ------------------------------------------------------------------ helpers

def _alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259                      # STILL_ACTIVE
    finally:
        kernel.CloseHandle(handle)


def _wait_dead(pid: int, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.2)
    return not _alive(pid)


# -------------------------------------------------------------------- tests

def test_run_video_reports_tracking_between_detecting_and_writing(video_silent, tmp_path):
    seen = []
    record = run_video(video_silent, tmp_path / "out.mp4", Settings(device="cpu", copy_clean=True),
                       serial_submit, on_progress=lambda stage, done, total: seen.append(stage))

    assert record.status == STATUS_DONE, record.error
    assert seen.index("detecting") < seen.index("tracking") < seen.index("writing")


def test_a_stage_past_its_limit_is_stopped_and_cleaned_up(tmp_path):
    dst = tmp_path / "clip_blurred.mp4"
    started = time.monotonic()

    record = supervise(stalls_in_tracking, tmp_path / "clip.mp4", dst, QUICK)

    elapsed = time.monotonic() - started
    assert record.status == STATUS_FAILED
    assert record.stopped_in_stage == "tracking"
    assert "tracking" in record.error and "limit 1 s" in record.error
    assert record.source == "clip.mp4" and record.output == "clip_blurred.mp4"
    assert record.settings == QUICK.to_dict()
    # The rest is the child's start-up and the poll.
    assert elapsed < 1.0 + 8.0
    assert not part_path(dst).exists()
    assert not (tmp_path / "faceblur_xyz").exists()
    assert not dst.exists()
    assert _wait_dead(int((tmp_path / "pid.txt").read_text()), 5)


def test_a_record_sent_in_time_comes_back_unchanged(tmp_path):
    seen = []
    record = supervise(finishes, tmp_path / "clip.mp4", tmp_path / "clip_blurred.mp4", QUICK,
                       on_progress=lambda *args: seen.append(args))

    assert record.status == STATUS_DONE
    assert record.frames == 10
    assert record.error == ""
    assert record.stopped_in_stage == ""
    assert record.settings == QUICK.to_dict()
    # The second message is the same stage inside the throttle window.
    assert seen[0] == ("detecting", 0, 10)


def test_a_child_that_dies_without_a_record_fails_the_video(tmp_path):
    record = supervise(dies, tmp_path / "clip.mp4", tmp_path / "clip_blurred.mp4", QUICK)

    assert record.status == STATUS_FAILED
    assert "died" in record.error and "exit code 3" in record.error
    assert record.stopped_in_stage == ""


def test_a_watchdog_that_is_off_does_not_stop_anything(tmp_path):
    off = Settings(device="cpu", watchdog_min_seconds=0.0,
                   watchdog_seconds_per_frame=(("detecting", 0.0), ("tracking", 0.0),
                                               ("writing", 0.0), ("checking", 0.0)))
    record = supervise(finishes_slowly, tmp_path / "clip.mp4", tmp_path / "clip_blurred.mp4",
                       off)

    assert record.status == STATUS_DONE, record.error


def test_a_child_that_never_reports_a_stage_is_stopped_at_the_floor(tmp_path):
    started = time.monotonic()

    record = supervise(never_reports, tmp_path / "clip.mp4", tmp_path / "clip_blurred.mp4",
                       FLOOR)

    elapsed = time.monotonic() - started
    assert record.status == STATUS_FAILED
    assert record.stopped_in_stage == "starting"
    assert "starting" in record.error and "no stage was reported" in record.error
    assert elapsed < 1.0 + 8.0
    # On a busy machine the child can still be importing when the floor runs
    # out, so it may never have reached the entry and written its PID. The
    # process supervise started is gone either way.
    assert not multiprocessing.active_children()
    pid_file = tmp_path / "pid.txt"
    if pid_file.exists() and pid_file.read_text().strip():
        assert _wait_dead(int(pid_file.read_text()), 5)


def test_a_sub_second_floor_is_printed_as_it_was_set(tmp_path):
    # A 0.3 s floor once printed as "limit 0 s", and 0 means no limit.
    record = supervise(never_reports, tmp_path / "clip.mp4", tmp_path / "clip_blurred.mp4",
                       QUICK.with_changes(watchdog_min_seconds=0.3))

    assert record.status == STATUS_FAILED
    assert "limit 0.3 s" in record.error, record.error


def test_pool_workers_leave_when_their_parent_is_killed(tmp_path):
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=holds_a_worker, args=(str(tmp_path),))
    child.start()
    marker = tmp_path / "grandchild.txt"
    end = time.monotonic() + 60
    while not (marker.exists() and marker.read_text().strip()) and time.monotonic() < end:
        time.sleep(0.2)
    assert marker.exists(), "the grandchild never started"
    pid = int(marker.read_text())

    child.kill()
    child.join()

    assert _wait_dead(pid, 10)


def test_the_cli_retries_a_stopped_video_once_and_finishes_the_batch(
        video_silent, tmp_path, monkeypatch):
    folder = tmp_path / "batch"
    folder.mkdir()
    # Alphabetical order runs the slow clip first, so the fine one shows the
    # batch carrying on after it.
    shutil.copy(video_silent, folder / "aaa_slow.mp4")
    shutil.copy(video_silent, folder / "bbb_fine.mp4")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    monkeypatch.setattr(watchdog, "video_entry", slow_clip_stalls)
    started = time.monotonic()

    # 0.01 makes the floor 6 s, and so every stage's limit on a short clip.
    code = cli.main([str(folder), "-o", str(out), "--workers", "1",
                     "--watchdog-scale", "0.01", "--report", str(report), *FAST])

    elapsed = time.monotonic() - started
    assert code == 1
    assert len((out / "attempts.txt").read_text(encoding="utf-8").splitlines()) == 2
    records = {r["source"]: r for r in json.loads(report.read_text(encoding="utf-8"))}
    assert len(records) == 2
    slow, fine = records["aaa_slow.mp4"], records["bbb_fine.mp4"]
    assert slow["status"] == STATUS_FAILED
    assert slow["stopped_in_stage"] == "tracking"
    assert "tracking" in slow["error"]
    assert fine["status"] == STATUS_DONE, fine["error"]
    assert (out / "bbb_fine_blurred.mp4").exists()
    assert not list(out.glob("*.part"))
    assert not list(out.glob("*.unchecked*"))
    assert not list(out.glob("faceblur_*"))
    assert elapsed < 60
