"""Blur a whole bucket of videos on every card of one machine, a file at a time.

The machine's disk is far smaller than the footage, so nothing is copied in
bulk. Each card takes the next video from a shared queue, largest first,
downloads it, runs `cli.py` on it, uploads the copy and its record, and
deletes both local files, while the video after it downloads in the
background. Disk use stays at a few videos per card.

    python cloud/stream_run.py

Configuration is read from the environment (see cloud/run.env.example):
S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, BUCKET, IN_PREFIX (default "in"),
OUT_PREFIX (default "out"), WORK (local scratch, default /tmp/faceblur),
WORKERS (pipeline workers per card), RUN_ARGS (extra cli.py arguments),
GPUS (default: every card nvidia-smi lists), SHARD (k/n: this machine's
share when n machines run at once, default 0/1), LANES (videos worked on at
once per card, default 1).

Two lanes keep a card busy: a video spends about half its time in steps that
use one core and no GPU (the tracker, the zone, the encode), and on the RTX
5090 VMs the card sat idle through them.

A video counts as finished once its record is in the bucket, delivered or in
quarantine; the copy is uploaded before the record, so a record never stands
beside a missing copy. Run it again after anything stops it and only the
unfinished videos are taken.
"""
from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from faceblur.pipeline import output_path  # noqa: E402
from faceblur.settings import VIDEO_EXT, Settings  # noqa: E402

ENV = os.environ
BUCKET = ENV["BUCKET"]
IN = f"vs:{BUCKET}/{ENV.get('IN_PREFIX', 'in')}"
OUT = f"vs:{BUCKET}/{ENV.get('OUT_PREFIX', 'out')}"
WORK = Path(ENV.get("WORK", "/tmp/faceblur"))
WORKERS = ENV.get("WORKERS", "6")
LANES = int(ENV.get("LANES", "1"))
RUN_ARGS = shlex.split(ENV.get("RUN_ARGS", "--quarantine"))
SUFFIX = Settings().suffix

# rclone reads a remote called "vs" from these, so no config file holds keys.
ENV.update({"RCLONE_CONFIG_VS_TYPE": "s3", "RCLONE_CONFIG_VS_PROVIDER": "Other",
            "RCLONE_CONFIG_VS_ENDPOINT": ENV["S3_ENDPOINT"],
            "RCLONE_CONFIG_VS_ACCESS_KEY_ID": ENV["S3_ACCESS_KEY"],
            "RCLONE_CONFIG_VS_SECRET_ACCESS_KEY": ENV["S3_SECRET_KEY"]})

log_lock = threading.Lock()
LOG = WORK / "stream_run.log"


def say(text: str) -> None:
    line = f"[{time.strftime('%F %T')}] {text}"
    with log_lock:
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def rclone(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["rclone", *args, "--retries", "5", "--low-level-retries", "20"],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])}: {r.stderr.strip()[-500:]}")
    return r


def listing(remote: str) -> dict[str, int]:
    r = rclone("lsjson", remote, "--recursive", "--files-only", check=False)
    if r.returncode != 0:
        return {}
    return {e["Path"]: e["Size"] for e in json.loads(r.stdout or "[]")}


def record_names(rel: str) -> tuple[str, str]:
    """Where a video's record lands in the output, delivered and held back."""
    p = PurePosixPath(rel)
    copy = output_path(Path(p.name), Path("."), SUFFIX).name
    return (str(p.parent / f"{copy}.json"), str(p.parent / "quarantine" / f"{copy}.json"))


# SHARD=k/n: this machine takes share k of n, so several machines can work
# from one bucket without taking the same video. The shares are balanced by
# size over the whole listing, finished videos included, so every machine and
# every restart computes the same split.
SHARD = ENV.get("SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in SHARD.split("/"))
TAG = f"shard{SHARD_K}of{SHARD_N}_" if SHARD_N > 1 else ""


def share(sizes: dict[str, int]) -> set[str]:
    """Largest first, each to the share with the least so far; ties by path."""
    loads = [0] * SHARD_N
    taken: set[str] = set()
    for rel in sorted(sizes, key=lambda r: (-sizes[r], r)):
        k = loads.index(min(loads))
        loads[k] += sizes[rel]
        if k == SHARD_K:
            taken.add(rel)
    return taken


def gpu_count() -> int:
    if ENV.get("GPUS"):
        return int(ENV["GPUS"])
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return len(r.stdout.split())


CUDA_CHECK = ("import onnxruntime as o, sys; "
              "s = o.InferenceSession(sys.argv[1], providers=['CUDAExecutionProvider']); "
              "sys.exit(0 if s.get_providers()[0] == 'CUDAExecutionProvider' else 1)")


def card_works(k: int) -> bool:
    """onnxruntime falls back to the CPU on a card it cannot use and says so
    only in its log. A card on the CPU runs some fifty times slower and takes
    cores from every other card, so it gets no work at all."""
    r = subprocess.run([sys.executable, "-c", CUDA_CHECK, str(HERE.parent / "models" / "yunet.onnx")],
                       env=dict(ENV, CUDA_VISIBLE_DEVICES=str(k)), capture_output=True)
    return r.returncode == 0


def on_cuda(record: Path) -> bool:
    try:
        compute = json.loads(record.read_text(encoding="utf-8")).get("compute", {})
    except (OSError, ValueError):
        return True
    return all(v == "CUDAExecutionProvider" for v in compute.values())


def fetch(rel: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rclone("copyto", f"{IN}/{rel}", str(dest), "--multi-thread-streams", "8")


class Disk:
    """Room on the work disk, shared by every card.

    A video in progress needs its source, its copy (about 1.6 times the
    source at the encoder's quality) and the segments the copy is built from,
    so each one reserves `NEED` times its size before it downloads and gives
    it back when it is done. A card waits for room instead of filling the
    disk; with nothing else reserved a video goes ahead whatever its size."""

    NEED, SPARE = 3.5, 5e9

    def __init__(self, path: Path):
        self.room = shutil.disk_usage(path).free - self.SPARE
        self.held = 0.0
        self.cond = threading.Condition()

    def take(self, size: int) -> float:
        need = size * self.NEED + 1e9
        with self.cond:
            while self.held and self.held + need > self.room:
                self.cond.wait(30)
            self.held += need
        return need

    def give(self, need: float) -> None:
        with self.cond:
            self.held -= need
            self.cond.notify_all()


def lane_dir(k: int, lane: int) -> str:
    return f"gpu{k}" if lane == 0 else f"gpu{k}_{lane}"


def card(k: int, lane: int, jobs: "queue.Queue[str]", tally: dict, sizes: dict,
         disk: Disk) -> None:
    base = WORK / lane_dir(k, lane)
    name = f"card {k}" + (f" lane {lane}" if LANES > 1 else "")
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(k))
    nxt: tuple[str, threading.Thread, list] | None = None

    def start_fetch() -> tuple[str, threading.Thread, list] | None:
        try:
            rel = jobs.get_nowait()
        except queue.Empty:
            return None
        err: list = []

        def run():
            # The room is taken here, in the download thread, so the video
            # being worked on now never waits for it.
            err.append(disk.take(sizes[rel]))
            try:
                fetch(rel, base / "in" / rel)
            except Exception as exc:          # reported when the file is reached
                err.append(exc)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return rel, t, err

    nxt = start_fetch()
    while nxt is not None:
        rel, t, err = nxt
        t.join()
        nxt = start_fetch()                   # the next one downloads while this one runs
        src = base / "in" / rel
        out_dir = base / "out" / PurePosixPath(rel).parent
        started = time.time()
        reserved, err = err[0], err[1:]
        try:
            if err:
                raise err[0]
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(base / "cli.log", "a", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%F %T')} {rel}\n")
                log.flush()
                code = subprocess.run(
                    [sys.executable, str(HERE.parent / "cli.py"), str(src), "-o", str(out_dir),
                     "--no-progress", "--workers", WORKERS, *RUN_ARGS],
                    env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            # The copy first, then its record: the record is what says finished.
            rclone("copy", str(base / "out"), OUT, "--exclude", "*.json",
                   "--exclude", "faceblur_*/**", "--exclude", "*.part", "--exclude", "*.unchecked")
            rclone("copy", str(base / "out"), OUT, "--include", "*.mp4.json")
            delivered, held = record_names(rel)
            if (base / "out" / held).exists():
                status = "held back"
            elif (base / "out" / delivered).exists():
                status = "delivered"
            else:
                status = "failed"
            tally[status] = tally.get(status, 0) + 1
            say(f"{name}: {rel} {status} (exit {code}) in {time.time() - started:.0f}s")
            record = base / "out" / (held if status == "held back" else delivered)
            if status != "failed" and not on_cuda(record):
                say(f"{name}: STOPPED, {rel} ran on the CPU. The card takes no more work.")
                return
        except Exception as exc:
            tally["failed"] = tally.get("failed", 0) + 1
            say(f"{name}: {rel} failed: {exc}")
        finally:
            src.unlink(missing_ok=True)
            shutil.rmtree(base / "out", ignore_errors=True)
            disk.give(reserved)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    # The recorder's empty stubs (24 bytes) hold no frames and are delivered
    # as they are; leaving them out also keeps the share split the same on a
    # machine that lists the bucket while they are still arriving.
    inputs = {p: s for p, s in listing(IN).items()
              if PurePosixPath(p).suffix.lower() in VIDEO_EXT and s >= 1024}
    outputs = listing(OUT)
    ours = share(inputs)
    todo = [p for p in inputs if p in ours
            and not any(r in outputs for r in record_names(p))]
    gb = sum(inputs[p] for p in todo) / 1e9
    n = gpu_count()
    say(f"{len(inputs)} videos in {IN}, {len(inputs) - len(todo)} finished, "
        f"{len(todo)} to do ({gb:.0f} GB) on {n} cards, {LANES} lanes of {WORKERS} workers each, "
        f"{' '.join(RUN_ARGS)}")
    jobs: "queue.Queue[str]" = queue.Queue()
    for p in sorted(todo, key=lambda p: inputs[p], reverse=True):
        jobs.put(p)
    tally: dict = {}
    cards = [k for k in range(n) if card_works(k)]
    for k in sorted(set(range(n)) - set(cards)):
        say(f"card {k}: CUDA does not work on it, it takes no work")
    if not cards:
        say("no card works with CUDA, stopping")
        return 2
    disk = Disk(WORK)
    say(f"work disk: {disk.room / 1e9:.0f} GB to use at {WORK}")
    threads = [threading.Thread(target=card, args=(k, lane, jobs, tally, inputs, disk), daemon=True)
               for k in cards for lane in range(LANES)]
    started = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        next(t for t in threads if t.is_alive()).join(timeout=600)
        hours = (time.time() - started) / 3600
        finished = sum(tally.values())
        say(f"progress: {finished}/{len(todo)} after {hours:.2f} h, {tally}")
        rclone("copyto", str(LOG), f"{OUT}/_logs/{TAG}stream_run.log", check=False)
    for k in range(n):
        for lane in range(LANES):
            d = lane_dir(k, lane)
            rclone("copyto", str(WORK / d / "cli.log"), f"{OUT}/_logs/{TAG}{d}_cli.log",
                   check=False)
    marker = WORK / "DONE"
    marker.write_text(json.dumps({"tally": tally, "todo": len(todo),
                                  "hours": round((time.time() - started) / 3600, 2)}))
    rclone("copyto", str(marker), f"{OUT}/_logs/{TAG}DONE.json", check=False)
    rclone("copyto", str(LOG), f"{OUT}/_logs/{TAG}stream_run.log", check=False)
    say(f"finished: {tally}")
    return 0 if tally.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
