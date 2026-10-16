"""Patch every held back copy in the bucket, on every card of one machine.

The patch pass the owner approved on 2026-10-16: for each copy the gate held
back, run the check again on (source, copy) and blur every face it still
finds that is not the wearer's hand or inside the handled zone
(`faceblur/patch.py`). Built on `stream_run.py` and run the same way: each
card takes the next copy from a shared queue, largest first, downloads it
and its source, runs `python -m faceblur.patch`, uploads the patched copy
and its record, and deletes the local files, while the next pair downloads.

    python cloud/patch_run.py

Configuration from the environment, as stream_run.py: S3_ENDPOINT,
S3_ACCESS_KEY, S3_SECRET_KEY, BUCKET, IN_PREFIX (default "in"), REPAIRED_PREFIX
(default "in_repaired"), OUT_PREFIX (default "out"), PATCH_PREFIX (default
"patched"), WORK (default /tmp/faceblur_patch), WORKERS (check workers per
card, default 6), HOLD (frames, default 2), MASK (default "face"), GPUS,
SHARD (k/n), LANES (default 1).

The work list is every `OUT/**/quarantine/*_blurred.mp4` that has its record
beside it. Its source is the video under `IN` at the same path without
`quarantine/` and with its own name, or the one under `REPAIRED` when that
exists: a copy made from a repaired source has the repaired source's frames,
and checking it against the original would compare frames one apart at the
end. The patched copy goes to `PATCH/<the copy's path without quarantine/>`,
its record beside it, the copy first; a pair counts as finished once its
record is there, so a restart takes only the rest. A failed patch uploads
nothing, so it is taken again on the next run.
"""
from __future__ import annotations

import json
import os
import queue
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
REPAIRED = f"vs:{BUCKET}/{ENV.get('REPAIRED_PREFIX', 'in_repaired')}"
OUT = f"vs:{BUCKET}/{ENV.get('OUT_PREFIX', 'out')}"
PATCH = f"vs:{BUCKET}/{ENV.get('PATCH_PREFIX', 'patched')}"
WORK = Path(ENV.get("WORK", "/tmp/faceblur_patch"))
WORKERS = ENV.get("WORKERS", "6")
LANES = int(ENV.get("LANES", "1"))
HOLD = ENV.get("HOLD", "2")
MASK = ENV.get("MASK", "face")
SUFFIX = Settings().suffix

# rclone reads a remote called "vs" from these, so no config file holds keys.
ENV.update({"RCLONE_CONFIG_VS_TYPE": "s3", "RCLONE_CONFIG_VS_PROVIDER": "Other",
            "RCLONE_CONFIG_VS_ENDPOINT": ENV["S3_ENDPOINT"],
            "RCLONE_CONFIG_VS_ACCESS_KEY_ID": ENV["S3_ACCESS_KEY"],
            "RCLONE_CONFIG_VS_SECRET_ACCESS_KEY": ENV["S3_SECRET_KEY"]})

log_lock = threading.Lock()
LOG = WORK / "patch_run.log"


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


def held_name(rel: str) -> str:
    """Where a source's held back copy sits in the output, as stream_run names it."""
    p = PurePosixPath(rel)
    return str(p.parent / "quarantine" / output_path(Path(p.name), Path("."), SUFFIX).name)


def patched_name(copy: str) -> str:
    """The patched copy's path: the held back copy's, without `quarantine/`."""
    p = PurePosixPath(copy)
    return str(p.parent.parent / p.name)


def work_list(outputs: dict[str, int], inputs: dict[str, int],
              repaired: dict[str, int]) -> dict[str, tuple[str, str]]:
    """{held back copy: (remote of its source, source path)}, for every copy
    whose record is in the output and whose source can be found.

    Sources are matched through the name the pipeline gave the copy rather
    than by cutting the copy's name back, because the copy's name has lost
    the source's extension (`.mp4` and `.MP4` both become `_blurred.mp4`).
    """
    sources = {held_name(rel): (IN, rel) for rel in inputs
               if PurePosixPath(rel).suffix.lower() in VIDEO_EXT}
    sources.update({held_name(rel): (REPAIRED, rel) for rel in repaired
                    if PurePosixPath(rel).suffix.lower() in VIDEO_EXT})
    todo = {}
    for rel in outputs:
        p = PurePosixPath(rel)
        if ("quarantine" not in p.parts or not p.name.endswith(f"{SUFFIX}.mp4")
                or f"{rel}.json" not in outputs):
            continue
        if rel in sources:
            todo[rel] = sources[rel]
        else:
            say(f"no source found for {rel}; left out")
    return todo


# SHARD=k/n, as in stream_run.py: share k of n, balanced by size over the
# whole list, finished copies included, so every machine computes the same split.
SHARD = ENV.get("SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in SHARD.split("/"))
TAG = f"shard{SHARD_K}of{SHARD_N}_"


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
    """As in stream_run.py: a card onnxruntime cannot use gets no work."""
    r = subprocess.run([sys.executable, "-c", CUDA_CHECK, str(HERE.parent / "models" / "yunet.onnx")],
                       env=dict(ENV, CUDA_VISIBLE_DEVICES=str(k)), capture_output=True)
    return r.returncode == 0


def on_cuda(record: Path) -> bool:
    """The hand models' providers, which is what the patch record carries."""
    try:
        compute = json.loads(record.read_text(encoding="utf-8")).get("compute", {})
    except (OSError, ValueError):
        return True
    return all(v == "CUDAExecutionProvider" for v in compute.values())


class Disk:
    """Room on the work disk, shared by every card, as in stream_run.py.

    A pair in progress holds its source, its copy, the patched copy and the
    segments it is built from: about `NEED` times the copy's size."""

    NEED, SPARE = 4.0, 5e9

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
         todo: dict, disk: Disk) -> None:
    base = WORK / lane_dir(k, lane)
    name = f"card {k}" + (f" lane {lane}" if LANES > 1 else "")
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(k))

    def start_fetch() -> tuple[str, threading.Thread, list] | None:
        try:
            rel = jobs.get_nowait()
        except queue.Empty:
            return None
        err: list = []

        def run():
            # The room is taken here, in the download thread, so the pair
            # being worked on now never waits for it.
            err.append(disk.take(sizes[rel]))
            try:
                remote, src = todo[rel]
                (base / "src").mkdir(parents=True, exist_ok=True)
                rclone("copyto", f"{remote}/{src}", str(base / "src" / src),
                       "--multi-thread-streams", "8")
                rclone("copyto", f"{OUT}/{rel}", str(base / "copy" / rel),
                       "--multi-thread-streams", "8")
            except Exception as exc:          # reported when the pair is reached
                err.append(exc)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return rel, t, err

    nxt = start_fetch()
    while nxt is not None:
        rel, t, err = nxt
        t.join()
        nxt = start_fetch()                   # the next pair downloads while this one runs
        src = base / "src" / todo[rel][1]
        copy = base / "copy" / rel
        target = patched_name(rel)
        out = base / "out" / target
        started = time.time()
        reserved, err = err[0], err[1:]
        try:
            if err:
                raise err[0]
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(base / "patch.log", "a", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%F %T')} {rel}\n")
                log.flush()
                # As a module from the checkout's root, the way the image's
                # WORKDIR /app runs cli.py, so `faceblur` imports from there.
                code = subprocess.run(
                    [sys.executable, "-m", "faceblur.patch", str(src), str(copy), str(out),
                     "--workers", WORKERS, "--hold", HOLD, "--mask", MASK],
                    cwd=str(HERE.parent), env=env, stdout=log,
                    stderr=subprocess.STDOUT).returncode
            record = Path(str(out) + ".json")
            if code != 0 or not out.is_file():
                reason = ""
                try:
                    reason = json.loads(record.read_text(encoding="utf-8")).get("error", "")
                except (OSError, ValueError):
                    pass
                raise RuntimeError(f"patch exit {code}: {reason[:300]}")
            # The copy first, then its record: the record is what says finished.
            rclone("copyto", str(out), f"{PATCH}/{target}")
            rclone("copyto", str(record), f"{PATCH}/{target}.json")
            masked = json.loads(record.read_text(encoding="utf-8")).get("masked", 0)
            status = "patched" if masked else "nothing to mask"
            tally[status] = tally.get(status, 0) + 1
            tally["rows masked"] = tally.get("rows masked", 0) + masked
            say(f"{name}: {rel} {status} ({masked} rows) in {time.time() - started:.0f}s")
            if not on_cuda(record):
                say(f"{name}: STOPPED, {rel} ran on the CPU. The card takes no more work.")
                return
        except Exception as exc:
            tally["failed"] = tally.get("failed", 0) + 1
            say(f"{name}: {rel} failed: {exc}")
        finally:
            for d in ("src", "copy", "out"):
                shutil.rmtree(base / d, ignore_errors=True)
            disk.give(reserved)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    outputs = listing(OUT)
    todo = work_list(outputs, listing(IN), listing(REPAIRED))
    sizes = {rel: outputs[rel] for rel in todo}
    finished = listing(PATCH)
    ours = share(sizes)
    left = [rel for rel in todo if rel in ours
            and f"{patched_name(rel)}.json" not in finished]
    gb = sum(sizes[p] for p in left) / 1e9
    n = gpu_count()
    say(f"{len(todo)} held back copies in {OUT}, {len(ours)} in share {SHARD}, "
        f"{len(ours) - len(left)} patched, {len(left)} to do ({gb:.0f} GB) on {n} cards, "
        f"{LANES} lanes of {WORKERS} workers each, hold {HOLD}, mask {MASK}")
    jobs: "queue.Queue[str]" = queue.Queue()
    for p in sorted(left, key=lambda p: sizes[p], reverse=True):
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
    threads = [threading.Thread(target=card, args=(k, lane, jobs, tally, sizes, todo, disk),
                                daemon=True)
               for k in cards for lane in range(LANES)]
    started = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        next(t for t in threads if t.is_alive()).join(timeout=600)
        hours = (time.time() - started) / 3600
        done = sum(v for key, v in tally.items() if key != "rows masked")
        say(f"progress: {done}/{len(left)} after {hours:.2f} h, {tally}")
        rclone("copyto", str(LOG), f"{PATCH}/_logs/{TAG}patch_run.log", check=False)
    for k in range(n):
        for lane in range(LANES):
            d = lane_dir(k, lane)
            rclone("copyto", str(WORK / d / "patch.log"), f"{PATCH}/_logs/{TAG}{d}_patch.log",
                   check=False)
    marker = WORK / "DONE"
    marker.write_text(json.dumps({"tally": tally, "todo": len(left), "share": len(ours),
                                  "hours": round((time.time() - started) / 3600, 2)}))
    rclone("copyto", str(marker), f"{PATCH}/_logs/{TAG}DONE.json", check=False)
    rclone("copyto", str(LOG), f"{PATCH}/_logs/{TAG}patch_run.log", check=False)
    say(f"finished: {tally}")
    return 0 if tally.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
