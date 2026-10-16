"""Blur a whole bucket of videos on every card of one machine, a file at a time.

The machine's disk is far smaller than the footage, so nothing is copied in
bulk. Each card takes the next video from a shared queue, largest first,
downloads it, runs `cli.py` on it, uploads the copy and its record, and
deletes both local files, while the video after it downloads in the
background. Disk use stays at a few videos per card.

    python cloud/stream_run.py

Configuration is read from the environment when main() starts (see
cloud/run.env.example): S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, BUCKET,
REMOTE (the rclone root, default vs:BUCKET; a plain directory path works as
a local remote), IN_PREFIX (default "in"), OUT_PREFIX (default "out"), WORK
(local scratch, default /tmp/faceblur), WORKERS (pipeline workers per card),
RUN_ARGS (extra cli.py arguments), GPUS (default: every card nvidia-smi
lists), SHARD (k/n: this machine's share when n machines run at once,
default 0/1), LANES (videos worked on at once per card, default 1), RCLONE
(the rclone binary), PIPELINE_CMD (what runs on each video, default cli.py),
CUDA_CHECK ("0" skips the per-card CUDA probe), and QUEUE, LEASE,
SELF_DELETE, GN_SERVER_NAME for queue mode.

Two lanes keep a card busy: a video spends about half its time in steps that
use one core and no GPU (the tracker, the zone, the encode), and on the RTX
5090 VMs the card sat idle through them.

A video counts as finished once its record is in the bucket, delivered or in
quarantine; the copy is uploaded before the record, so a record never stands
beside a missing copy. Run it again after anything stops it and only the
unfinished videos are taken.

QUEUE=1 lets any number of machines work from one bucket with no fixed
split. A machine takes the largest unfinished video nobody holds and leaves
a claim at OUT/_claims/<video>.json naming itself and the time; it rewrites
the claim every LEASE/3 seconds while it works and deletes it when the video
is finished. A claim older than LEASE seconds (default 1800) belongs to a
machine that died, and the video is taken again. A machine with nothing left
to claim leaves at once instead of waiting for other machines' claims to
run out; a video whose machine died after the others left is taken by the
next start of this script. SHARD is ignored in queue mode. CLAIM_SETTLE
(seconds, default 2) is how long a claim is left before it is read back; it
exists for the tests and needs no setting in production.

SELF_DELETE=1 deletes this machine, with its volumes, through the vServer
API once the run is over and its DONE marker and log are in the bucket, even
when some videos failed: the failures are in the log, and the machine holds
nothing the bucket does not. GN_SERVER_NAME names the machine (it must fall
within GN_NAME_PREFIX, default faceblur-), and GN_CLIENT_ID,
GN_CLIENT_SECRET and GN_PROJECT_ID must be in the container's environment.
vm_bootstrap.sh no longer copies the API keys onto a VM, so whether they go
into /root/faceblur.env is the owner's choice; without them the machine
stays and says why.
"""
from __future__ import annotations

import importlib.util
import json
import os
import queue
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from faceblur.pipeline import output_path  # noqa: E402
from faceblur.settings import VIDEO_EXT, Settings  # noqa: E402

SUFFIX = Settings().suffix


def on(value: str) -> bool:
    return value not in ("", "0")


def new_owner() -> str:
    """Names this process in its claims, so two runs on one host differ."""
    return f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class Config:
    """Everything the run reads from its environment, read once in main().

    Reading it at import time made the module impossible to load in a test;
    with it here a test can point the run at a local directory and a fake
    pipeline, and production runs exactly as before."""

    remote: str
    in_prefix: str = "in"
    out_prefix: str = "out"
    work: Path = Path("/tmp/faceblur")
    workers: str = "6"
    lanes: int = 1
    run_args: list = field(default_factory=lambda: ["--quarantine"])
    gpus: int | None = None
    shard_k: int = 0
    shard_n: int = 1
    rclone: str = "rclone"
    pipeline_cmd: list = field(default_factory=lambda: [sys.executable, str(HERE.parent / "cli.py")])
    cuda_check: bool = True
    # Queue mode.
    queue: bool = False
    lease: int = 1800
    self_delete: bool = False
    gn_server_name: str = ""
    claim_settle: float = 2.0
    owner: str = ""                           # set by main() in queue mode

    @property
    def inp(self) -> str:
        return f"{self.remote}/{self.in_prefix}"

    @property
    def out(self) -> str:
        return f"{self.remote}/{self.out_prefix}"

    @property
    def tag(self) -> str:
        # Every machine of a queue run writes its own logs beside the others'.
        if self.queue and self.owner:
            return f"q-{self.owner}_"
        return f"shard{self.shard_k}of{self.shard_n}_" if self.shard_n > 1 else ""

    @property
    def log(self) -> Path:
        return self.work / "stream_run.log"


def config_from_env(env: Mapping[str, str] = os.environ) -> Config:
    if not env.get("REMOTE") and not env.get("BUCKET"):
        raise SystemExit("stream_run: set BUCKET (the vStorage bucket) or REMOTE (an rclone root)")
    # SHARD=k/n: this machine takes share k of n; see share().
    k, n = (int(x) for x in env.get("SHARD", "0/1").split("/"))
    extra = {}
    if env.get("PIPELINE_CMD"):
        extra["pipeline_cmd"] = shlex.split(env["PIPELINE_CMD"])
    return Config(
        remote=env.get("REMOTE") or f"vs:{env['BUCKET']}",
        in_prefix=env.get("IN_PREFIX", "in"),
        out_prefix=env.get("OUT_PREFIX", "out"),
        work=Path(env.get("WORK", "/tmp/faceblur")),
        workers=env.get("WORKERS", "6"),
        lanes=int(env.get("LANES", "1")),
        run_args=shlex.split(env.get("RUN_ARGS", "--quarantine")),
        gpus=int(env["GPUS"]) if env.get("GPUS") else None,
        shard_k=k, shard_n=n,
        rclone=env.get("RCLONE") or "rclone",
        cuda_check=env.get("CUDA_CHECK", "1") != "0",
        queue=on(env.get("QUEUE", "0")),
        lease=int(env.get("LEASE", "1800")),
        self_delete=on(env.get("SELF_DELETE", "0")),
        gn_server_name=env.get("GN_SERVER_NAME", ""),
        claim_settle=float(env.get("CLAIM_SETTLE", "2.0")),
        **extra)


def vs_remote(env=os.environ) -> None:
    """rclone reads a remote called "vs" from these, so no config file holds
    keys. A local directory remote needs none of it, so nothing is set
    without S3_ENDPOINT."""
    if "S3_ENDPOINT" in env:
        env.update({"RCLONE_CONFIG_VS_TYPE": "s3", "RCLONE_CONFIG_VS_PROVIDER": "Other",
                    "RCLONE_CONFIG_VS_ENDPOINT": env["S3_ENDPOINT"],
                    "RCLONE_CONFIG_VS_ACCESS_KEY_ID": env["S3_ACCESS_KEY"],
                    "RCLONE_CONFIG_VS_SECRET_ACCESS_KEY": env["S3_SECRET_KEY"]})


log_lock = threading.Lock()


def say(cfg: Config, text: str) -> None:
    line = f"[{time.strftime('%F %T')}] {text}"
    with log_lock:
        print(line, flush=True)
        with open(cfg.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def rclone(cfg: Config, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run([cfg.rclone, *args, "--retries", "5", "--low-level-retries", "20"],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])}: {r.stderr.strip()[-500:]}")
    return r


def listing(cfg: Config, remote: str) -> dict[str, int]:
    r = rclone(cfg, "lsjson", remote, "--recursive", "--files-only", check=False)
    if r.returncode != 0:
        return {}
    return {e["Path"]: e["Size"] for e in json.loads(r.stdout or "[]")}


def videos(cfg: Config) -> dict[str, int]:
    # The recorder's empty stubs (24 bytes) hold no frames and are delivered
    # as they are; leaving them out also keeps the share split the same on a
    # machine that lists the bucket while they are still arriving.
    return {p: s for p, s in listing(cfg, cfg.inp).items()
            if PurePosixPath(p).suffix.lower() in VIDEO_EXT and s >= 1024}


def record_names(rel: str) -> tuple[str, str]:
    """Where a video's record lands in the output, delivered and held back."""
    p = PurePosixPath(rel)
    copy = output_path(Path(p.name), Path("."), SUFFIX).name
    return (str(p.parent / f"{copy}.json"), str(p.parent / "quarantine" / f"{copy}.json"))


# SHARD=k/n: this machine takes share k of n, so several machines can work
# from one bucket without taking the same video. The shares are balanced by
# size over the whole listing, finished videos included, so every machine and
# every restart computes the same split.
def share(sizes: dict[str, int], k_ours: int, n: int) -> set[str]:
    """Largest first, each to the share with the least so far; ties by path."""
    loads = [0] * n
    taken: set[str] = set()
    for rel in sorted(sizes, key=lambda r: (-sizes[r], r)):
        k = loads.index(min(loads))
        loads[k] += sizes[rel]
        if k == k_ours:
            taken.add(rel)
    return taken


# QUEUE=1: claims instead of shares. rclone cannot create a file only if it
# is absent on S3, so a claim is written, left to settle, and read back; the
# last writer is the owner. Two machines both win only if one stalls longer
# than the settle time between its check and its write, and then one video
# is blurred twice into the same copy, which costs time but loses nothing.
def claim_path(cfg: Config, rel: str) -> str:
    return f"{cfg.out}/_claims/{rel}.json"


def read_claim(cfg: Config, rel: str) -> dict | None:
    r = rclone(cfg, "cat", claim_path(cfg, rel), check=False)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:                        # half written: as good as none
        return None


def write_claim(cfg: Config, rel: str, owner: str) -> None:
    # A failed write is not raised: the read-back finds the claim is not
    # ours, and a renewal is tried again in LEASE/3 seconds.
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"owner": owner, "time": time.time(), "lease": cfg.lease}, f)
    try:
        rclone(cfg, "copyto", tmp, claim_path(cfg, rel), check=False)
    finally:
        os.unlink(tmp)


def claim(cfg: Config, rel: str, owner: str) -> bool:
    held = read_claim(cfg, rel)
    if (held and held.get("owner") != owner
            and time.time() - held.get("time", 0) < held.get("lease", cfg.lease)):
        return False
    write_claim(cfg, rel, owner)
    time.sleep(cfg.claim_settle)
    held = read_claim(cfg, rel)
    return bool(held) and held.get("owner") == owner


def claim_times(cfg: Config) -> dict[str, float]:
    """When each claim was last written, from one listing, so a live claim
    is skipped without reading it."""
    r = rclone(cfg, "lsjson", f"{cfg.out}/_claims", "--recursive", "--files-only", check=False)
    if r.returncode != 0:
        return {}
    return {e["Path"]: datetime.fromisoformat(e["ModTime"]).timestamp()
            for e in json.loads(r.stdout or "[]")}


def next_claim(cfg: Config, owner: str, ours: set[str]) -> tuple[str, int, float] | None:
    """The largest unfinished video this machine could claim, claimed, with
    its size and the monotonic time just before its claim was written, or
    None when there is none. Everything is listed afresh because the other
    machines change it all the time. `ours` are the videos this process
    has handed out already, held or finished: claim() would hand one of them
    out a second time."""
    inputs = videos(cfg)
    outputs = listing(cfg, cfg.out)
    now = time.time()
    held = claim_times(cfg)
    for rel in sorted(inputs, key=lambda r: (-inputs[r], r)):
        if rel in ours or any(r in outputs for r in record_names(rel)):
            continue
        if now - held.get(f"{rel}.json", 0) < cfg.lease:
            continue
        written = time.monotonic()
        if claim(cfg, rel, owner):
            return rel, inputs[rel], written
    return None


class Jobs(queue.Queue):
    """SHARD mode's job source: the videos are fixed at the start and
    nothing needs doing when one is finished."""

    def done(self, rel: str) -> None:
        pass


class ClaimQueue:
    """Queue mode's job source, with the surface card() uses from Jobs.

    Each video handed out is claimed first and its claim renewed in the
    background until done(), so other machines leave it alone while it is
    fetched and run. done() deletes the claim: the record in OUT is what
    says finished, so a claim left behind by a crash does no harm.

    A video is handed out once per process, so one that fails here is not
    retried in a loop, while another machine may still try it once, in case
    the failure was this machine's own (a CPU fallback, a full disk)."""

    def __init__(self, cfg: Config, sizes: dict):
        self.cfg, self.sizes = cfg, sizes
        self.held: dict[str, tuple[threading.Event, threading.Thread]] = {}
        self.given: set[str] = set()
        self.lock = threading.Lock()
        # Every claim of one process names the same owner, so two cards
        # claiming at once would both read their claim back as theirs.
        # They take turns.
        self.claiming = threading.Lock()

    def get_nowait(self) -> str:
        with self.claiming:
            with self.lock:
                ours = set(self.given)
            got = next_claim(self.cfg, self.cfg.owner, ours)
            if got is None:
                raise queue.Empty
            rel, size, due = got
            stop = threading.Event()

            def renew():
                # On a fixed beat from the claim's own write, so neither the
                # settle nor the time a write takes stretches the gap between
                # two writes past LEASE/3.
                nonlocal due
                while True:
                    due += self.cfg.lease / 3
                    if stop.wait(max(0.0, due - time.monotonic())):
                        return
                    write_claim(self.cfg, rel, self.cfg.owner)
            t = threading.Thread(target=renew, daemon=True)
            with self.lock:
                self.sizes[rel] = size
                self.held[rel] = (stop, t)
                self.given.add(rel)
            t.start()
        return rel

    def done(self, rel: str) -> None:
        with self.lock:
            stop, t = self.held.pop(rel)
        stop.set()
        t.join()                              # a renewal in flight would bring the claim back
        rclone(self.cfg, "deletefile", claim_path(self.cfg, rel), check=False)


def gpu_count(cfg: Config) -> int:
    if cfg.gpus:
        return cfg.gpus
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return len(r.stdout.split())


CUDA_CHECK = ("import onnxruntime as o, sys; "
              "s = o.InferenceSession(sys.argv[1], providers=['CUDAExecutionProvider']); "
              "sys.exit(0 if s.get_providers()[0] == 'CUDAExecutionProvider' else 1)")


def card_works(cfg: Config, k: int) -> bool:
    """onnxruntime falls back to the CPU on a card it cannot use and says so
    only in its log. A card on the CPU runs some fifty times slower and takes
    cores from every other card, so it gets no work at all. A test with a fake
    pipeline has no card to probe and turns the check off."""
    if not cfg.cuda_check:
        return True
    r = subprocess.run([sys.executable, "-c", CUDA_CHECK, str(HERE.parent / "models" / "yunet.onnx")],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k)), capture_output=True)
    return r.returncode == 0


def on_cuda(record: Path) -> bool:
    try:
        compute = json.loads(record.read_text(encoding="utf-8")).get("compute", {})
    except (OSError, ValueError):
        return True
    return all(v == "CUDAExecutionProvider" for v in compute.values())


def fetch(cfg: Config, rel: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rclone(cfg, "copyto", f"{cfg.inp}/{rel}", str(dest), "--multi-thread-streams", "8")


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


def card(cfg: Config, k: int, lane: int, jobs: Jobs | ClaimQueue, tally: dict, sizes: dict,
         disk: Disk) -> None:
    base = cfg.work / lane_dir(k, lane)
    name = f"card {k}" + (f" lane {lane}" if cfg.lanes > 1 else "")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
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
                fetch(cfg, rel, base / "in" / rel)
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
                    [*cfg.pipeline_cmd, str(src), "-o", str(out_dir),
                     "--no-progress", "--workers", cfg.workers, *cfg.run_args],
                    env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            # The copy first, then its record: the record is what says finished.
            rclone(cfg, "copy", str(base / "out"), cfg.out, "--exclude", "*.json",
                   "--exclude", "faceblur_*/**", "--exclude", "*.part", "--exclude", "*.unchecked")
            rclone(cfg, "copy", str(base / "out"), cfg.out, "--include", "*.mp4.json")
            delivered, held = record_names(rel)
            if (base / "out" / held).exists():
                status = "held back"
            elif (base / "out" / delivered).exists():
                status = "delivered"
            else:
                status = "failed"
            tally[status] = tally.get(status, 0) + 1
            say(cfg, f"{name}: {rel} {status} (exit {code}) in {time.time() - started:.0f}s")
            record = base / "out" / (held if status == "held back" else delivered)
            if status != "failed" and not on_cuda(record):
                say(cfg, f"{name}: STOPPED, {rel} ran on the CPU. The card takes no more work.")
                return
        except Exception as exc:
            tally["failed"] = tally.get("failed", 0) + 1
            say(cfg, f"{name}: {rel} failed: {exc}")
        finally:
            src.unlink(missing_ok=True)
            shutil.rmtree(base / "out", ignore_errors=True)
            disk.give(reserved)
            jobs.done(rel)


def self_delete(cfg: Config, env: Mapping[str, str] = os.environ) -> None:
    """Delete this machine through the vServer API. Nothing here is raised:
    the run is over and its results are in the bucket whatever happens, so a
    failure only means the machine stays, and the log says why."""
    name = cfg.gn_server_name
    if not name:
        say(cfg, "self-delete: GN_SERVER_NAME is not set, the machine stays")
        return
    e = dict(env)
    # Checked here because the token request would be the first thing
    # delete_server sends, and it should not go out with no keys.
    if not all(e.get(k) for k in ("GN_CLIENT_ID", "GN_CLIENT_SECRET", "GN_PROJECT_ID")):
        say(cfg, "self-delete: GN_CLIENT_ID/GN_CLIENT_SECRET/GN_PROJECT_ID missing, the machine stays")
        return
    try:
        # Loaded by path and only here: the script runs from cloud/, which is
        # not a package, and a run without SELF_DELETE never touches it.
        spec = importlib.util.spec_from_file_location("gn_api", HERE / "gn_api.py")
        gn_api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gn_api)
    except Exception as exc:
        say(cfg, f"self-delete: cannot load gn_api.py ({exc!r}), the machine stays")
        return
    say(cfg, f"self-delete: deleting this machine, {name}, through the vServer API")
    try:
        sid = gn_api.delete_server(name, e)
    except (ValueError, LookupError, gn_api.ApiError) as exc:
        say(cfg, f"self-delete: {exc}; the machine stays")
    except Exception as exc:
        say(cfg, f"self-delete: failed with {exc!r}; the machine stays")
    else:
        say(cfg, f"self-delete: the API accepted the delete of {name} ({sid})")


def main(argv=None) -> int:
    cfg = config_from_env()
    if cfg.queue:
        cfg = replace(cfg, owner=new_owner())
    vs_remote()
    cfg.work.mkdir(parents=True, exist_ok=True)
    inputs = videos(cfg)
    outputs = listing(cfg, cfg.out)
    # In queue mode the count is only what was unfinished at the start;
    # the other machines take from it too.
    ours = set(inputs) if cfg.queue else share(inputs, cfg.shard_k, cfg.shard_n)
    todo = [p for p in inputs if p in ours
            and not any(r in outputs for r in record_names(p))]
    gb = sum(inputs[p] for p in todo) / 1e9
    n = gpu_count(cfg)
    if cfg.queue:
        mode = f"queue mode as {cfg.owner}, lease {cfg.lease}s"
    else:
        mode = f"shard {cfg.shard_k}/{cfg.shard_n}"
    say(cfg, f"{mode}: {len(inputs)} videos in {cfg.inp}, {len(inputs) - len(todo)} finished, "
        f"{len(todo)} to do ({gb:.0f} GB) on {n} cards, {cfg.lanes} lanes of {cfg.workers} workers each, "
        f"{' '.join(cfg.run_args)}")
    if cfg.queue:
        jobs: Jobs | ClaimQueue = ClaimQueue(cfg, inputs)
    else:
        jobs = Jobs()
        for p in sorted(todo, key=lambda p: inputs[p], reverse=True):
            jobs.put(p)
    tally: dict = {}
    cards = [k for k in range(n) if card_works(cfg, k)]
    for k in sorted(set(range(n)) - set(cards)):
        say(cfg, f"card {k}: CUDA does not work on it, it takes no work")
    if not cards:
        say(cfg, "no card works with CUDA, stopping")
        return 2
    disk = Disk(cfg.work)
    say(cfg, f"work disk: {disk.room / 1e9:.0f} GB to use at {cfg.work}")
    threads = [threading.Thread(target=card, args=(cfg, k, lane, jobs, tally, inputs, disk), daemon=True)
               for k in cards for lane in range(cfg.lanes)]
    started = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        next(t for t in threads if t.is_alive()).join(timeout=600)
        hours = (time.time() - started) / 3600
        finished = sum(tally.values())
        say(cfg, f"progress: {finished}/{len(todo)} after {hours:.2f} h, {tally}")
        rclone(cfg, "copyto", str(cfg.log), f"{cfg.out}/_logs/{cfg.tag}stream_run.log", check=False)
    for k in range(n):
        for lane in range(cfg.lanes):
            d = lane_dir(k, lane)
            rclone(cfg, "copyto", str(cfg.work / d / "cli.log"), f"{cfg.out}/_logs/{cfg.tag}{d}_cli.log",
                   check=False)
    marker = cfg.work / "DONE"
    marker.write_text(json.dumps({"tally": tally, "todo": len(todo),
                                  "hours": round((time.time() - started) / 3600, 2)}))
    rclone(cfg, "copyto", str(marker), f"{cfg.out}/_logs/{cfg.tag}DONE.json", check=False)
    rclone(cfg, "copyto", str(cfg.log), f"{cfg.out}/_logs/{cfg.tag}stream_run.log", check=False)
    say(cfg, f"finished: {tally}")
    # Last of all, so the DONE marker and the log are in the bucket before
    # the machine that wrote them goes.
    if cfg.self_delete:
        self_delete(cfg)
        # Up again so the hook's own lines reach the bucket: if the machine
        # stays, the reason can be read there without logging in to it.
        rclone(cfg, "copyto", str(cfg.log), f"{cfg.out}/_logs/{cfg.tag}stream_run.log", check=False)
    return 0 if tally.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
