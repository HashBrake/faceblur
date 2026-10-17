"""Put the source's frame timestamps back on delivered copies, without re-encoding.

Copies written before `faceblur/timestamps.py` existed hold every frame of
their source, frame n for frame n, but on the encoder's constant rate, so
each frame is a few milliseconds off the camera's own time and the step at
every dropped frame is lost. This remuxes each copy's packets, unchanged,
with the source's pts and dts (`timestamps.retime`), then checks the result:
the same frame count, every pts equal to the source's in the source's time
base, the same video payload bytes as the copy, no audio, and a clean decode.

It refuses a copy whose frame count differs from its source's, and a copy
whose decode order is not its presentation order (B-frames): the pipeline's
H.264 copies have none, and that is checked per file, not assumed.

One file, locally:

    python cloud/retime.py one SOURCE COPY OUTPUT

The delivered tree, in the bucket (inside the faceblur image, rclone remote
`vs` configured from the S3_* variables):

    python cloud/retime.py tree --work /scratch/retime --workers 8 [--limit N] [--dry-run]

For each video in `vs:faceblur-blurred/` (stubs under 1 KiB skipped): the
copy and its source (`vs:faceblur/in_repaired/<path>` where that exists,
else `vs:faceblur/in/<path>`) are downloaded, the copy is retimed and
checked, uploaded over the delivered one with a private ACL, the local files
are deleted, and a record goes to `vs:faceblur/faceblur-records/retime/<path>.json`.
A ledger (`<work>/ledger.jsonl`) makes the run resumable: a video already
retimed or found already right is not done again.

While the delivered tree is being rewritten by `cloud/deliver.py`, the same
work can be done ahead of it without touching that tree:

    python cloud/retime.py tree --stage ...     # copies from the work bucket
    python cloud/retime.py publish --work ...   # once deliver.py has finished

`--stage` takes each video's copy from where `deliver.py --patched` takes it
(`vs:faceblur/patched/` or `vs:faceblur/out/`, chosen by `deliver.pick`) and
uploads the result to `vs:faceblur/retimed/<path>`. `publish` then puts each
staged file over the delivered one with a server side copy, but only where
the delivered object is exactly the size of the copy that was retimed;
anything else is left alone and listed for the direct mode.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from faceblur import __version__  # noqa: E402
from faceblur.timestamps import (TimingError, copy_layout, decode_check,  # noqa: E402
                                 retime, same_timing, source_timing, _max_error_ms)

DELIVERED = "vs:faceblur-blurred"
WORK_BUCKET = "vs:faceblur"
STAGED = "vs:faceblur/retimed"
SOURCES = "vs:faceblur/in"
REPAIRED = "vs:faceblur/in_repaired"
RECORDS = "vs:faceblur/faceblur-records/retime"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
EMPTY = 1024                      # bytes; the recorder's stubs are 24
DONE = ("retimed", "already")


def retime_file(source: Path, copy: Path, out: Path) -> dict:
    """Retime one copy into `out` and check it. Returns the record.

    status: "retimed" (out written and checked), "already" (the copy already
    carries the source's timestamps; nothing written), "refused" (frame
    counts differ, or the copy reorders frames), "failed" (anything else)."""
    t0 = time.time()
    rec = {"source": Path(source).name, "copy": Path(copy).name, "status": "failed",
           "frames": 0, "source_frames": 0, "time_base": "", "max_error_before_ms": None,
           "max_error_after_ms": None, "packets_sha256": "", "error": "",
           "faceblur_version": __version__}
    try:
        timing = source_timing(source)
        rec["source_frames"] = len(timing)
        rec["time_base"] = f"{timing.time_base.numerator}/{timing.time_base.denominator}"
        layout = copy_layout(copy)
        rec["frames"] = layout.frames
        if layout.frames != len(timing):
            rec["status"] = "refused"
            rec["error"] = (f"frame counts differ: copy {layout.frames}, "
                            f"source {len(timing)}")
            return rec
        if layout.reorder:
            rec["status"] = "refused"
            rec["error"] = (f"decode order is not presentation order "
                            f"(reorder delay {layout.reorder})")
            return rec
        if same_timing(copy, timing):
            rec.update(status="already", max_error_before_ms=0.0, max_error_after_ms=0.0)
            return rec
        res = retime(copy, source, out, allow_reorder=False, timing=timing)
        problem = decode_check(out, len(timing))
        if problem:
            out.unlink(missing_ok=True)
            raise TimingError(problem)
        from faceblur.video import has_audio
        if has_audio(out):
            out.unlink(missing_ok=True)
            raise TimingError("the retimed file has an audio stream")
        rec.update(status="retimed", max_error_before_ms=res.max_error_before_ms,
                   max_error_after_ms=res.max_error_after_ms,
                   packets_sha256=res.packets_sha256, bytes=out.stat().st_size)
    except TimingError as exc:
        msg = str(exc)
        rec["status"] = "refused" if ("differ" in msg or "presentation order" in msg) else "failed"
        rec["error"] = msg
        Path(out).unlink(missing_ok=True)
    except Exception as exc:          # one bad file must not stop the tree
        rec["error"] = f"{type(exc).__name__}: {exc}"
        Path(out).unlink(missing_ok=True)
    finally:
        rec["seconds"] = round(time.time() - t0, 2)
    return rec


# ------------------------------------------------------------------ the tree

def rclone(*args: str, stdin: str | None = None, timeout: int = 7200) -> str:
    r = subprocess.run(["rclone", *args], input=stdin, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rclone {' '.join(args[:2])}: {r.stderr.strip()[-400:]}")
    return r.stdout


def listing(remote: str) -> dict[str, int]:
    try:
        out = rclone("lsjson", remote, "--recursive", "--files-only")
    except RuntimeError:
        return {}
    return {e["Path"]: e["Size"] for e in json.loads(out or "[]")}


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("status") in DONE:
                    self.done[row["path"]] = row

    def add(self, row: dict) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if row.get("status") in DONE:
                self.done[row["path"]] = row


_room = threading.Condition()
_reserved = [0]


def _reserve(work: Path, need: int) -> None:
    """Wait until the disk has room for this job's three files (the copy,
    its source and the result) beside the jobs already running."""
    with _room:
        while True:
            free = shutil.disk_usage(work).free - _reserved[0]
            if free > need + 10 * 2**30 or _reserved[0] == 0:
                _reserved[0] += need
                return
            _room.wait(timeout=30)


def _release(need: int) -> None:
    with _room:
        _reserved[0] -= need
        _room.notify_all()


def do_one(rel: str, size: int, repaired: set, work: Path, dry_run: bool,
           copy_remote: str | None = None) -> dict:
    """One video. `copy_remote` None: the delivered copy, and the result goes
    over it. Given: that copy (`--stage`), and the result goes to STAGED."""
    need = int(size * 3.3)
    _reserve(work, need)
    d = work / f"job_{abs(hash(rel)) % 10**10}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.time()
    src_remote = f"{REPAIRED}/{rel}" if rel in repaired else f"{SOURCES}/{rel}"
    stage = copy_remote is not None
    copy_remote = copy_remote or f"{DELIVERED}/{rel}"
    target = f"{STAGED}/{rel}" if stage else f"{DELIVERED}/{rel}"
    row = {"path": rel, "source": src_remote, "copy_remote": copy_remote,
           "delivered_bytes": size, "target": target}
    try:
        name = PurePosixPath(rel).name
        copy, src, out = d / f"copy_{name}", d / f"src_{name}", d / f"out_{name}"
        rclone("copyto", copy_remote, str(copy))
        if copy.stat().st_size != size:
            raise RuntimeError(f"downloaded {copy.stat().st_size} bytes, listed {size}")
        src_size = next(iter(listing(src_remote).values()), None)
        if src_size is None:
            raise RuntimeError(f"no source at {src_remote}")
        if src_size == size:
            raise RuntimeError("the delivered file is the size of its source")
        # The whole source: the camera writes fragmented MP4, whose timestamps
        # are spread over a box per fragment through the file.
        rclone("copyto", src_remote, str(src))
        rec = retime_file(src, copy, out)
        rec["path"] = rel
        rec["source_remote"] = src_remote
        if rec["status"] == "retimed" and not dry_run:
            rclone("copyto", str(out), target, "--s3-acl", "private")
            back = listing(target)
            got = next(iter(back.values()), None)
            if got != out.stat().st_size:
                rec["status"] = "failed"
                rec["error"] = f"upload: listed {got} bytes, wrote {out.stat().st_size}"
            else:
                rec["uploaded"] = True
        if rec["status"] == "retimed" and dry_run:
            rec["status"] = "dry_run"
        if not dry_run:
            rclone("rcat", f"{RECORDS}/{rel}.json", stdin=json.dumps(rec, indent=1))
        row.update({k: rec.get(k) for k in ("status", "frames", "source_frames", "time_base",
                                            "uploaded",
                                            "max_error_before_ms", "max_error_after_ms",
                                            "error", "bytes")})
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        shutil.rmtree(d, ignore_errors=True)       # sources never stay on the machine
        _release(need)
        row["seconds"] = round(time.time() - t0, 1)
    return row


def staged_copies() -> dict[str, tuple[str, int]]:
    """Each source video's copy, as `deliver.py --patched` picks it:
    {source path: (copy remote, copy size)}. Stubs and videos without a
    finished copy are left out."""
    sys.path.insert(0, str(HERE))
    from deliver import pick
    from faceblur.settings import Settings
    sources = listing(SOURCES)
    out, patched = listing(f"{WORK_BUCKET}/out"), listing(f"{WORK_BUCKET}/patched")
    suffix = Settings().suffix
    got = {}
    for rel, size in sources.items():
        if PurePosixPath(rel).suffix.lower() not in VIDEO_EXT or size < EMPTY:
            continue
        p = pick(rel, out, patched, suffix, True)
        if p is not None:
            got[rel] = (f"{WORK_BUCKET}/{p['video']}", p["size"])
    return got


def tree(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(work / "ledger.jsonl")
    if args.stage:
        chosen = staged_copies()
        videos = {rel: size for rel, (_, size) in chosen.items()}
        remotes = {rel: remote for rel, (remote, _) in chosen.items()}
        stubs: list = []
    else:
        delivered = listing(DELIVERED)
        if not delivered:
            print("nothing listed in", DELIVERED)
            return 1
        videos = {rel: size for rel, size in delivered.items()
                  if PurePosixPath(rel).suffix.lower() in VIDEO_EXT}
        stubs = sorted(rel for rel, size in videos.items() if size < EMPTY)
        remotes = {}
    todo = sorted(rel for rel, size in videos.items() if size >= EMPTY)
    repaired = set(listing(REPAIRED))
    skip = [rel for rel in todo if rel in ledger.done
            and ledger.done[rel].get("delivered_bytes") == videos[rel]]
    todo = [rel for rel in todo if rel not in skip]
    if args.limit:
        todo = todo[:args.limit]
    print(f"videos {len(videos)}; stubs skipped {len(stubs)}; done before (ledger) {len(skip)}; "
          f"repaired sources {len(repaired)}; to do now {len(todo)}"
          + (" (staged)" if args.stage else ""), flush=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_one, rel, videos[rel], repaired, work, args.dry_run,
                               remotes.get(rel)): rel
                   for rel in todo}
        for fut in as_completed(futures):
            row = fut.result()
            ledger.add(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"{row['status']:8s} {row['path']} frames={row.get('frames')} "
                  f"before={row.get('max_error_before_ms')} after={row.get('max_error_after_ms')} "
                  f"{row.get('seconds')}s {row.get('error') or ''}", flush=True)
    rows = list(ledger.done.values())
    before = [r["max_error_before_ms"] for r in rows if r.get("max_error_before_ms") is not None]
    after = [r["max_error_after_ms"] for r in rows if r.get("max_error_after_ms") is not None]
    print(f"this pass: {counts}; done in total {len(rows)} of {len(videos) - len(stubs)}; "
          f"max error before {max(before, default=None)} ms, after {max(after, default=None)} ms",
          flush=True)
    return 0 if not any(k not in DONE and k != "dry_run" for k in counts) else 1


def publish(args) -> int:
    """Put staged files over the delivered copies they were made from.

    Only where the delivered object is exactly as large as the copy that was
    retimed (the same object, server side copied by deliver.py); the record
    says so. Rows found "already" need nothing. Everything else is listed,
    for the direct mode (`tree` without `--stage`)."""
    work = Path(args.work)
    ledger = Ledger(work / "ledger.jsonl")
    delivered = listing(DELIVERED)
    staged = listing(STAGED)
    rows = [r for r in ledger.done.values() if r.get("target", "").startswith(STAGED)]
    plan, mismatched, already = [], [], 0
    for r in rows:
        rel = r["path"]
        if delivered.get(rel) != r["delivered_bytes"]:
            mismatched.append(rel)
        elif r["status"] == "already":
            already += 1
        elif staged.get(rel) == r.get("bytes"):
            plan.append(r)
        else:
            mismatched.append(rel)
    print(f"staged rows {len(rows)}; to publish {len(plan)}; already right {already}; "
          f"not publishable {len(mismatched)}", flush=True)
    for rel in mismatched:
        print("  not publishable:", rel, flush=True)
    if args.dry_run:
        return 0
    lock = threading.Lock()
    done = []

    def one(r):
        rel = r["path"]
        rclone("copyto", f"{STAGED}/{rel}", f"{DELIVERED}/{rel}", "--s3-acl", "private")
        got = next(iter(listing(f"{DELIVERED}/{rel}").values()), None)
        ok = got == r["bytes"]
        rec = json.loads(rclone("cat", f"{RECORDS}/{rel}.json"))
        rec.update(published=ok, published_bytes=got, published_at=time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime()))
        rclone("rcat", f"{RECORDS}/{rel}.json", stdin=json.dumps(rec, indent=1))
        with lock:
            done.append(ok)
            with (work / "published.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"path": rel, "ok": ok, "bytes": got}) + "\n")
        print(f"{'published' if ok else 'FAILED'} {rel} {got}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one, r) for r in plan]):
            fut.result()
    print(f"published {sum(done)} of {len(plan)}; failed {len(done) - sum(done)}; "
          f"not publishable {len(mismatched)}", flush=True)
    return 0 if all(done) and not mismatched else 1


def measure(source: Path, copy: Path) -> dict:
    """The comparison.json measure: every frame within 2 ms of the source's
    container pts after removing one constant offset."""
    a, b = source_timing(source), source_timing(copy)
    n = min(len(a), len(b))
    err = _max_error_ms(b.seconds()[:n], a.seconds()[:n])
    return {"frames_copy": len(b), "frames_raw": len(a), "max_abs_diff_ms_after_offset":
            round(err, 3), "timestamps_preserved": len(a) == len(b) and err <= 2.0,
            "exact": a.pts == b.pts and a.time_base == b.time_base}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one", help="retime one local copy")
    one.add_argument("source", type=Path)
    one.add_argument("copy", type=Path)
    one.add_argument("output", type=Path)
    t = sub.add_parser("tree", help="retime the delivered tree in the bucket")
    t.add_argument("--work", default="/scratch/retime")
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--dry-run", action="store_true",
                   help="download, retime and check, but upload nothing")
    t.add_argument("--stage", action="store_true",
                   help="take the copies from the work bucket and upload to vs:faceblur/retimed/")
    pb = sub.add_parser("publish", help="put staged files over the delivered copies")
    pb.add_argument("--work", default="/scratch/retime")
    pb.add_argument("--workers", type=int, default=16)
    pb.add_argument("--dry-run", action="store_true")
    m = sub.add_parser("measure", help="the comparison.json timestamp measure for one pair")
    m.add_argument("source", type=Path)
    m.add_argument("copy", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "one":
        rec = retime_file(args.source, args.copy, args.output)
        print(json.dumps(rec, indent=1))
        return 0 if rec["status"] in DONE else 1
    if args.cmd == "measure":
        res = measure(args.source, args.copy)
        print(json.dumps(res))
        return 0 if res["timestamps_preserved"] else 1
    if args.cmd == "publish":
        return publish(args)
    return tree(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
