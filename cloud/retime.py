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
from faceblur.timestamps import (TimingError, check_retimed, copy_layout,  # noqa: E402
                                 decode_check, retime, same_timing, source_timing,
                                 _max_error_ms)

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


# ----------------------------------------- without moving the pictures (compose)
#
# The bucket gives a VM about 40 MB/s, which puts a download of every copy and
# source, and an upload of every result, at some 20 hours. `compose` moves
# almost nothing instead. A retimed copy differs from the copy only in its
# index: the video payload is the same bytes in the same order. So:
#
# - the copy's index (`ftyp`, `moov`) and the source's (`moov` and one `moof`
#   per fragment; the camera writes fragmented MP4) are fetched with ranged
#   reads into sparse local files whose media data are holes;
# - `retime` runs on those, which yields the new file's header (everything
#   before the first picture) with the source's timestamps; the payload it
#   writes is zeros and is thrown away;
# - the new object is assembled in the bucket: part 1 is the header plus the
#   first 8 MiB of the copy's payload, the other parts are server side copies
#   of the copy's payload byte ranges (S3 UploadPartCopy);
# - the object is checked: its size, its header read back, the timestamps of
#   every frame against the source's, and a sample of packets byte for byte
#   against the copy's. Small files are also downloaded and decoded.

PART = 1 << 30                    # 1 GiB per server side copied part
FIRST = 8 << 20                   # payload bytes uploaded with the header (parts are >= 5 MiB)
DECODE_UNDER = 256 << 20          # results smaller than this are downloaded and decoded


def s3_client():
    import boto3
    from botocore.config import Config
    ep = os.environ["S3_ENDPOINT"]
    if not ep.startswith("http"):
        ep = "https://" + ep
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
                        config=Config(max_pool_connections=128, s3={"addressing_style": "path"},
                                      retries={"max_attempts": 10, "mode": "standard"}))


def split_remote(remote: str) -> tuple[str, str]:
    """"vs:bucket/key" -> (bucket, key)."""
    path = remote.split(":", 1)[1]
    bucket, key = path.split("/", 1)
    return bucket, key


class Object:
    def __init__(self, s3, remote: str):
        self.s3, self.remote = s3, remote
        self.bucket, self.key = split_remote(remote)
        self.size = s3.head_object(Bucket=self.bucket, Key=self.key)["ContentLength"]

    def read(self, offset: int, count: int) -> bytes:
        count = min(count, self.size - offset)
        if count <= 0:
            return b""
        r = self.s3.get_object(Bucket=self.bucket, Key=self.key,
                               Range=f"bytes={offset}-{offset + count - 1}")
        data = r["Body"].read()
        if len(data) != count:
            raise TimingError(f"{self.key}: read {len(data)} bytes of {count} at {offset}")
        return data


def sparse_index(read, total: int, dst: Path, chunk: int = 1 << 16) -> list:
    """Write `dst`: `total` bytes long, every top level box in place except
    the media data, of which only the box headers are written; the rest are
    holes. Returns [(kind, offset, size, header length)]."""
    import struct
    boxes = []
    off = 0
    with open(dst, "wb") as f:
        f.truncate(total)
        while off < total:
            base = off
            buf = read(off, min(chunk, total - off))
            while True:
                pos = off - base
                if pos + 8 > len(buf):
                    break
                size, kind = struct.unpack(">I4s", buf[pos:pos + 8])
                hl = 8
                if size == 1:
                    if pos + 16 > len(buf):
                        break
                    size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
                    hl = 16
                elif size == 0:
                    size = total - off
                if size < hl or off + size > total:
                    raise TimingError(f"bad box {kind!r} at {off}, size {size}")
                boxes.append((kind, off, size, hl))
                f.seek(off)
                if kind == b"mdat":
                    f.write(buf[pos:pos + hl] if pos + hl <= len(buf) else read(off, hl))
                    off += size
                    break
                if pos + size <= len(buf):
                    f.write(buf[pos:pos + size])
                    off += size
                    continue
                f.write(read(off, size))
                off += size
                break
            if off == base:
                raise TimingError(f"no progress at {off}")
    return boxes


def packet_spans(path: Path) -> list[tuple[int, int]]:
    """(byte offset, size) of every video packet, in decode order."""
    av = __import__("av")
    with av.open(str(path)) as c:
        return [(p.pos, p.size) for p in c.demux(c.streams.video[0]) if p.size]


def contiguous(spans: list) -> bool:
    return all(a[0] + a[1] == b[0] for a, b in zip(spans, spans[1:]))


CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _parse(data: bytes) -> list:
    """[(kind, children or payload bytes)] for a run of boxes."""
    import struct
    out, off = [], 0
    while off < len(data):
        size, kind = struct.unpack(">I4s", data[off:off + 8])
        if size < 8 or size == 1 or off + size > len(data):
            raise TimingError(f"unsupported box {kind!r} in the index")
        body = data[off + 8:off + size]
        out.append([kind, _parse(body) if kind in CONTAINERS else body])
        off += size
    return out


def _build(boxes: list) -> bytes:
    import struct
    out = b""
    for kind, body in boxes:
        raw = _build(body) if isinstance(body, list) else body
        out += struct.pack(">I4s", 8 + len(raw), kind) + raw
    return out


def _find(boxes: list, *path: bytes) -> list:
    """Every box at this path of kinds."""
    found = [b for b in boxes if b[0] == path[0]]
    if len(path) == 1:
        return found
    return [x for b in found for x in _find(b[1], *path[1:])]


def retimed_moov(moov: bytes, pts: list, last_duration: int, time_base, shift: int) -> bytes:
    """The copy's `moov` box (whole, header included) with the source's
    timestamps: new sample durations (`stts`), new track, media and movie
    durations, and chunk offsets moved by `shift` plus the change in the
    box's own size. Everything else stays the copy's, byte for byte.

    For a one track video copy with no reordering (no `ctts` offsets), a
    media time scale of the source's time base and a source whose first
    frame is at 0; anything else is refused."""
    import struct
    tree = _parse(moov[8:])
    traks = _find(tree, b"trak")
    if len(traks) != 1:
        raise TimingError(f"the copy has {len(traks)} tracks")
    trak = traks[0][1]
    hdlr = _find(trak, b"mdia", b"hdlr")[0][1]
    if hdlr[8:12] != b"vide":
        raise TimingError("the copy's track is not video")
    if pts[0] != 0:
        raise TimingError(f"the source's first frame is at {pts[0]}, not 0")
    for ctts in _find(trak, b"mdia", b"minf", b"stbl", b"ctts"):
        body = ctts[1]
        n = struct.unpack(">I", body[4:8])[0]
        if any(struct.unpack(">I", body[8 + 8 * i + 4:8 + 8 * i + 8])[0] for i in range(n)):
            raise TimingError("the copy reorders frames (ctts)")
    mdhd = _find(trak, b"mdia", b"mdhd")[0]
    b = bytearray(mdhd[1])
    v = b[0]
    ts_at = 20 if v == 1 else 12
    media_ts = struct.unpack(">I", b[ts_at:ts_at + 4])[0]
    if time_base.numerator != 1 or time_base.denominator != media_ts:
        raise TimingError(f"time scale {media_ts}, source time base {time_base}")
    durations = [y - x for x, y in zip(pts, pts[1:])] + [last_duration]
    total = sum(durations)
    if v == 1:
        b[24:32] = struct.pack(">Q", total)
    else:
        b[16:20] = struct.pack(">I", total)
    mdhd[1] = bytes(b)
    # stts: run length coded durations.
    runs = []
    for d in durations:
        if runs and runs[-1][1] == d:
            runs[-1][0] += 1
        else:
            runs.append([1, d])
    stts = _find(trak, b"mdia", b"minf", b"stbl", b"stts")[0]
    stts[1] = stts[1][:4] + struct.pack(">I", len(runs)) + b"".join(
        struct.pack(">II", c, d) for c, d in runs)
    # Movie time scale durations: mvhd, tkhd, elst.
    mvhd = _find(tree, b"mvhd")[0]
    m = bytearray(mvhd[1])
    mts_at = 20 if m[0] == 1 else 12
    movie_ts = struct.unpack(">I", m[mts_at:mts_at + 4])[0]
    movie_total = -(-total * movie_ts // media_ts)          # rounded up, as ffmpeg does
    if m[0] == 1:
        m[24:32] = struct.pack(">Q", movie_total)
    else:
        m[16:20] = struct.pack(">I", movie_total)
    mvhd[1] = bytes(m)
    tkhd = _find(trak, b"tkhd")[0]
    t = bytearray(tkhd[1])
    if t[0] == 1:
        t[28:36] = struct.pack(">Q", movie_total)
    else:
        t[20:24] = struct.pack(">I", movie_total)
    tkhd[1] = bytes(t)
    for elst in _find(trak, b"edts", b"elst"):
        e = bytearray(elst[1])
        n = struct.unpack(">I", e[4:8])[0]
        if n != 1:
            raise TimingError(f"the copy's edit list has {n} entries")
        if e[0] == 1:
            media_time = struct.unpack(">q", e[16:24])[0]
            e[8:16] = struct.pack(">Q", movie_total)
        else:
            media_time = struct.unpack(">i", e[12:16])[0]
            e[8:12] = struct.pack(">I", movie_total)
        if media_time != 0:
            raise TimingError(f"the copy's edit list starts at {media_time}")
        elst[1] = bytes(e)
    stbl = _find(trak, b"mdia", b"minf", b"stbl")[0][1]

    def offsets(delta: int) -> None:
        for box in stbl:
            if box[0] == b"stco":
                body = box[1]
                n = struct.unpack(">I", body[4:8])[0]
                vals = struct.unpack(f">{n}I", body[8:8 + 4 * n])
                if any(x + delta >= 2**32 or x + delta < 0 for x in vals):
                    raise TimingError("chunk offsets overflow 32 bits")
                box[1] = body[:8] + struct.pack(f">{n}I", *(x + delta for x in vals))
            elif box[0] == b"co64":
                body = box[1]
                n = struct.unpack(">I", body[4:8])[0]
                vals = struct.unpack(f">{n}Q", body[8:8 + 8 * n])
                box[1] = body[:8] + struct.pack(f">{n}Q", *(x + delta for x in vals))

    size_now = len(_build([[b"moov", tree]]))
    delta = shift + size_now - len(moov)
    offsets(delta)
    out = _build([[b"moov", tree]])
    if len(out) != size_now:               # the offsets do not change the size
        raise TimingError("the index changed size while moving its offsets")
    return out


def compose_upload(s3, header: bytes, src: Object, a: int, b: int, dst_remote: str) -> None:
    """dst = header + src[a:b], assembled in the bucket, private."""
    bucket, key = split_remote(dst_remote)
    k = min(b - a, FIRST)
    first = header + src.read(a, k)
    if a + k >= b:
        s3.put_object(Bucket=bucket, Key=key, Body=first, ACL="private", ContentType="video/mp4")
        return
    up = s3.create_multipart_upload(Bucket=bucket, Key=key, ACL="private",
                                    ContentType="video/mp4")["UploadId"]
    try:
        parts = [{"PartNumber": 1, "ETag": s3.upload_part(
            Bucket=bucket, Key=key, UploadId=up, PartNumber=1, Body=first)["ETag"]}]
        ranges, off = [], a + k
        while off < b:
            end = min(b, off + PART)
            ranges.append((len(ranges) + 2, off, end))
            off = end

        def copy_part(r):
            n, x, y = r
            res = s3.upload_part_copy(Bucket=bucket, Key=key, UploadId=up, PartNumber=n,
                                      CopySource={"Bucket": src.bucket, "Key": src.key},
                                      CopySourceRange=f"bytes={x}-{y - 1}")
            return {"PartNumber": n, "ETag": res["CopyPartResult"]["ETag"]}

        with ThreadPoolExecutor(max_workers=4) as pool:
            parts += list(pool.map(copy_part, ranges))
        s3.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=up,
                                     MultipartUpload={"Parts": parts})
    except BaseException:
        try:
            s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=up)
        except Exception:
            pass
        raise


def compose_one(s3, rel: str, copy_remote: str, src_remote: str, work: Path,
                dry_run: bool) -> dict:
    """Retime one copy into STAGED/rel without moving its pictures. Returns
    the ledger row; the record goes to RECORDS/rel.json."""
    import random
    d = work / f"job_{abs(hash(rel)) % 10**10}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    t0 = time.time()
    target = f"{STAGED}/{rel}"
    row = {"path": rel, "source": src_remote, "copy_remote": copy_remote, "target": target,
           "method": "compose"}
    rec = {"path": rel, "source_remote": src_remote, "copy_remote": copy_remote,
           "status": "failed", "error": "", "method": "compose", "faceblur_version": __version__}
    try:
        copy, src = Object(s3, copy_remote), Object(s3, src_remote)
        row["delivered_bytes"] = copy.size
        if src.size == copy.size:
            raise TimingError("the copy is the size of its source")
        ci, si = d / "copy_index.mp4", d / "source_index.mp4"
        cboxes = sparse_index(copy.read, copy.size, ci)
        sparse_index(src.read, src.size, si)
        timing = source_timing(si)
        layout = copy_layout(ci)
        rec.update(frames=layout.frames, source_frames=len(timing),
                   time_base=f"{timing.time_base.numerator}/{timing.time_base.denominator}")
        if layout.frames != len(timing):
            rec.update(status="refused", error=f"frame counts differ: copy {layout.frames}, "
                                                f"source {len(timing)}")
            return row
        if layout.reorder:
            rec.update(status="refused", error=f"decode order is not presentation order "
                                                f"(reorder delay {layout.reorder})")
            return row
        if same_timing(ci, timing):
            rec.update(status="already", max_error_before_ms=0.0, max_error_after_ms=0.0)
            return row
        kinds = [k for k, _, _, _ in cboxes]
        if kinds.count(b"mdat") != 1 or kinds.count(b"moov") != 1 or kinds[-1] != b"mdat"                 or kinds.index(b"moov") > kinds.index(b"mdat"):
            raise TimingError(f"the copy's layout is not index first: {kinds}")
        _, mo, ms, _ = cboxes[kinds.index(b"moov")]
        _, do, ds, dh = cboxes[-1]
        a, b = do + dh, do + ds
        cspans = packet_spans(ci)
        if not contiguous(cspans) or cspans[0][0] != a or cspans[-1][0] + cspans[-1][1] != b:
            raise TimingError("the copy's pictures do not fill its media data in order")
        with open(ci, "rb") as f:
            index = f.read(a)
        before = _max_error_ms(source_timing(ci).seconds(), timing.seconds())
        moov = retimed_moov(index[mo:mo + ms], list(timing.pts), timing.last_duration,
                            timing.time_base, 0)
        header = index[:mo] + moov + index[mo + ms:a]
        h = len(header)
        final = h + (b - a)
        rec.update(max_error_before_ms=round(before, 3), header_bytes=h, bytes=final,
                   copy_bytes=copy.size)
        # Check the new index before anything is written: as a sparse file
        # it must show the source's timestamps and the copy's packet sizes.
        pi = d / "planned_index.mp4"
        with open(pi, "wb") as f:
            f.truncate(final)
            f.write(header)
        got = source_timing(pi)
        if got.time_base != timing.time_base or got.pts != timing.pts:
            raise TimingError("the new index does not carry the source's timestamps")
        ospans = packet_spans(pi)
        if [s for _, s in ospans] != [s for _, s in cspans] or ospans[0][0] != h:
            raise TimingError("the new index does not point at the copy's packets")
        if dry_run:
            rec["status"] = "dry_run"
            return row
        compose_upload(s3, header, copy, a, b, target)
        # Check what is in the bucket now.
        dst = Object(s3, target)
        if dst.size != final:
            raise TimingError(f"assembled {dst.size} bytes, expected {final}")
        if dst.read(0, h) != header:
            raise TimingError("the assembled header differs")
        rec["max_error_after_ms"] = round(_max_error_ms(got.seconds(), timing.seconds()), 3)
        picks = sorted({0, len(cspans) - 1, *random.Random(rel).sample(
            range(len(cspans)), min(14, len(cspans)))})
        for i in picks:
            (cp, cs), (op, os_) = cspans[i], ospans[i]
            if copy.read(cp, cs) != dst.read(op, os_):
                raise TimingError(f"packet {i} differs from the copy's")
        rec["packets_compared"] = len(picks)
        if final < DECODE_UNDER:
            full = d / "final.mp4"
            s3.download_file(dst.bucket, dst.key, str(full))
            problem = check_retimed(full, list(timing.pts), timing.time_base) or \
                decode_check(full, len(timing))
            if problem:
                raise TimingError(f"downloaded check: {problem}")
            rec["decoded"] = True
        rec["status"] = "retimed"
        rec["uploaded"] = True
    except TimingError as exc:
        msg = str(exc)
        rec["status"] = "refused" if ("differ" in msg or "presentation order" in msg) else "failed"
        rec["error"] = msg
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        shutil.rmtree(d, ignore_errors=True)
        rec["seconds"] = round(time.time() - t0, 2)
        row.update({k: rec.get(k) for k in ("status", "frames", "source_frames", "time_base",
                                            "max_error_before_ms", "max_error_after_ms",
                                            "error", "bytes", "uploaded")})
        row["seconds"] = rec["seconds"]
        if not dry_run:
            try:
                rb, rk = split_remote(f"{RECORDS}/{rel}.json")
                s3.put_object(Bucket=rb, Key=rk, Body=json.dumps(rec, indent=1).encode(),
                              ACL="private", ContentType="application/json")
            except Exception as exc:
                row["record_error"] = str(exc)[:200]
    return row


def compose_tree(args) -> int:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(work / "ledger.jsonl")
    s3 = s3_client()
    chosen = staged_copies()
    # --from-in: compare against the source as uploaded even where a
    # repaired one exists (a copy made before the repair holds its frames).
    repaired = set() if args.from_in else set(listing(REPAIRED))
    todo = sorted(rel for rel, (_, size) in chosen.items() if size >= EMPTY)
    skip = [rel for rel in todo if rel in ledger.done
            and ledger.done[rel].get("delivered_bytes") == chosen[rel][1]]
    todo = [rel for rel in todo if rel not in skip]
    if args.only:
        todo = [rel for rel in todo if args.only in rel]
    if args.limit:
        todo = todo[:args.limit]
    print(f"videos {len(chosen)}; done before (ledger) {len(skip)}; repaired sources "
          f"{len(repaired)}; to do now {len(todo)} (compose)", flush=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(compose_one, s3, rel, chosen[rel][0],
                               f"{REPAIRED}/{rel}" if rel in repaired else f"{SOURCES}/{rel}",
                               work, args.dry_run) for rel in todo]
        for fut in as_completed(futures):
            row = fut.result()
            if not args.dry_run:
                ledger.add(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"{row['status']:8s} {row['path']} frames={row.get('frames')} "
                  f"before={row.get('max_error_before_ms')} after={row.get('max_error_after_ms')} "
                  f"{row.get('seconds')}s {row.get('error') or ''}", flush=True)
    rows = list(ledger.done.values())
    before = [r["max_error_before_ms"] for r in rows if r.get("max_error_before_ms") is not None]
    print(f"this pass: {counts}; done in total {len(rows)} of {len(todo) + len(skip)}; "
          f"max error before {max(before, default=None)} ms", flush=True)
    return 0 if not any(k not in DONE and k != "dry_run" for k in counts) else 1


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
    cp = sub.add_parser("compose", help="retime the copies deliver.py picks into "
                                         "vs:faceblur/retimed/, moving only their index")
    cp.add_argument("--work", default="/scratch/retime")
    cp.add_argument("--workers", type=int, default=16)
    cp.add_argument("--limit", type=int, default=0)
    cp.add_argument("--only", default="", help="only paths containing this text")
    cp.add_argument("--from-in", action="store_true",
                    help="use vs:faceblur/in/ even where vs:faceblur/in_repaired/ has the source")
    cp.add_argument("--dry-run", action="store_true")
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
    if args.cmd == "compose":
        return compose_tree(args)
    return tree(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
