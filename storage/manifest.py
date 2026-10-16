"""The manifest: one Parquet row per stored file, saying what it is.

A converted file replaces its source under the same name, so without a record
nobody could tell later what it was before, whether it was checked, or which
bytes it held. Each row keeps both sha256 sums, both sizes and codecs, the
frame count, the length and the VMAF score. Parquet with zstd keeps a manifest
of a whole tree small and readable from pandas or pyarrow alike.

Writes go to `<path>.part` and are renamed into place, so a crash mid-write
never leaves a half manifest where the old one was.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from faceblur.settings import VIDEO_EXT
from faceblur.video import VideoError
from storage.probe import probe

MANIFEST_NAME = "manifest.parquet"


@dataclass(frozen=True)
class ManifestRow:
    path: str
    sha256_before: str
    sha256_after: str
    size_before: int
    size_after: int
    codec_before: str
    codec_after: str
    frames: int
    duration_s: float
    vmaf: float | None
    vmaf_min: float | None
    checks_passed: bool
    status: str
    encoder: str
    quality: int | None
    converted_at: str


SCHEMA = pa.schema([
    pa.field("path", pa.string(), nullable=False),
    pa.field("sha256_before", pa.string(), nullable=False),
    pa.field("sha256_after", pa.string(), nullable=False),
    pa.field("size_before", pa.int64(), nullable=False),
    pa.field("size_after", pa.int64(), nullable=False),
    pa.field("codec_before", pa.string(), nullable=False),
    pa.field("codec_after", pa.string(), nullable=False),
    pa.field("frames", pa.int64(), nullable=False),
    pa.field("duration_s", pa.float64(), nullable=False),
    pa.field("vmaf", pa.float64(), nullable=True),
    pa.field("vmaf_min", pa.float64(), nullable=True),
    pa.field("checks_passed", pa.bool_(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("encoder", pa.string(), nullable=False),
    pa.field("quality", pa.int64(), nullable=True),
    pa.field("converted_at", pa.string(), nullable=False),
])


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[ManifestRow]:
    """Every row of the manifest at `path`, in order; none when it does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    # Read through a handle closed right here, so no open handle stops the
    # rename of the next write on Windows.
    with open(path, "rb") as f:
        table = pq.read_table(pa.BufferReader(f.read()), schema=SCHEMA, memory_map=False)
    return [ManifestRow(**r) for r in table.to_pylist()]


def _replace(src: Path, dst: Path, tries: int = 20) -> None:
    """os.replace, retried briefly: on Windows a virus scanner or the indexer
    can hold a just-written file open for a moment, and the rename then fails."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.1)


def write_rows(path: Path, rows: list[ManifestRow]) -> None:
    path = Path(path)
    data = [asdict(r) for r in rows]
    table = pa.Table.from_pydict(
        {f.name: [d[f.name] for d in data] for f in fields(ManifestRow)}, schema=SCHEMA)
    part = Path(str(path) + ".part")
    with open(part, "wb") as f:
        pq.write_table(table, f, compression="zstd")
    _replace(part, path)


def append_row(path: Path, row: ManifestRow) -> None:
    write_rows(path, [*read_rows(path), row])


def _listed_row(root: Path, file: Path, stamp: str) -> ManifestRow:
    codec, frames, duration = "", 0, 0.0
    if file.suffix.lower() in VIDEO_EXT:
        try:
            facts = probe(file)
            codec, frames, duration = facts.codec, facts.frames, facts.duration_s
        except VideoError:
            codec = "unreadable"
    digest, size = sha256_of(file), file.stat().st_size
    return ManifestRow(path=file.relative_to(root).as_posix(), sha256_before=digest,
                       sha256_after=digest, size_before=size, size_after=size,
                       codec_before=codec, codec_after=codec, frames=frames,
                       duration_s=duration, vmaf=None, vmaf_min=None,
                       checks_passed=False, status="listed", encoder="",
                       quality=None, converted_at=stamp)


def build_tree_manifest(root: Path, out: Path | None = None) -> Path:
    """List every regular file under `root` in one manifest, sorted by path.

    The converter's rows already in `out` (every status but "listed") are kept
    first, unchanged; only the earlier listing is replaced.
    """
    root = Path(root)
    out = Path(out) if out is not None else root / MANIFEST_NAME
    stamp = now_utc()
    own = out.resolve()
    try:
        kept = [r for r in read_rows(out) if r.status != "listed"]
    except pa.ArrowInvalid:
        kept = []  # not a manifest at all: nothing in it to keep
    rows = [_listed_row(root, f, stamp) for f in sorted(root.rglob("*"))
            if f.is_file() and f.suffix != ".part" and f.resolve() != own]
    write_rows(out, kept + rows)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m storage.manifest",
                                     description="Write a Parquet manifest of a folder tree.")
    parser.add_argument("tree", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.tree.is_dir():
        print(f"Not a folder: {args.tree}", file=sys.stderr)
        return 2
    out = build_tree_manifest(args.tree, args.out)
    print(f"{len(read_rows(out))} files listed in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
