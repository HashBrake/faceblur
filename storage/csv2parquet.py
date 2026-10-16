"""Lossless CSV to Parquet for the IMU and pts side files.

A Parquet file is kept only when the CSV regenerated from it, read back from
disk, equals the source byte for byte. What a CSV parser would forget (the
exact header line, the line terminator, the decimals of each float column,
whether the file ends with a terminator) is detected from the file and stored
in the Parquet schema metadata. A file the format cannot reproduce (mixed
decimal counts, scientific notation, CRLF mixed with LF, an unknown header)
fails the round trip and is refused, never approximated. The CSV is deleted
only with `--apply` and only when its Parquet sits beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

FORMAT_VERSION = 1
META_KEY = b"csv2parquet"
IMU_COLUMNS = ["timestamp_us", "x", "y", "z", "type"]
PTS_COLUMNS = ["timestamp_us"]
FLOAT_COLUMNS = ["x", "y", "z"]
BLOCK = 64 << 20
BATCH_ROWS = 500_000
COMPARE_CHUNK = 1 << 20


class FormatError(ValueError):
    """The file is not in a form this converter can reproduce byte for byte."""


@dataclass(frozen=True)
class Result:
    path: Path
    out: Path
    size_before: int
    size_after: int
    ratio: float
    ok: bool
    error: str = ""


def _schema(columns: list[str]) -> pa.Schema:
    if columns == IMU_COLUMNS:
        return pa.schema([("timestamp_us", pa.int64()), ("x", pa.float64()),
                          ("y", pa.float64()), ("z", pa.float64()),
                          ("type", pa.dictionary(pa.int32(), pa.string()))])
    return pa.schema([("timestamp_us", pa.int64())])


def sniff(path: Path) -> dict:
    """The facts a regeneration needs, read from the start and the end of the file."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
        end = head.find(b"\n")
        if end < 0:
            raise FormatError("no line terminator after the header")
        term = b"\r\n" if head[end - 1:end] == b"\r" else b"\n"
        size = f.seek(0, 2)
        f.seek(max(0, size - len(term)))
        trailing = f.read() == term
    header = head[:end + 1 - len(term)].decode("ascii")
    columns = [c.strip() for c in header.split(",")]
    if columns not in (IMU_COLUMNS, PTS_COLUMNS):
        raise FormatError(f"unrecognised header {header!r}")
    decimals = {}
    if columns == IMU_COLUMNS:
        row_end = head.find(term, end + 1)
        first = head[end + 1:row_end if row_end >= 0 else len(head)].decode("ascii").split(",")
        if head[end + 1:]:
            if len(first) != len(columns):
                raise FormatError("the first data row does not have five fields")
            for name, field in zip(columns, first):
                if name in FLOAT_COLUMNS:
                    decimals[name] = len(field.split(".")[1]) if "." in field else 0
    return {"version": FORMAT_VERSION, "header": header, "terminator": term.decode("ascii"),
            "decimals": decimals, "trailing_terminator": trailing, "columns": columns}


def _reader(path: Path, meta: dict) -> pacsv.CSVStreamingReader:
    schema = _schema(meta["columns"])
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(skip_rows=1, column_names=meta["columns"],
                                       block_size=BLOCK),
        parse_options=pacsv.ParseOptions(delimiter=",", quote_char=False,
                                         escape_char=False, newlines_in_values=False,
                                         ignore_empty_lines=False),
        convert_options=pacsv.ConvertOptions(
            column_types={f.name: f.type for f in schema}, null_values=[],
            strings_can_be_null=False, quoted_strings_can_be_null=False))


def _with_meta(meta: dict) -> pa.Schema:
    return _schema(meta["columns"]).with_metadata({META_KEY: json.dumps(meta).encode()})


def read_csv(path: Path) -> pa.Table:
    """The CSV as a table, read block by block, with its regeneration facts as metadata."""
    meta = sniff(Path(path))
    schema = _with_meta(meta)
    return pa.Table.from_batches([b for b in _reader(Path(path), meta)], schema=schema)


def _fmt_float(arr: pa.Array, d: int) -> pa.Array:
    v = arr.to_numpy(zero_copy_only=False)
    scaled = np.abs(v) * 10.0 ** d
    if not np.all(np.isfinite(scaled)) or np.any(scaled >= 2.0 ** 53):
        raise FormatError("a float that fixed-point text cannot hold")
    digits = pc.utf8_lpad(pa.array(np.rint(scaled).astype(np.int64)).cast(pa.string()),
                          width=d + 1, padding="0")
    sign = pa.array(np.where(np.signbit(v), "-", ""))
    if d == 0:
        return pc.binary_join_element_wise(sign, digits, "")
    whole = pc.utf8_slice_codeunits(digits, 0, -d)
    frac = pc.utf8_slice_codeunits(digits, -d, 2 ** 31 - 1)
    return pc.binary_join_element_wise(pc.binary_join_element_wise(sign, whole, ""), frac, ".")


def _batch_bytes(batch: pa.RecordBatch, meta: dict) -> bytes:
    if batch.num_rows == 0:
        return b""
    fields = []
    for name in meta["columns"]:
        col = batch.column(name)
        if name in FLOAT_COLUMNS:
            fields.append(_fmt_float(col, meta["decimals"][name]))
        elif name == "type":
            fields.append(col.dictionary_decode() if pa.types.is_dictionary(col.type) else col)
        else:
            fields.append(col.cast(pa.string()))
    line = fields[0] if len(fields) == 1 else pc.binary_join_element_wise(*fields, ",")
    line = pc.binary_join_element_wise(line, pa.scalar(""), meta["terminator"])
    # Every row now ends with the terminator, so the value buffer is the text itself.
    offsets = np.frombuffer(line.buffers()[1], dtype=np.int32)[line.offset:line.offset + len(line) + 1]
    return line.buffers()[2].to_pybytes()[offsets[0]:offsets[-1]]


def _chunks(meta: dict, batches):
    """The regenerated CSV as a sequence of byte chunks."""
    term = meta["terminator"].encode()
    prev = meta["header"].encode() + term
    for batch in batches:
        cur = _batch_bytes(batch, meta)
        if cur:
            yield prev
            prev = cur
    if not meta["trailing_terminator"]:
        prev = prev[:-len(term)]
    yield prev


def _meta_of(schema: pa.Schema) -> dict:
    return json.loads(schema.metadata[META_KEY])


def to_csv_bytes(table: pa.Table) -> bytes:
    """The CSV regenerated from a table and its metadata."""
    return b"".join(_chunks(_meta_of(table.schema), table.to_batches()))


def _same_as_file(chunks, path: Path) -> bool:
    with open(path, "rb") as f:
        for chunk in chunks:
            view = memoryview(chunk)
            for i in range(0, len(view), COMPARE_CHUNK):
                part = view[i:i + COMPARE_CHUNK]
                if f.read(len(part)) != part:
                    return False
        return f.read(1) == b""


def _write(path: Path, out: Path) -> None:
    meta = sniff(path)
    schema = _with_meta(meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out, schema, compression="zstd", compression_level=9,
                          use_dictionary=True) as writer:
        for batch in _reader(path, meta):
            writer.write_batch(batch.replace_schema_metadata(schema.metadata))


def _round_trip_ok(path: Path, out: Path) -> bool:
    with pq.ParquetFile(out) as pf:
        meta = _meta_of(pf.schema_arrow)
        return _same_as_file(_chunks(meta, pf.iter_batches(batch_size=BATCH_ROWS)), path)


def convert(path: Path, out: Path, apply: bool = False) -> Result:
    path, out = Path(path), Path(out)
    before = path.stat().st_size
    error = ""
    try:
        _write(path, out)
        if not _round_trip_ok(path, out):
            error = "the regenerated CSV differs from the source"
    except (FormatError, pa.ArrowException, UnicodeDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error:
        out.unlink(missing_ok=True)
        return Result(path, out, before, 0, 0.0, False, error)
    after = out.stat().st_size
    if apply and out.resolve().parent == path.resolve().parent:
        path.unlink()
    return Result(path, out, before, after, after / before if before else 0.0, True)


def _wanted(path: Path) -> bool:
    return path.name.endswith("_imu.csv") or path.name.endswith("_pts.csv")


def _target(path: Path, tree: Path | None, out_dir: Path | None) -> Path:
    if out_dir is None:
        return path.with_suffix(".parquet")
    rel = Path(path.name)
    if tree is not None:
        try:
            rel = path.resolve().relative_to(tree.resolve())
        except ValueError:
            pass
    return (out_dir / rel).with_suffix(".parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m storage.csv2parquet",
        description="Convert IMU and pts CSV files to Parquet, checked byte for byte.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--tree", type=Path, default=None,
                        help="also convert every *_imu.csv and *_pts.csv under this folder")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the Parquet files here, mirrored, instead of beside the CSV")
    parser.add_argument("--apply", action="store_true",
                        help="delete each CSV after a passing round trip when its Parquet is beside it")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    files = list(args.paths)
    if args.tree is not None:
        files += sorted(p for p in args.tree.rglob("*.csv") if _wanted(p))
    seen, done, failed, total_before, total_after = set(), 0, 0, 0, 0
    for path in files:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        if not _wanted(path):
            print(f"skipped  {path} (not an *_imu.csv or *_pts.csv file)")
            continue
        done += 1
        if not path.is_file():
            print(f"failed   {path} (no such file)")
            failed += 1
            continue
        r = convert(path, _target(path, args.tree, args.out), apply=args.apply)
        total_before += r.size_before
        if r.ok:
            total_after += r.size_after
            print(f"ok       {path}  {r.size_before} -> {r.size_after} bytes  ratio {r.ratio:.4f}")
        else:
            failed += 1
            print(f"failed   {path}  {r.size_before} bytes  ({r.error})")
    ratio = total_after / total_before if total_before else 0.0
    print(f"total    {done} files, {failed} failed  {total_before} -> {total_after} bytes"
          f"  ratio {ratio:.4f}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
