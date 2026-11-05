"""The audit record of one run: an append-only, hash-chained JSONL file.

The partner pipeline keeps the de-identified episode and throws the input
away, so audit.jsonl is the only evidence of what was removed and with which
models. A reviewer has to be able to trust it without trusting us, so every
line carries the sha256 of the line before it: change, drop or reorder a line
and every hash after it stops matching. `verify` walks the chain and names the
first line that breaks it.

A hash is only reproducible if the bytes it covers are, so a line is hashed in
one canonical form (sorted keys, no spaces, ASCII, no NaN) whatever order its
fields were built in. The file itself is written as that same canonical form
plus the `hash` field, one line per record.

Each line type has a writer method with fixed field names, and
docs/audit_schema.json describes the same names, so the two cannot drift
without a test noticing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator

GENESIS = "0" * 64
CLASSES = ("face", "plate", "screen", "text")
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "audit_schema.json"


def canonical(obj) -> bytes:
    """The one byte form a record is hashed in. NaN raises: it has no JSON form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def line_hash(prev: str, payload: dict) -> str:
    """sha256 of the previous hash and the line without its own `hash` field."""
    return hashlib.sha256(prev.encode() + canonical(payload)).hexdigest()


def _now() -> str:
    """UTC to the second, with Z, as the schema's date-time asks."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _last_line(path: Path) -> bytes:
    """The last non-empty line of a file, read from the end: the file may be large."""
    with open(path, "rb") as f:
        end = f.seek(0, 2)
        pos = end
        while pos > 0:
            pos = max(0, pos - (1 << 16))
            f.seek(pos)
            lines = f.read(end - pos).rstrip(b"\n").split(b"\n")
            if len(lines) > 1 or pos == 0:
                return lines[-1]
    return b""


def _round(values) -> list[float]:
    return [round(float(v), 1) for v in values]


class AuditWriter:
    """Appends chained lines to audit.jsonl; every write is flushed at once."""

    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.seq = 0
        self.prev = GENESIS
        if append and self.path.exists() and self.path.stat().st_size > 0:
            last = json.loads(_last_line(self.path))
            self.seq = last["seq"] + 1
            self.prev = last["hash"]
        self._file = open(self.path, "a" if append else "w", encoding="ascii", newline="\n")

    def write(self, type: str, **fields) -> dict:
        payload = {"seq": self.seq, "prev": self.prev, "type": type, "time": _now(), **fields}
        line = {**payload, "hash": line_hash(self.prev, payload)}
        self._file.write(canonical(line).decode() + "\n")
        # Flushed per line so a crash leaves a chain that verifies up to the crash.
        self._file.flush()
        self.seq += 1
        self.prev = line["hash"]
        return line

    def header(self, input_sha256: str, input_name: str, cameras: list[str],
               models: dict[str, str], settings: dict, image_digest: str | None,
               tool_version: str) -> dict:
        return self.write("header", input_sha256=input_sha256, input_name=input_name,
                          cameras=list(cameras), models=dict(models), settings=settings,
                          image_digest=image_digest, tool_version=tool_version)

    def frame(self, camera: str, frame: int, pts_ns: int, counts: dict[str, int],
              boxes: dict[str, list[list[float]]],
              quads: dict[str, list[list[list[float]]]] | None = None,
              zone_kept: dict[str, list[list[float]]] | None = None) -> dict:
        # Every class is counted, zero included, so a reader never has to guess
        # whether a missing class meant none found or not looked for.
        # `counts` and `boxes` are what was painted; `zone_kept` lists what was
        # found but left clear because the handled zone covers it.
        fields = {
            "camera": camera, "frame": int(frame), "pts_ns": int(pts_ns),
            "counts": {c: int(counts.get(c, 0)) for c in CLASSES},
            "boxes": {c: [_round(b) for b in bs] for c, bs in boxes.items()},
        }
        if quads is not None:
            fields["quads"] = {c: [[_round(p) for p in q] for q in qs]
                               for c, qs in quads.items()}
        if zone_kept:
            fields["zone_kept"] = {c: [_round(b) for b in bs] for c, bs in zone_kept.items()}
        return self.write("frame", **fields)

    def speech(self, start_s: float, end_s: float, pad_ms: int) -> dict:
        return self.write("speech", start_s=float(start_s), end_s=float(end_s),
                          pad_ms=int(pad_ms))

    def camera_done(self, camera: str, frames: int, seconds: float, record: dict) -> dict:
        return self.write("camera_done", camera=camera, frames=int(frames),
                          seconds=float(seconds), record=record)

    def footer(self, output_sha256: str, output_name: str, status: str,
               error: str = "") -> dict:
        return self.write("footer", output_sha256=output_sha256, output_name=output_name,
                          status=status, error=error)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> AuditWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read(path) -> list[dict]:
    with open(path, encoding="ascii") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(path) -> tuple[bool, int, str]:
    """Walk the chain. Returns (ok, lines checked, the first problem or "")."""
    with open(path, "rb") as f:
        raw = [line for line in f.read().split(b"\n") if line.strip()]
    prev = GENESIS
    checked = 0
    for number, text in enumerate(raw, start=1):
        try:
            line = json.loads(text)
            seq, stored = line["seq"], line.pop("hash")
            computed = line_hash(str(line.get("prev", "")), line)
        except (ValueError, KeyError, TypeError, AttributeError):
            return False, checked, f"bad json at line {number}"
        # Hash, then link, then position: a deleted line leaves every survivor's
        # own hash intact, so the break shows first as a prev that points nowhere.
        if computed != stored:
            return False, checked, f"hash mismatch at seq {seq}"
        if line.get("prev") != prev:
            return False, checked, f"prev mismatch at seq {seq}"
        if seq != number - 1:
            return False, checked, f"seq mismatch at seq {seq}"
        kind = line.get("type")
        if (seq == 0) != (kind == "header"):
            return False, checked, f"header misplaced at seq {seq}"
        if kind == "footer" and number != len(raw):
            return False, checked, f"footer not last at seq {seq}"
        prev = stored
        checked += 1
    if not checked:
        return False, 0, "no header"
    return True, checked, ""


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_line(line: dict, schema: dict | None = None) -> None:
    """Raises jsonschema.ValidationError if the line does not match the schema."""
    Draft202012Validator(schema if schema is not None else load_schema()).validate(line)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    """Streamed: the inputs are hundreds of MB and this PC has little memory free."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()
