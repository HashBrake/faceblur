"""audit.jsonl: the hash chain, its verifier, the JSON Schema and sha256_file."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from faceblur import audit

SHA = "ab" * 32


def write_run(path, footer=True):
    """header, 3 frames over two cameras, 2 speech, 2 camera_done, footer."""
    with audit.AuditWriter(path) as w:
        w.header(SHA, "in.h5", ["left", "right"], {"yunet": SHA}, {"blur": 5},
                 None, "0.2.0")
        w.frame("left", 0, 0, {"face": 1}, {"face": [[10.04, 20.0, 30.0, 40.0]]})
        w.frame("left", 1, 33_000_000, {"face": 1, "text": 1},
                {"face": [[12.345, 20.0, 30.0, 40.0]], "text": [[1, 2, 3, 4]]},
                quads={"text": [[[1, 2], [4, 2], [4, 6], [1, 6]]]})
        w.frame("right", 0, 0, {}, {})
        w.speech(1.5, 2.25, 200)
        w.speech(4.0, 5.0, 200)
        w.camera_done("left", 2, 0.5, {"source": "left", "frames": 2})
        w.camera_done("right", 1, 0.25, {"source": "right", "frames": 1})
        if footer:
            w.footer(SHA, "episode.h5", "ok")


def rewrite(path, lines):
    path.write_bytes(b"".join(line + b"\n" for line in lines))


def test_chain_and_read(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path)
    assert audit.verify(path) == (True, 9, "")
    lines = audit.read(path)
    assert [line["seq"] for line in lines] == list(range(9))
    assert lines[0]["prev"] == audit.GENESIS
    for before, after in zip(lines, lines[1:]):
        assert after["prev"] == before["hash"]
    assert lines[1]["counts"] == {"face": 1, "plate": 0, "screen": 0, "text": 0}
    assert lines[1]["boxes"]["face"] == [[10.0, 20.0, 30.0, 40.0]]
    assert lines[2]["boxes"]["face"] == [[12.3, 20.0, 30.0, 40.0]]
    assert lines[0]["time"].endswith("Z") and len(lines[0]["time"]) == 20


def test_every_line_validates_and_bad_lines_fail(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path)
    schema = audit.load_schema()
    lines = audit.read(path)
    for line in lines:
        audit.validate_line(line, schema)

    no_plate = copy.deepcopy(lines[1])
    del no_plate["counts"]["plate"]
    five = copy.deepcopy(lines[1])
    five["boxes"]["face"] = [[1, 2, 3, 4, 5]]
    unknown = copy.deepcopy(lines[4])
    unknown["type"] = "gossip"
    for bad in (no_plate, five, unknown):
        with pytest.raises(ValidationError):
            audit.validate_line(bad, schema)


def test_tamper_box_value(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path)
    lines = path.read_bytes().splitlines()
    assert b"12.3" in lines[2]
    lines[2] = lines[2].replace(b"12.3", b"13.3")
    rewrite(path, lines)
    ok, _, problem = audit.verify(path)
    assert not ok and problem == "hash mismatch at seq 2"


def test_tamper_deleted_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path)
    lines = path.read_bytes().splitlines()
    del lines[4]
    rewrite(path, lines)
    assert audit.verify(path) == (False, 4, "prev mismatch at seq 5")


def test_tamper_appended_line_with_wrong_prev(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path, footer=False)
    payload = {"seq": 8, "prev": "1" * 64, "type": "footer", "time": "2026-11-05T00:00:00Z",
               "output_sha256": SHA, "output_name": "episode.h5", "status": "ok", "error": ""}
    line = {**payload, "hash": audit.line_hash(payload["prev"], payload)}
    with open(path, "ab") as f:
        f.write(audit.canonical(line) + b"\n")
    assert audit.verify(path) == (False, 8, "prev mismatch at seq 8")


def test_bad_json_named_by_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path)
    lines = path.read_bytes().splitlines()
    lines[3] = lines[3][:-5]
    rewrite(path, lines)
    assert audit.verify(path) == (False, 3, "bad json at line 4")


def test_canonical_is_order_independent_and_rejects_nan():
    a = {"seq": 1, "prev": audit.GENESIS, "type": "speech", "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "type": "speech", "prev": audit.GENESIS, "seq": 1}
    assert list(a) != list(b)
    assert audit.canonical(a) == audit.canonical(b)
    assert audit.line_hash(audit.GENESIS, a) == audit.line_hash(audit.GENESIS, b)
    with pytest.raises(ValueError):
        audit.canonical({"x": float("nan")})


def test_append_continues_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_run(path, footer=False)
    with audit.AuditWriter(path, append=True) as w:
        line = w.footer(SHA, "episode.h5", "ok")
    assert line["seq"] == 8
    assert audit.verify(path) == (True, 9, "")


def test_sha256_file(tmp_path):
    data = bytes(range(256)) * 5000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert audit.sha256_file(path, chunk=4096) == hashlib.sha256(data).hexdigest()


def test_schema_is_valid_draft_2020_12():
    schema = json.loads(audit.SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_zone_kept_is_written_only_when_there_is_something_and_validates(tmp_path):
    path = tmp_path / "audit.jsonl"
    with audit.AuditWriter(path) as w:
        w.header(SHA, "in.h5", ["left"], {"yunet": SHA}, {}, None, "0.2.0")
        empty = w.frame("left", 0, 0, {}, {}, zone_kept={})
        kept = w.frame("left", 1, 1, {"text": 1}, {"text": [[1, 2, 3, 4]]},
                       zone_kept={"face": [[10.04, 20, 30, 40]], "screen": [[0, 0, 5, 5]]})
    assert "zone_kept" not in empty
    lines = audit.read(path)
    assert "zone_kept" not in lines[1]
    assert lines[2]["zone_kept"] == {"face": [[10.0, 20.0, 30.0, 40.0]],
                                     "screen": [[0.0, 0.0, 5.0, 5.0]]}
    assert lines[2]["counts"]["face"] == 0
    assert kept["hash"] == lines[2]["hash"]
    assert audit.verify(path) == (True, 3, "")
    schema = audit.load_schema()
    for line in lines:
        audit.validate_line(line, schema)
    five = copy.deepcopy(lines[2])
    five["zone_kept"]["face"] = [[1, 2, 3, 4, 5]]
    with pytest.raises(ValidationError):
        audit.validate_line(five, schema)
    unknown = copy.deepcopy(lines[2])
    unknown["zone_kept"]["hand"] = [[1, 2, 3, 4]]
    with pytest.raises(ValidationError):
        audit.validate_line(unknown, schema)
