"""CSV to Parquet: the regenerated CSV equals the source byte for byte, or nothing is kept."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from storage import csv2parquet as c2p

IMU_HEADER = b"timestamp_us\t,x\t,y\t,z\t,type\n"
IMU_ROWS = (b"18423359,-1.515530,-9.624691,-1.630451,accel\n"
            b"18423359,-0.051133,0.034088,-0.026632,gyro\n"
            b"18428359,-0.000000,0.000000,123456.789012,accel\n"
            b"18428359,0.000001,-0.000001,-98765.432109,gyro\n"
            b"18433359,9.806650,-0.000000,0.000000,accel\n")
PTS = b"timestamp_us\n339862298581\n339862341146\n339862383711\n"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_imu_round_trip_is_byte_identical(tmp_path):
    src = _write(tmp_path / "rec_imu.csv", IMU_HEADER + IMU_ROWS)
    out = tmp_path / "rec_imu.parquet"
    r = c2p.convert(src, out)
    assert r.ok and r.size_before == src.stat().st_size and r.size_after == out.stat().st_size
    table = pq.read_table(out)
    assert c2p.to_csv_bytes(table) == src.read_bytes()
    assert table.schema.field("timestamp_us").type == pa.int64()
    for name in ("x", "y", "z"):
        assert table.schema.field(name).type == pa.float64()
    assert table.schema.field("type").type == pa.dictionary(pa.int32(), pa.string())
    meta = c2p._meta_of(table.schema)
    assert meta["header"] == "timestamp_us\t,x\t,y\t,z\t,type"
    assert meta["terminator"] == "\n" and meta["trailing_terminator"] is True
    assert meta["decimals"] == {"x": 6, "y": 6, "z": 6} and meta["version"] == 1
    assert src.exists()


def test_read_csv_keeps_negative_zero(tmp_path):
    src = _write(tmp_path / "rec_imu.csv", IMU_HEADER + IMU_ROWS)
    table = c2p.read_csv(src)
    x = table.column("x").to_pylist()
    assert str(x[2]) == "-0.0" and table.num_rows == 5
    assert c2p.to_csv_bytes(table) == src.read_bytes()


def test_pts_round_trip_is_byte_identical(tmp_path):
    src = _write(tmp_path / "rec_camera_left_part0001_pts.csv", PTS)
    out = tmp_path / "rec_camera_left_part0001_pts.parquet"
    assert c2p.convert(src, out).ok
    table = pq.read_table(out)
    assert table.schema.field("timestamp_us").type == pa.int64() and table.num_columns == 1
    assert c2p.to_csv_bytes(table) == PTS


def test_no_trailing_newline_round_trips(tmp_path):
    for name, data in (("a_imu.csv", IMU_HEADER + IMU_ROWS[:-1]), ("a_pts.csv", PTS[:-1])):
        src = _write(tmp_path / name, data)
        out = src.with_suffix(".parquet")
        assert c2p.convert(src, out).ok
        assert c2p.to_csv_bytes(pq.read_table(out)) == data


def test_round_trip_across_many_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(c2p, "BLOCK", 256)
    monkeypatch.setattr(c2p, "BATCH_ROWS", 7)
    monkeypatch.setattr(c2p, "COMPARE_CHUNK", 13)
    rows = b"".join(b"%d,%.6f,%.6f,%.6f,%s\n" % (i, i * 0.001 - 1, -i * 1.5, i / 7, b"gyro" if i % 2 else b"accel")
                    for i in range(100))
    src = _write(tmp_path / "b_imu.csv", IMU_HEADER + rows)
    out = tmp_path / "b_imu.parquet"
    assert c2p.convert(src, out).ok
    with pq.ParquetFile(out) as pf:
        assert pf.metadata.num_row_groups > 1


def test_unreproducible_files_are_refused(tmp_path):
    bad = {
        "seven_imu.csv": IMU_HEADER + IMU_ROWS + b"1,0.1234567,0.000000,0.000000,gyro\n",
        "sci_imu.csv": IMU_HEADER + IMU_ROWS + b"1,1e-05,0.000000,0.000000,gyro\n",
        "crlf_imu.csv": IMU_HEADER + IMU_ROWS + b"1,0.123456,0.000000,0.000000,gyro\r\n",
        "header_imu.csv": b"timestamp_us,x,y,z,kind\n" + IMU_ROWS,
        "lead_pts.csv": b"timestamp_us\n5\n007\n",
    }
    for name, data in bad.items():
        src = _write(tmp_path / name, data)
        out = src.with_suffix(".parquet")
        r = c2p.convert(src, out)
        assert not r.ok and r.error, name
        assert not out.exists(), name
        assert src.read_bytes() == data, name
    src = tmp_path / "sci_imu.csv"
    assert c2p.main([str(src), "--apply"]) == 1
    assert src.exists() and not src.with_suffix(".parquet").exists()


def test_apply_deletes_only_after_success_beside_it(tmp_path, capsys):
    good = _write(tmp_path / "g_imu.csv", IMU_HEADER + IMU_ROWS)
    assert c2p.main([str(good), "--apply"]) == 0
    assert not good.exists() and good.with_suffix(".parquet").exists()
    assert "ok" in capsys.readouterr().out
    kept = _write(tmp_path / "k_pts.csv", PTS)
    other = tmp_path / "elsewhere"
    assert c2p.main([str(kept), "--apply", "--out", str(other)]) == 0
    assert kept.exists() and (other / "k_pts.parquet").exists()
    dry = _write(tmp_path / "d_pts.csv", PTS)
    assert c2p.main([str(dry)]) == 0
    assert dry.exists() and dry.with_suffix(".parquet").exists()


def test_tree_finds_recursively_and_mirrors_under_out(tmp_path, capsys):
    tree = tmp_path / "tree"
    _write(tree / "dev" / "rec" / "rec_imu.csv", IMU_HEADER + IMU_ROWS)
    _write(tree / "dev" / "rec" / "rec_audio_pts.csv", PTS)
    _write(tree / "top_pts.csv", PTS)
    _write(tree / "dev" / "rec" / "notes.csv", b"a,b\n1,2\n")
    out = tmp_path / "out"
    assert c2p.main(["--tree", str(tree), "--out", str(out)]) == 0
    found = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert found == ["dev/rec/rec_audio_pts.parquet", "dev/rec/rec_imu.parquet", "top_pts.parquet"]
    assert (tree / "dev" / "rec" / "rec_imu.csv").exists()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("total") and "3 files, 0 failed" in lines[-1]


def test_explicit_other_csv_is_skipped(tmp_path, capsys):
    other = _write(tmp_path / "notes.csv", b"a,b\n1,2\n")
    assert c2p.main([str(other)]) == 0
    assert "skipped" in capsys.readouterr().out
    assert not other.with_suffix(".parquet").exists()
