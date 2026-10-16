"""Calibration: the size-vs-VMAF curve over sample files and the chosen quality."""
from __future__ import annotations

import json

import pytest

from storage import calibrate
from storage.calibrate import Point, Summary, choose, summarise
from conftest import run_ffmpeg


def _summary(quality, mean_ratio, min_vmaf, all_ok=True):
    return Summary(codec="hevc", quality=quality, files=4, mean_ratio=mean_ratio,
                   min_ratio=mean_ratio - 0.05, max_ratio=mean_ratio + 0.05,
                   min_vmaf=min_vmaf, mean_vmaf=min_vmaf + 1, mean_fps=100.0, all_ok=all_ok)


def test_choose_smallest_ratio_above_floor():
    picked = choose([_summary(20, 0.90, 98.0), _summary(24, 0.65, 96.0),
                     _summary(28, 0.50, 94.0)])
    assert picked.quality == 24 and picked.meets_floor and picked.meets_target
    assert picked.mean_ratio == 0.65 and picked.min_vmaf == 96.0


def test_choose_ignores_failed_checks():
    picked = choose([_summary(20, 0.90, 98.0), _summary(24, 0.65, 96.0, all_ok=False)])
    assert picked.quality == 20 and picked.meets_floor
    assert not picked.meets_target


def test_choose_tie_prefers_lower_quality_number():
    assert choose([_summary(26, 0.60, 96.0), _summary(24, 0.60, 97.0)]).quality == 24


def test_choose_best_achievable_when_none_reaches_floor():
    picked = choose([_summary(20, 0.90, 93.0), _summary(24, 0.65, 91.0)])
    assert picked.quality == 20 and not picked.meets_floor and not picked.meets_target


def test_choose_target_rule():
    assert choose([_summary(24, 0.70, 96.0)]).meets_target
    assert not choose([_summary(24, 0.71, 96.0)]).meets_target
    assert not choose([_summary(24, 0.60, 96.0)], target=0.5).meets_target


def test_choose_empty():
    assert choose([]) is None


def _point(file, ratio, vmaf, fps, ok=True):
    return Point(file=file, codec="hevc", quality=28, encoder="software", size_bytes=100,
                 ratio=ratio, vmaf=vmaf, frames_ok=True, duration_ok=ok,
                 encode_seconds=1.0, encode_fps=fps)


def test_summarise_two_files():
    out = summarise([_point("a.mp4", 0.5, 96.0, 100.0), _point("b.mp4", 0.7, 98.0, 200.0)])
    assert list(out) == [("hevc", 28)]
    s = out[("hevc", 28)]
    assert s.files == 2 and s.all_ok
    assert s.mean_ratio == pytest.approx(0.6)
    assert (s.min_ratio, s.max_ratio) == (0.5, 0.7)
    assert s.min_vmaf == 96.0 and s.mean_vmaf == pytest.approx(97.0)
    assert s.mean_fps == pytest.approx(150.0)


def test_summarise_failed_check_marks_quality():
    out = summarise([_point("a.mp4", 0.5, 96.0, 100.0), _point("b.mp4", 0.7, 98.0, 1.0, ok=False)])
    assert not out[("hevc", 28)].all_ok


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("calib_inputs")
    for name, pattern in (("a.mp4", "testsrc"), ("b.mp4", "testsrc2")):
        r = run_ffmpeg(["-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"{pattern}=size=320x240:rate=10:duration=3",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        str(folder / name)])
        assert r.returncode == 0, r.stderr
    return folder


@pytest.fixture(scope="module")
def run_all(clips, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calib_run")
    args = [str(clips), "--codecs", "hevc", "--hevc-qualities", "24,32",
            "--encoder", "software", "--out", str(tmp / "curve"),
            "--workdir", str(tmp / "calib"), "--label", "test"]
    assert calibrate.main(args) == 0
    return tmp, args


def test_end_to_end(run_all):
    tmp, _ = run_all
    data = json.loads((tmp / "curve.json").read_text())
    assert data["label"] == "test" and len(data["inputs"]) == 2
    points = data["points"]
    assert len(points) == 4
    for p in points:
        assert p["ratio"] > 0 and p["vmaf"] > 0
        assert p["frames_ok"] and p["duration_ok"]
        assert p["encoder"] == "software" and p["encode_fps"] > 0
    size = {(p["file"], p["quality"]): p["size_bytes"] for p in points}
    for name in ("a.mp4", "b.mp4"):
        assert size[(name, 32)] < size[(name, 24)]
    assert data["choice"]["hevc"]["quality"] in (24, 32)
    md = (tmp / "curve.md").read_text(encoding="utf-8")
    assert "| 24 |" in md and "| 32 |" in md
    assert "Chosen HEVC quality" in md or "no quality reaches" in md


def test_measure_stage_reuses_encodes(run_all):
    tmp, args = run_all
    encodes = sorted((tmp / "calib").glob("*.mp4"))
    assert len(encodes) == 4
    mtimes = [p.stat().st_mtime_ns for p in encodes]
    (tmp / "curve.json").unlink()
    assert calibrate.main([*args, "--stage", "measure"]) == 0
    assert [p.stat().st_mtime_ns for p in encodes] == mtimes
    assert len(json.loads((tmp / "curve.json").read_text())["points"]) == 4


def test_encode_stage_only(clips, tmp_path):
    args = [str(clips), "--codecs", "hevc", "--hevc-qualities", "30",
            "--encoder", "software", "--out", str(tmp_path / "curve"),
            "--workdir", str(tmp_path / "calib"), "--stage", "encode"]
    assert calibrate.main(args) == 0
    for stem in ("a", "b"):
        assert (tmp_path / "calib" / f"{stem}.hevc.q30.mp4").is_file()
        meta = json.loads((tmp_path / "calib" / f"{stem}.hevc.q30.json").read_text())
        assert meta["encoder"] == "software" and meta["size_bytes"] > 0 and meta["seconds"] > 0
    assert not (tmp_path / "curve.json").exists()


def test_no_mp4_files(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    assert calibrate.main([str(tmp_path), "--out", str(tmp_path / "curve")]) == 2


def test_unknown_codec(clips, tmp_path):
    assert calibrate.main([str(clips), "--codecs", "vp9", "--out", str(tmp_path / "c")]) == 2
