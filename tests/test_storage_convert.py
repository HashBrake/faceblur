"""The converter: dry run by default, replace only with --apply and every check passed."""
from __future__ import annotations

import shutil

import pytest

from storage import convert, manifest
from storage.checks import Check, CheckReport
from storage.probe import probe
from conftest import audio_codec

# libx265 at CRF 20 runs anywhere. The floor is lowered from 95 because the
# small synthetic clip scores below 99 even against itself (see storage/ffmpeg.py).
OPTS = ["--encoder", "software", "--quality", "20", "--floor", "90"]


def _copy(src, tmp_path, name="clip.mp4"):
    dst = tmp_path / name
    shutil.copy(src, dst)
    return dst


def _leftovers(folder):
    return sorted(p.name for p in folder.iterdir() if ".tmp" in p.name or p.suffix == ".part")


def test_dry_run(tmp_path, video_long):
    src = _copy(video_long, tmp_path)
    sha = manifest.sha256_of(src)
    assert convert.main([str(src), *OPTS]) == 0
    assert manifest.sha256_of(src) == sha
    assert _leftovers(tmp_path) == []
    rows = manifest.read_rows(tmp_path / "manifest.parquet")
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "dry-run" and row.checks_passed
    assert row.vmaf >= 90 and row.vmaf_min is not None
    assert row.codec_before == "h264" and row.codec_after == "hevc"
    assert row.size_after > 0 and row.frames == 120
    assert row.sha256_before == sha and row.sha256_after != sha
    assert row.encoder == "software" and row.quality == 20
    assert row.path == src.name  # beside its manifest: the bare name


def test_convert_returns_row(tmp_path, video_silent):
    src = _copy(video_silent, tmp_path)
    lines = []
    row, report = convert.convert(src, encoder="software", quality=20, floor=90,
                                  manifest=tmp_path / "m.parquet", log=lines.append)
    assert row.status == "dry-run" and report.passed
    assert manifest.read_rows(tmp_path / "m.parquet") == [row]
    assert not (tmp_path / "manifest.parquet").exists()
    assert "VMAF" in lines[-1]


def test_apply(tmp_path, video_long):
    src = _copy(video_long, tmp_path)
    assert convert.main([str(src), "--apply", *OPTS]) == 0
    facts = probe(src)
    assert facts.codec == "hevc" and facts.frames == 120
    row = manifest.read_rows(tmp_path / "manifest.parquet")[0]
    assert row.status == "applied"
    assert row.sha256_after == manifest.sha256_of(src)
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("apply", [False, True])
def test_failed_check(tmp_path, video_silent, monkeypatch, apply):
    src = _copy(video_silent, tmp_path)
    sha = manifest.sha256_of(src)
    monkeypatch.setattr(convert.checks, "run_all", lambda *a, **k: CheckReport(
        checks=[Check("frame_count", False, "forced")], frames=19, duration_s=1.9,
        vmaf_mean=None, vmaf_min=None))
    argv = [str(src), *OPTS] + (["--apply"] if apply else [])
    assert convert.main(argv) == 1
    assert manifest.sha256_of(src) == sha and probe(src).codec == "h264"
    assert _leftovers(tmp_path) == []
    row = manifest.read_rows(tmp_path / "manifest.parquet")[0]
    assert row.status == "failed" and not row.checks_passed
    assert row.sha256_after == sha and row.vmaf is None


def test_vmaf_floor_fails(tmp_path, video_silent):
    src = _copy(video_silent, tmp_path)
    assert convert.main([str(src), "--apply", "--encoder", "software", "--quality", "20",
                         "--floor", "101"]) == 1
    assert probe(src).codec == "h264"
    assert _leftovers(tmp_path) == []
    assert manifest.read_rows(tmp_path / "manifest.parquet")[0].status == "failed"


def test_keep(tmp_path, video_silent):
    src = _copy(video_silent, tmp_path)
    assert convert.main([str(src), "--keep", *OPTS]) == 0
    kept = tmp_path / "clip.hevc.tmp.mp4"
    assert kept.exists() and probe(kept).codec == "hevc"
    assert probe(src).codec == "h264"


def test_same_codec_refused(tmp_path, video_silent):
    src = _copy(video_silent, tmp_path)
    assert convert.main([str(src), "--apply", *OPTS]) == 0
    (tmp_path / "manifest.parquet").unlink()
    assert convert.main([str(src), *OPTS]) == 2
    assert not (tmp_path / "manifest.parquet").exists()


def test_bad_input(tmp_path):
    assert convert.main([str(tmp_path / "missing.mp4")]) == 2
    assert convert.main([str(tmp_path / "x.mp4"), "--codec", "vp9"]) == 2


def test_encode_error_cleans_up(tmp_path, video_silent, monkeypatch):
    src = _copy(video_silent, tmp_path)

    def broken(src_, dst, **kwargs):
        (dst.parent / (dst.name + ".part")).write_bytes(b"half")
        dst.write_bytes(b"half")
        raise RuntimeError("boom")

    monkeypatch.setattr(convert, "encode_file", broken)
    with pytest.raises(RuntimeError):
        convert.convert(src, encoder="software")
    assert _leftovers(tmp_path) == []


def test_audio_survives(tmp_path, video_with_audio):
    src = _copy(video_with_audio, tmp_path)
    assert convert.main([str(src), "--apply", *OPTS]) == 0
    assert probe(src).codec == "hevc"
    assert audio_codec(src) == "aac"


def test_default_quality():
    s = convert.Settings()
    assert convert.default_quality("hevc", "nvenc") == s.hevc_cq == 28
    assert convert.default_quality("hevc", "software") == s.hevc_crf
    assert convert.default_quality("av1", "nvenc") == 24


def test_software_default_is_hevc_crf(tmp_path, video_silent, monkeypatch):
    """`auto` is resolved once, and the quality and the encoder used agree."""
    src = _copy(video_silent, tmp_path)
    monkeypatch.setattr(convert, "resolve_encoder", lambda codec, encoder: "software")
    seen = {}
    real = convert.encode_file

    def spy(src_, dst, **kwargs):
        seen.update(kwargs)
        return real(src_, dst, **kwargs)

    monkeypatch.setattr(convert, "encode_file", spy)
    row, _ = convert.convert(src, floor=90, manifest=tmp_path / "m.parquet", log=lambda _: None)
    assert seen["encoder"] == "software" and seen["gop"] == convert.Settings().gop == 60
    assert row.encoder == "software" and row.quality == convert.Settings().hevc_crf


def test_nvenc_default_is_hevc_cq(tmp_path, video_silent, monkeypatch):
    src = _copy(video_silent, tmp_path)
    monkeypatch.setattr(convert, "resolve_encoder", lambda codec, encoder: "nvenc")
    seen = {}

    def stop(src_, dst, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop before encoding")

    monkeypatch.setattr(convert, "encode_file", stop)
    with pytest.raises(RuntimeError):
        convert.convert(src, manifest=tmp_path / "m.parquet")
    assert seen["encoder"] == "nvenc" and seen["quality"] == convert.Settings().hevc_cq


def test_failed_rename_writes_no_row(tmp_path, video_silent, monkeypatch):
    src = _copy(video_silent, tmp_path)
    data = src.read_bytes()

    def refuse(a, b):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(convert.os, "replace", refuse)
    with pytest.raises(PermissionError):
        convert.convert(src, apply=True, encoder="software", quality=20, floor=90,
                        log=lambda _: None)
    assert src.read_bytes() == data
    assert _leftovers(tmp_path) == []
    assert manifest.read_rows(tmp_path / "manifest.parquet") == []


def test_row_path_relative_or_absolute(tmp_path, video_silent):
    (tmp_path / "rec").mkdir()
    src = _copy(video_silent, tmp_path / "rec")
    row, _ = convert.convert(src, encoder="software", quality=20, floor=90, log=lambda _: None)
    assert row.path == "clip.mp4"
    assert manifest.read_rows(tmp_path / "rec" / "manifest.parquet") == [row]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    row, _ = convert.convert(src, encoder="software", quality=20, floor=90,
                             manifest=elsewhere / "m.parquet", log=lambda _: None)
    assert row.path == str(src)


def test_windows_must_be_positive(tmp_path, video_silent):
    src = _copy(video_silent, tmp_path)
    assert convert.main([str(src), *OPTS, "--windows", "0"]) == 2
    assert not (tmp_path / "manifest.parquet").exists()


def test_os_error_is_a_message(tmp_path, video_silent, monkeypatch, capsys):
    src = _copy(video_silent, tmp_path)

    def denied(*a, **k):
        raise PermissionError(5, "Access is denied", str(src))

    monkeypatch.setattr(convert, "convert", denied)
    assert convert.main([str(src), *OPTS]) == 2
    err = capsys.readouterr().err
    assert "Access is denied" in err and "Traceback" not in err
