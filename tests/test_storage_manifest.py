"""The Parquet manifest: round trip, atomic write, tree listing, hashing, CLI."""
from __future__ import annotations

import hashlib
import shutil

import pyarrow.parquet as pq

from storage import manifest
from storage.manifest import ManifestRow, SCHEMA


def _row(path="a.mp4", vmaf=None, vmaf_min=None, quality=None, status="dry-run"):
    return ManifestRow(path=path, sha256_before="ab" * 32, sha256_after="cd" * 32,
                       size_before=1000, size_after=400, codec_before="h264",
                       codec_after="hevc", frames=120, duration_s=12.0, vmaf=vmaf,
                       vmaf_min=vmaf_min, checks_passed=vmaf is not None, status=status,
                       encoder="software", quality=quality,
                       converted_at="2026-10-16T08:00:00+00:00")


def test_append_round_trip(tmp_path):
    out = tmp_path / "manifest.parquet"
    first = _row("a.mp4")
    second = _row("b/c.mp4", vmaf=97.25, vmaf_min=96.5, quality=24, status="applied")
    manifest.append_row(out, first)
    manifest.append_row(out, second)
    assert manifest.read_rows(out) == [first, second]
    assert not (tmp_path / "manifest.parquet.part").exists()
    assert pq.read_schema(out).remove_metadata() == SCHEMA


def test_read_rows_missing(tmp_path):
    assert manifest.read_rows(tmp_path / "none.parquet") == []


def test_sha256_of(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello hot storage" * 1000)
    assert manifest.sha256_of(f, chunk=4096) == hashlib.sha256(f.read_bytes()).hexdigest()


def _tree(tmp_path, video_silent):
    root = tmp_path / "tree"
    (root / "day1").mkdir(parents=True)
    shutil.copy(video_silent, root / "day1" / "cam.mp4")
    (root / "notes.txt").write_text("hello")
    (root / "manifest.parquet").write_bytes(b"old")
    return root


def test_build_tree_manifest(tmp_path, video_silent):
    root = _tree(tmp_path, video_silent)
    out = manifest.build_tree_manifest(root)
    assert out == root / "manifest.parquet"
    rows = manifest.read_rows(out)
    assert [r.path for r in rows] == ["day1/cam.mp4", "notes.txt"]
    video, text = rows
    assert video.codec_before == video.codec_after == "h264"
    assert video.frames == 20
    assert video.duration_s > 1.9
    assert video.status == "listed"
    assert video.sha256_before == video.sha256_after == manifest.sha256_of(
        root / "day1" / "cam.mp4")
    assert video.size_before == video.size_after == (root / "day1" / "cam.mp4").stat().st_size
    assert text.codec_before == "" and text.frames == 0 and text.duration_s == 0.0
    assert text.vmaf is None and text.quality is None and not text.checks_passed


def test_unreadable_video_is_listed(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "broken.mp4").write_bytes(b"not a video")
    rows = manifest.read_rows(manifest.build_tree_manifest(root))
    assert [(r.path, r.codec_before, r.frames) for r in rows] == [("broken.mp4", "unreadable", 0)]


def test_cli(tmp_path, video_silent, capsys):
    root = _tree(tmp_path, video_silent)
    assert manifest.main([str(root)]) == 0
    assert len(manifest.read_rows(root / "manifest.parquet")) == 2
    assert "2 files" in capsys.readouterr().out
    other = tmp_path / "elsewhere.parquet"
    assert manifest.main([str(root), "-o", str(other)]) == 0
    assert len(manifest.read_rows(other)) == 3  # the old manifest is now just a file


def test_rewrite_after_read(tmp_path):
    """Read then replace, over and over: no handle may stay open (Windows rename)."""
    out = tmp_path / "manifest.parquet"
    for i in range(5):
        manifest.append_row(out, _row(f"{i}.mp4"))
        assert len(manifest.read_rows(out)) == i + 1
    manifest.write_rows(out, manifest.read_rows(out)[:2])
    assert [r.path for r in manifest.read_rows(out)] == ["0.mp4", "1.mp4"]


def test_tree_manifest_over_one_just_read(tmp_path, video_silent):
    root = _tree(tmp_path, video_silent)
    out = manifest.build_tree_manifest(root)
    assert len(manifest.read_rows(out)) == 2
    manifest.build_tree_manifest(root)
    assert len(manifest.read_rows(out)) == 2


def test_replace_retries_a_held_file(tmp_path, monkeypatch):
    calls = []
    real = manifest.os.replace

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(manifest.os, "replace", flaky)
    monkeypatch.setattr(manifest.time, "sleep", lambda s: None)
    manifest.write_rows(tmp_path / "m.parquet", [_row()])
    assert len(calls) == 3 and manifest.read_rows(tmp_path / "m.parquet") == [_row()]


def test_tree_listing_keeps_converter_rows(tmp_path, video_silent):
    from storage import convert
    root = tmp_path / "tree"
    root.mkdir()
    shutil.copy(video_silent, root / "clip.mp4")
    (root / "notes.txt").write_text("hello")
    applied, _ = convert.convert(root / "clip.mp4", apply=True, encoder="software",
                                 quality=20, floor=90, log=lambda _: None)
    assert applied.status == "applied" and applied.path == "clip.mp4"
    for _ in range(2):
        rows = manifest.read_rows(manifest.build_tree_manifest(root))
        assert rows[0] == applied
        listed = rows[1:]
        assert [(r.path, r.status) for r in listed] == [("clip.mp4", "listed"),
                                                        ("notes.txt", "listed")]
        assert listed[0].sha256_before == applied.sha256_after
        assert listed[0].codec_before == "hevc"
