"""The command line front end."""
from __future__ import annotations

import json
import shutil

import pytest

import cli
from faceblur.pipeline import STATUS_DONE

FAST = ["--det-sizes", "320", "--persist", "1", "--no-progress"]


@pytest.fixture
def batch(tmp_path, video_silent):
    """A folder holding three copies of the same short video."""
    folder = tmp_path / "batch"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        shutil.copy(video_silent, folder / name)
    return folder


def test_finds_only_video_files(batch):
    (batch / "notes.txt").write_text("hello")
    (batch / "picture.jpg").write_bytes(b"\xff\xd8\xff")
    assert [p.name for p in cli.find_videos(batch, recursive=False)] == \
        ["a.mp4", "b.mp4", "c.mp4"]


def test_a_single_file_input_gives_one_job(batch):
    assert cli.find_videos(batch / "a.mp4", recursive=False) == [batch / "a.mp4"]


def test_recursive_walks_subfolders(batch, video_silent):
    deep = batch / "day2" / "morning"
    deep.mkdir(parents=True)
    shutil.copy(video_silent, deep / "d.mp4")
    assert len(cli.find_videos(batch, recursive=False)) == 3
    assert len(cli.find_videos(batch, recursive=True)) == 4


def test_default_output_folder_sits_next_to_the_input(batch):
    """A folder gets a sibling folder. A single file gets its copy next to itself."""
    assert cli.default_output_dir(batch) == batch.parent / "batch_blurred"
    assert cli.default_output_dir(batch / "a.mp4") == batch


def test_a_single_file_writes_its_copy_beside_the_source(batch, tmp_path):
    code = cli.main([str(batch / "a.mp4"), "--workers", "1", *FAST])
    assert code == 0
    assert (batch / "a_blurred.mp4").exists()


def test_a_second_run_does_not_blur_its_own_copies(batch):
    """Copies land beside the sources here, so the second run must skip them."""
    cli.main([str(batch / "a.mp4"), "--workers", "1", *FAST])
    assert (batch / "a_blurred.mp4").exists()

    found = cli.find_videos(batch, recursive=False)
    kept = cli.drop_own_output(found, batch, batch, "_blurred")
    assert [p.name for p in kept] == ["a.mp4", "b.mp4", "c.mp4"]
    assert not (batch / "a_blurred_blurred.mp4").exists()


def test_copies_in_their_own_folder_are_left_alone(batch, tmp_path):
    """A separate output folder never holds sources, so nothing is dropped."""
    found = cli.find_videos(batch, recursive=False)
    assert cli.drop_own_output(found, tmp_path / "out", batch, "_blurred") == found


def test_a_flat_run_puts_every_copy_in_one_folder(batch, tmp_path):
    src = batch / "day2" / "a.mp4"
    dst = cli.destination(src, batch, tmp_path / "out", recursive=False, suffix="_blurred")
    assert dst == tmp_path / "out" / "a_blurred.mp4"


def test_a_recursive_run_mirrors_the_folder_structure(batch, tmp_path):
    src = batch / "day2" / "morning" / "a.mp4"
    dst = cli.destination(src, batch, tmp_path / "out", recursive=True, suffix="_blurred")
    assert dst == tmp_path / "out" / "day2" / "morning" / "a_blurred.mp4"


def test_a_folder_run_writes_a_copy_and_a_sidecar_for_each(batch, tmp_path, capsys):
    out = tmp_path / "out"
    code = cli.main([str(batch), "-o", str(out), "--workers", "1", *FAST])
    assert code == 0
    assert sorted(p.name for p in out.glob("*.mp4")) == \
        ["a_blurred.mp4", "b_blurred.mp4", "c_blurred.mp4"]
    assert len(list(out.glob("*.mp4.json"))) == 3
    assert "3 videos done. 0 failed." in capsys.readouterr().out


def test_a_broken_file_fails_the_run_but_not_the_batch(batch, tmp_path, capsys):
    (batch / "b.mp4").write_bytes(b"not a video")
    out = tmp_path / "out"
    code = cli.main([str(batch), "-o", str(out), "--workers", "1", *FAST])

    assert code == 1
    assert len(list(out.glob("*.mp4"))) == 2
    printed = capsys.readouterr().out
    assert "Failed: Could not read this video" in printed
    assert "2 videos done. 1 failed." in printed


def test_the_report_holds_one_record_per_video(batch, tmp_path):
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    cli.main([str(batch), "-o", str(out), "--workers", "1",
              "--report", str(report), *FAST])

    records = json.loads(report.read_text(encoding="utf-8"))
    assert len(records) == 3
    assert {r["status"] for r in records} == {STATUS_DONE}
    assert all(r["model_sha256"]["yunet"].startswith("8f2383e4") for r in records)


def test_running_twice_skips_the_copies_that_exist(batch, tmp_path, capsys):
    out = tmp_path / "out"
    cli.main([str(batch), "-o", str(out), "--workers", "1", *FAST])
    capsys.readouterr()

    code = cli.main([str(batch), "-o", str(out), "--workers", "1", *FAST])
    assert code == 0
    assert "3 already done." in capsys.readouterr().out


def test_a_missing_input_reports_and_stops(tmp_path, capsys):
    assert cli.main([str(tmp_path / "nope.mp4"), *FAST]) == 2
    assert "could not find" in capsys.readouterr().err


def test_a_folder_with_no_videos_reports_and_stops(tmp_path, capsys):
    assert cli.main([str(tmp_path), *FAST]) == 2
    assert "found no videos" in capsys.readouterr().err


def test_a_setting_outside_its_range_reports_and_stops(batch, capsys):
    assert cli.main([str(batch), "--conf", "5", *FAST]) == 2
    assert "conf must be" in capsys.readouterr().err


def test_default_workers_is_half_the_cpu_count(monkeypatch):
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 8)
    assert cli.default_workers() == 4
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 1)
    assert cli.default_workers() == 1


def test_every_flag_the_plan_lists_is_accepted(batch, tmp_path):
    args = cli.build_parser().parse_args([
        str(batch), "-o", str(tmp_path), "--engine", "both", "--conf", "0.3",
        "--det-sizes", "640,1280", "--stride", "2", "--persist", "8",
        "--pad", "0.4", "--mode", "pixelate", "--workers", "3",
        "--report", "r.json", "--no-progress", "--recursive",
    ])
    assert args.engine == "both"
    assert args.mode == "pixelate"
    assert args.workers == 3
    assert args.recursive is True
    assert args.progress is False
