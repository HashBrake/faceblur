"""The command line front end."""
from __future__ import annotations

import json
import shutil

import pytest

import cli
from faceblur import watchdog
from faceblur.pipeline import STATUS_DONE, STATUS_SKIPPED, AuditRecord

FAST = ["--det-sizes", "320", "--no-verify", "--min-track", "1", "--no-progress"]


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
    assert all(r["model_sha256"]["yunet"].startswith("c0aa2a66") for r in records)


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
        "--det-sizes", "640,1280", "--stride", "2", "--min-track", "2", "--max-gap", "4", "--tail", "1", "--no-verify", "--max-face", "0.3",
        "--pad", "0.4", "--mode", "pixelate", "--workers", "3",
        "--report", "r.json", "--no-progress", "--recursive",
    ])
    assert args.engine == "both"
    assert args.mode == "pixelate"
    assert args.workers == 3
    assert args.recursive is True
    assert args.progress is False


@pytest.fixture
def paths(monkeypatch):
    """Records which way main ran the batch, without running it."""
    taken = []

    def supervise(entry, src, dst, settings, workers=1, on_progress=None):
        taken.append(("supervise", workers, settings))
        return AuditRecord(source=src.name, output=dst.name, status=STATUS_SKIPPED)

    monkeypatch.setattr(watchdog, "supervise", supervise)
    monkeypatch.setattr(cli, "run_serial",
                        lambda jobs, settings, reporter: taken.append(("serial",)) or [])
    monkeypatch.setattr(cli, "run_parallel",
                        lambda jobs, settings, workers, reporter:
                        taken.append(("parallel", workers)) or [])
    return taken


@pytest.mark.parametrize("workers", ["1", "3"])
def test_by_default_every_video_runs_under_the_watchdog(batch, tmp_path, paths, workers):
    code = cli.main([str(batch), "-o", str(tmp_path / "out"), "--workers", workers, *FAST])

    assert code == 0
    assert [p[:2] for p in paths] == [("supervise", int(workers))] * 3


def test_a_watchdog_scale_of_0_runs_in_this_process_as_before(batch, tmp_path, paths):
    cli.main([str(batch), "-o", str(tmp_path / "out"), "--workers", "1",
              "--watchdog-scale", "0", *FAST])
    cli.main([str(batch), "-o", str(tmp_path / "out"), "--workers", "3",
              "--watchdog-scale", "0", *FAST])

    assert paths == [("serial",), ("parallel", 3)]


def test_the_watchdog_scale_multiplies_every_limit(batch, tmp_path, paths):
    cli.main([str(batch / "a.mp4"), "-o", str(tmp_path / "out"), "--workers", "1",
              "--watchdog-scale", "0.5", *FAST])

    settings = paths[0][2]
    assert settings.watchdog_min_seconds == 300.0
    assert dict(settings.watchdog_seconds_per_frame) == \
        {"detecting": 0.5, "tracking": 0.025, "writing": 0.5, "checking": 0.5}


def test_a_negative_watchdog_scale_reports_and_stops(batch, capsys):
    assert cli.main([str(batch), "--watchdog-scale", "-1", *FAST]) == 2
    assert "--watchdog-scale must be 0 or more" in capsys.readouterr().err

def _settings_of_a_run(tmp_path, monkeypatch, *flags):
    """The Settings main() hands to the run, without running anything."""
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"")
    seen = {}

    def fake_run_serial(jobs, settings, reporter):
        seen["settings"] = settings
        return []

    monkeypatch.setattr(cli, "run_serial", fake_run_serial)
    # In this process, so that run_serial sees the settings (the watchdog,
    # on by default, would hand the video to a child process instead).
    assert cli.main([str(folder), "-o", str(tmp_path / "out"), "--workers", "1",
                     "--no-progress", "--watchdog-scale", "0", *flags]) == 0
    return seen["settings"]


def test_the_new_switches_default_to_the_settings(tmp_path, monkeypatch):
    defaults = cli.Settings()
    args = cli.build_parser().parse_args(["x"])
    assert args.second_chance_scope == defaults.second_chance_scope
    assert args.second_chance_margin == defaults.second_chance_margin_seconds
    assert args.check_overlap == defaults.check_overlap
    assert args.adaptive_stride == defaults.adaptive_stride
    assert args.adaptive_stride_max == defaults.adaptive_stride_max
    assert args.encode_gop == defaults.encode_gop_seconds
    settings = _settings_of_a_run(tmp_path, monkeypatch)
    for name in ("second_chance_scope", "second_chance_margin_seconds", "check_overlap",
                 "adaptive_stride", "adaptive_stride_max", "encode_gop_seconds"):
        assert getattr(settings, name) == getattr(defaults, name)


def test_the_new_switches_reach_the_settings_of_the_run(tmp_path, monkeypatch):
    settings = _settings_of_a_run(
        tmp_path, monkeypatch,
        "--second-chance", "1", "--second-chance-scope", "all",
        "--second-chance-margin", "1.25", "--no-check-overlap",
        "--no-adaptive-stride", "--adaptive-stride-max", "3", "--encode-gop", "2.5")
    assert settings.second_chance_scope == "all"
    assert settings.second_chance_margin_seconds == 1.25
    assert settings.check_overlap is False
    assert settings.adaptive_stride is False
    assert settings.adaptive_stride_max == 3
    assert settings.encode_gop_seconds == 2.5
    # The audit record's settings dict is this.
    recorded = settings.to_dict()
    assert recorded["second_chance_scope"] == "all"
    assert recorded["second_chance_margin_seconds"] == 1.25
    assert recorded["check_overlap"] is False
    assert recorded["adaptive_stride"] is False
    assert recorded["adaptive_stride_max"] == 3
    assert recorded["encode_gop_seconds"] == 2.5


def test_adaptive_stride_can_be_turned_on_explicitly():
    args = cli.build_parser().parse_args(["x", "--adaptive-stride"])
    assert args.adaptive_stride is True
    args = cli.build_parser().parse_args(["x", "--no-adaptive-stride"])
    assert args.adaptive_stride is False


def test_an_unknown_second_chance_scope_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["x", "--second-chance-scope", "bogus"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--second-chance-scope", "--second-chance-margin", "--no-check-overlap",
                 "--adaptive-stride", "--no-adaptive-stride", "--adaptive-stride-max",
                 "--encode-gop"):
        assert flag in out

@pytest.mark.parametrize("encoder", ["auto", "x264"])
def test_codec_hevc_writes_an_hevc_copy_that_verifies(video_silent, tmp_path, encoder):
    """`--codec hevc` writes the whole copy in HEVC. With `--encoder x264`, the
    CPU fallback, the encoder is libx265: still HEVC, never H.264."""
    from faceblur import segments
    from faceblur.settings import Settings
    from faceblur.verify import check
    from storage.probe import probe

    out_dir = tmp_path / "out"
    code = cli.main([str(video_silent), "-o", str(out_dir), "--workers", "1",
                     "--codec", "hevc", "--encoder", encoder, *FAST])
    assert code == 0
    out = out_dir / "silent_blurred.mp4"
    facts = probe(out)
    assert facts.codec == "hevc"
    assert facts.frames == 20
    assert segments.verify(out, 20, 1.9) == ""
    report = check(video_silent, out, Settings(device="cpu", det_sizes=(320,), verify=False))
    assert report["residual_faces"] == 0
    assert report["frames_checked"] == 20
    record = json.loads((out_dir / "silent_blurred.mp4.json").read_text(encoding="utf-8"))
    assert record["settings"]["codec"] == "hevc"


def test_the_default_codec_is_still_h264(video_silent, tmp_path):
    from storage.probe import probe

    out_dir = tmp_path / "out"
    assert cli.main([str(video_silent), "-o", str(out_dir), "--workers", "1", *FAST]) == 0
    assert probe(out_dir / "silent_blurred.mp4").codec == "h264"
    record = json.loads((out_dir / "silent_blurred.mp4.json").read_text(encoding="utf-8"))
    assert record["settings"]["codec"] == "h264"


def test_the_codec_flag_is_accepted(batch, tmp_path):
    parser = cli.build_parser()
    assert parser.parse_args([str(batch), "-o", str(tmp_path)]).codec == "h264"
    assert parser.parse_args([str(batch), "-o", str(tmp_path), "--codec", "hevc"]).codec == "hevc"
    with pytest.raises(SystemExit):
        parser.parse_args([str(batch), "-o", str(tmp_path), "--codec", "av1"])
