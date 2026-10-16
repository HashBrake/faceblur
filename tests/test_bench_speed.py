"""The speed benchmark's arithmetic, resume rules and report shape, on fake records.

None of this needs the footage or the GPU: the steps that do are thin shells
around these functions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval import bench_speed as bench

REPO = Path(__file__).resolve().parents[1]
STEMS = ["a", "b"]


def entry(wall=None, faces=None, recall=None):
    return {"wall_seconds": wall, "recall": recall,
            "verify": None if faces is None else {"residual_by_size": {"40+ px": faces}}}


# ---------------------------------------------------------------- gates

def test_thirty_percent_less_wall_passes_and_the_threshold_is_inclusive():
    assert bench.gate(entry(wall=100), entry(wall=70))["wall_ok"] is True
    # 1 - 7/10 is 0.29999999999999993 in floats; it is still 30 percent.
    assert bench.gate(entry(wall=10), entry(wall=7))["wall_ok"] is True
    g = bench.gate(entry(wall=100), entry(wall=70.1))
    assert g["wall_ok"] is False and g["wall_reduction"] == pytest.approx(0.299)
    assert bench.gate(entry(wall=100), entry(wall=120))["wall_reduction"] == pytest.approx(-0.2)


def test_faces_of_40_px_may_not_rise():
    assert bench.gate(entry(faces=3), entry(faces=3))["faces40_ok"] is True
    assert bench.gate(entry(faces=3), entry(faces=0))["faces40_ok"] is True
    g = bench.gate(entry(faces=3), entry(faces=4))
    assert g["faces40_ok"] is False
    assert (g["faces40_baseline"], g["faces40_branch"]) == (3, 4)


def test_recall_may_move_half_a_point_either_way():
    assert bench.gate(entry(recall=0.950), entry(recall=0.955))["recall_ok"] is True
    assert bench.gate(entry(recall=0.950), entry(recall=0.945))["recall_ok"] is True
    assert bench.gate(entry(recall=0.950), entry(recall=0.9449))["recall_ok"] is False
    assert bench.gate(entry(recall=0.950), entry(recall=0.9551))["recall_ok"] is False


def test_a_missing_side_or_no_consensus_leaves_the_gate_null():
    g = bench.gate(entry(wall=100, faces=1, recall=0.9), None)
    assert g["wall_ok"] is None and g["wall_reduction"] is None
    assert g["faces40_ok"] is None and g["recall_ok"] is None
    assert g["faces40_baseline"] == 1 and g["recall_baseline"] == 0.9
    g = bench.gate(entry(wall=100, faces=0), entry(wall=60, faces=0))
    assert g["recall_ok"] is None and g["wall_ok"] is True and g["faces40_ok"] is True


def test_all_pass_needs_every_file_and_ignores_files_without_consensus():
    ok = bench.gate(entry(100, 1, 0.9), entry(60, 1, 0.9))
    no_labels = bench.gate(entry(100, 0), entry(60, 0))
    slow = bench.gate(entry(100, 1, 0.9), entry(80, 1, 0.9))
    pending = bench.gate(entry(100, 1, 0.9), None)
    assert bench.all_pass({"a": ok, "b": no_labels}) is True
    assert bench.all_pass({"a": ok, "b": slow}) is False
    assert bench.all_pass({"a": ok, "b": pending}) is None
    assert bench.all_pass({}) is None


# ---------------------------------------------------------------- resume

def write_sidecar(out: Path, stem: str, quarantined=False, status="done", **extra):
    copy = (out / "quarantine" if quarantined else out) / f"{stem}_blurred.mp4"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(b"")
    Path(str(copy) + ".json").write_text(
        json.dumps({"status": status, "quarantined": quarantined, **extra}), encoding="utf-8")
    return copy


def test_resume_finds_delivered_and_quarantined_copies(tmp_path):
    assert bench.missing(tmp_path, STEMS) == STEMS
    a = write_sidecar(tmp_path, "a")
    b = write_sidecar(tmp_path, "b", quarantined=True)
    assert bench.missing(tmp_path, STEMS) == []
    assert bench.copy_path(tmp_path, "a") == a
    assert bench.copy_path(tmp_path, "b") == b


def test_a_copy_without_its_sidecar_or_not_done_is_run_again(tmp_path):
    (tmp_path / "a_blurred.mp4").write_bytes(b"")      # killed before the sidecar
    write_sidecar(tmp_path, "b", status="failed")
    assert bench.copy_path(tmp_path, "a") is None
    assert bench.missing(tmp_path, STEMS) == ["a", "b"]


def test_a_stored_recall_from_other_settings_is_scored_again(tmp_path):
    path = tmp_path / "a.json"
    assert not bench.recall_current(path, {"conf": 0.5})
    path.write_text(json.dumps({"recall": 0.9, "settings": {"conf": 0.5, "det_sizes": [640]}}),
                    encoding="utf-8")
    assert bench.recall_current(path, {"conf": 0.5, "det_sizes": (640,)})
    assert not bench.recall_current(path, {"conf": 0.6, "det_sizes": (640,)})


# ---------------------------------------------------------------- commands

def test_the_production_command_goes_through_the_lock_with_the_production_flags(tmp_path):
    cmd = bench.timed_command(bench.cli_command(tmp_path, 4), tmp_path / "run.json")
    assert cmd[:4] == ["py", "-3.12", str(bench.GPU_LOCK), "--"]
    cli = cmd[cmd.index("cli.py") - 1:]
    assert cli[0] == sys.executable
    assert cli[cli.index("--workers") + 1] == "4"
    assert "--quarantine" in cli
    assert cli[cli.index("--second-chance") + 1] == "1"
    assert cli[cli.index("--report") + 1] == str(tmp_path / "report.json")
    assert "--no-progress" in cli
    assert cli[cli.index("-o") + 1] == str(tmp_path)


def test_the_verify_command_is_the_full_check():
    cmd = bench.locked(bench.verify_command(Path("s.mp4"), Path("c.mp4"), Path("r.json"), 4))
    assert cmd[:4] == ["py", "-3.12", str(bench.GPU_LOCK), "--"]
    assert cmd[4:7] == [sys.executable, "-m", "faceblur.verify"]
    assert "--stride" not in cmd
    assert cmd[cmd.index("--json") + 1] == "r.json"


def test_timed_writes_the_wall_and_exit_code(tmp_path):
    run = tmp_path / "run.json"
    code = bench.timed(run, [sys.executable, "-c", "raise SystemExit(3)"])
    data = json.loads(run.read_text(encoding="utf-8"))
    assert code == 3 and data["exit_code"] == 3
    assert data["total_wall_seconds"] >= 0
    assert "nvidia_smi_before" in data


# ---------------------------------------------------------------- report

def fake_side(root: Path, build: str, walls: dict, faces: dict, recalls: dict):
    out = root / build
    for stem, wall in walls.items():
        write_sidecar(out, stem, quarantined=True, wall_seconds=wall, detect_seconds=1.0,
                      encode_seconds=2.0, check_seconds=3.0, second_chance_seconds=4.0,
                      second_chance_rounds=1, frames=10, residual_faces=5,
                      residual_by_size={"40+ px": 1, "24-40 px": 2, "under 24 px": 2},
                      settings={"conf": 0.5})
    for stem, n in faces.items():
        (out / "verify").mkdir(parents=True, exist_ok=True)
        (out / "verify" / f"{stem}.json").write_text(json.dumps({
            "residual_faces": n, "residual_by_size": {"40+ px": n}, "residual_max_px": 50,
            "frames_checked": 10, "residual_list": []}), encoding="utf-8")
    for stem, r in recalls.items():
        (root / "recall" / build).mkdir(parents=True, exist_ok=True)
        (root / "recall" / build / f"{stem}.json").write_text(json.dumps({
            "recall": r, "recall_by_size": {}, "exposed_40": 0, "exposed_24_40": 0,
            "continuity": 0.9, "masked_mean": 0.01}), encoding="utf-8")
    (out / "runs.json").write_text(json.dumps({"runs": [
        {"nvidia_smi_before": "1 MiB, 8 MiB, 0 %", "total_wall_seconds": 30.0},
        {"nvidia_smi_before": "2 MiB, 8 MiB, 0 %", "total_wall_seconds": 12.5}]}),
        encoding="utf-8")


def test_the_report_with_the_baseline_side_only(tmp_path):
    fake_side(tmp_path, "baseline", {"a": 100.0, "b": 50.0}, {"a": 2, "b": 0},
              {"a": 0.9, "b": None})
    builds = {b: bench.collect(b, STEMS, tmp_path) for b in bench.BUILDS}
    assert builds["branch"] is None
    report = bench.assemble("2026-10-16", "abc", 4, STEMS, builds)
    json.dumps(report)                                     # serialisable
    assert report["baseline_commit"] == bench.BASELINE_COMMIT
    assert report["verify_code"] == "baseline 3eb0100"
    assert "--second-chance 1" in report["command"] and "--workers 4" in report["command"]
    base = report["builds"]["baseline"]
    assert base["settings"] == {"conf": 0.5}
    assert base["nvidia_smi_before"] == "1 MiB, 8 MiB, 0 %"
    assert base["total_wall_seconds"] == 42.5 and base["runs"] == 2
    fa = base["files"]["a"]
    assert fa["wall_seconds"] == 100.0 and fa["check_seconds"] == 3.0
    assert fa["quarantined"] is True and fa["second_chance_rounds"] == 1
    assert fa["record_residual_by_size"]["40+ px"] == 1
    assert fa["verify"] == {"residual_faces": 2, "residual_by_size": {"40+ px": 2},
                            "residual_max_px": 50, "frames_checked": 10}
    assert fa["recall"] == 0.9 and fa["masked_mean"] == 0.01
    assert base["files"]["b"]["recall"] is None
    assert report["gates"]["a"]["faces40_baseline"] == 2
    assert report["gates"]["a"]["wall_ok"] is None
    assert report["all_pass"] is None


def test_the_report_with_both_sides(tmp_path):
    fake_side(tmp_path, "baseline", {"a": 100.0, "b": 50.0}, {"a": 2, "b": 0},
              {"a": 0.9, "b": None})
    fake_side(tmp_path, "branch", {"a": 60.0, "b": 35.0}, {"a": 2, "b": 0},
              {"a": 0.903, "b": None})
    builds = {b: bench.collect(b, STEMS, tmp_path) for b in bench.BUILDS}
    report = bench.assemble("2026-10-16", "abc", 4, STEMS, builds)
    assert report["gates"]["a"]["wall_reduction"] == pytest.approx(0.4)
    assert report["gates"]["b"]["wall_ok"] is True
    assert report["gates"]["a"]["recall_ok"] is True
    assert report["gates"]["b"]["recall_ok"] is None
    assert report["all_pass"] is True


# ---------------------------------------------------------------- cache override

def test_the_eval_cache_directory_follows_the_environment(tmp_path):
    code = ("import eval.common as c, eval.bench_speed; print(c.CACHE_DIR)")
    env = dict(os.environ, FACEBLUR_EVAL_CACHE=str(tmp_path))
    got = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert Path(got) == tmp_path
    env.pop("FACEBLUR_EVAL_CACHE")
    got = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert Path(got) == REPO / "eval" / "cache"


# ---------------------------------------------------------------- settings and keys

def test_the_baseline_is_scored_without_the_adaptive_stride():
    from faceblur.settings import Settings
    base = bench.baseline_settings()
    assert base.adaptive_stride is False
    assert base == Settings(adaptive_stride=False)


def test_the_branch_is_scored_with_its_production_defaults():
    from faceblur.settings import Settings
    assert bench.branch_settings() == Settings()


def test_the_new_record_fields_are_reported_and_none_on_older_records():
    new = ("overlap_check_seconds", "second_chance_reencoded_frames",
           "frames_skipped_adaptive", "second_chance_seeds", "second_chance_seeds_kept")
    assert set(new) <= set(bench.RECORD_KEYS)
    branch = bench.file_entry({"wall_seconds": 10, "overlap_check_seconds": 2.5,
                               "second_chance_reencoded_frames": 30,
                               "frames_skipped_adaptive": 100, "second_chance_seeds": 7,
                               "second_chance_seeds_kept": 1}, None, None)
    assert [branch[k] for k in new] == [2.5, 30, 100, 7, 1]
    baseline = bench.file_entry({"wall_seconds": 10}, None, None)
    assert all(baseline[k] is None for k in new)
