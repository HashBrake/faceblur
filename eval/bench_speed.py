"""Is this branch 30 percent faster than the baseline, at no loss of quality?

Runs the production command on the four sample files twice, once with the
baseline build (commit 3eb0100, exported with `git archive` into
`tools/baseline/`) and once with this worktree, and writes the comparison to
`docs/bench/speed_<date>.json`. The gates come from the build brief:

- wall time per file at least 30 percent lower than the baseline's;
- `faceblur.verify` finds no more faces of 40 px or more on the branch's
  copy than on the baseline's. The verify that judges both is the baseline's
  own, so the quality gate is not measured by code this branch changed;
- masked-frame recall from the eval harness within 0.5 points of baseline.

Every step is resumable and can run alone, because the whole thing is hours
of GPU time on a machine three other sessions share and that kills jobs when
RAM runs low:

    python eval/bench_speed.py [--only export,baseline,branch,verify,recall,report]
                               [--date YYYY-MM-DD] [--workers 4]

Every GPU-heavy or timed child goes through `gpu_lock.py`, so the sessions
take turns and a timing is never taken with somebody else's job on the card.
Outputs go to `bench_out/` and caches to this worktree's `eval/cache`; git
ignores both, and `tools/`.
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAIN = Path("D:/Claude Projects/Player One Face Removal")
FOOTAGE = MAIN / "footage"
MAIN_CACHE = MAIN / "eval" / "cache"
GPU_LOCK = Path("D:/Claude Projects/gpu_lock.py")

BASELINE_COMMIT = "3eb0100"
BASELINE_DIR = REPO / "tools" / "baseline"
OUT = REPO / "bench_out"
WORK_CACHE = REPO / "eval" / "cache"
BENCH_DIR = REPO / "docs" / "bench"
BUILDS = ("baseline", "branch")
STEPS = ("export", "baseline", "branch", "verify", "recall", "report")
VERIFY_CODE = f"baseline {BASELINE_COMMIT}"

WALL_GATE = 0.30          # at least this much less wall per file, inclusive
RECALL_TOL = 0.005        # half a point of recall, either way
_EPS = 1e-9               # 1 - 7/10 is 0.29999999999999993 in floats


# ---------------------------------------------------------------- settings

def baseline_settings():
    """What the baseline is scored with in the recall harness.

    `Settings(adaptive_stride=False)` reproduces 3eb0100's masks exactly (D15):
    the tracker, the detector's candidate filters, the zone and the plain
    stride are unchanged on this branch, and the adaptive stride (T07) is the
    only default since then that changes which frames are detected. The other
    new defaults (second-chance scope, overlapped check, GOP, ROI work) touch
    the encode and the check, never the masks the recall harness scores.
    """
    sys.path.insert(0, str(REPO))
    from faceblur.settings import Settings
    return Settings(adaptive_stride=False)


def branch_settings():
    """What this branch is scored with: its production defaults."""
    sys.path.insert(0, str(REPO))
    from faceblur.settings import Settings
    return Settings()


SIDE_SETTINGS = {"baseline": baseline_settings, "branch": branch_settings}


# ---------------------------------------------------------------- commands

def locked(cmd: list[str]) -> list[str]:
    """Run `cmd` under the PC's single GPU lock."""
    return ["py", "-3.12", str(GPU_LOCK), "--", *cmd]


def build_dir(build: str) -> Path:
    return BASELINE_DIR if build == "baseline" else REPO


def build_out(build: str, out_root: Path = OUT) -> Path:
    return out_root / build


def cli_command(out: Path, workers: int, footage: Path = FOOTAGE) -> list[str]:
    """The production command whose wall time is the gate. Run with cwd at the build."""
    return [sys.executable, "cli.py", str(footage), "-o", str(out),
            "--workers", str(workers), "--quarantine", "--second-chance", "1",
            "--report", str(out / "report.json"), "--no-progress"]


def timed_command(cmd: list[str], run_json: Path) -> list[str]:
    """`cmd` wrapped so the clock starts once the lock is held, not before.

    The wall of a run that waited 20 minutes for another session's job is not
    the wall of the build.
    """
    return locked([sys.executable, str(Path(__file__).resolve()),
                   "--timed", str(run_json), "--", *cmd])


def verify_command(source: Path, copy: Path, report: Path, workers: int) -> list[str]:
    """The reference check, run with cwd at the baseline export."""
    return [sys.executable, "-m", "faceblur.verify", str(source), str(copy),
            "--workers", str(workers), "--json", str(report)]


def nvidia_smi() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def stems(footage: Path = FOOTAGE) -> list[str]:
    return sorted(p.stem for p in footage.glob("*.mp4"))


# ---------------------------------------------------------------- resume

def copy_path(out: Path, stem: str) -> Path | None:
    """The copy the CLI wrote for `stem`, delivered or held in quarantine.

    Only a copy with its sidecar counts: the sidecar is written last, so a
    copy without one is a run that was killed.
    """
    for p in (out / f"{stem}_blurred.mp4", out / "quarantine" / f"{stem}_blurred.mp4"):
        if Path(str(p) + ".json").is_file():
            return p
    return None


def load_record(out: Path, stem: str) -> dict | None:
    copy = copy_path(out, stem)
    if copy is None:
        return None
    try:
        return json.loads(Path(str(copy) + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def missing(out: Path, names: list[str]) -> list[str]:
    """The files of this build that have no finished record yet."""
    return [s for s in names if (load_record(out, s) or {}).get("status") != "done"]


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- steps

def step_export() -> None:
    """Extract the baseline commit, models included, unless it is already there."""
    if (BASELINE_DIR / "cli.py").is_file():
        print(f"export: {BASELINE_DIR} present")
        return
    data = subprocess.run(["git", "archive", "--format=tar", BASELINE_COMMIT], cwd=REPO,
                          capture_output=True, check=True).stdout
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(BASELINE_DIR, filter="data")
    (BASELINE_DIR / "COMMIT").write_text(git("rev-parse", BASELINE_COMMIT) + "\n",
                                         encoding="utf-8")
    print(f"export: {BASELINE_COMMIT} -> {BASELINE_DIR}")


def step_run(build: str, workers: int) -> None:
    """The production command for one build, until every file has a record.

    A re-run of the same command skips what is done, so a run killed halfway
    picks up where it stopped; its walls are kept in `runs.json`, one entry
    per attempt.
    """
    out = build_out(build)
    names = stems()
    todo = missing(out, names)
    if not todo:
        print(f"{build}: all {len(names)} records present")
        return
    if build == "baseline" and not (BASELINE_DIR / "cli.py").is_file():
        raise SystemExit("baseline: run --only export first")
    out.mkdir(parents=True, exist_ok=True)
    run_json = out / "run.part.json"
    run_json.unlink(missing_ok=True)
    print(f"{build}: running {len(todo)} of {len(names)} files", flush=True)
    code = subprocess.call(timed_command(cli_command(out, workers), run_json),
                           cwd=build_dir(build))
    run = load_json(run_json) or {"exit_code": code}
    run["files_todo"] = todo
    runs_path = out / "runs.json"
    runs = load_json(runs_path) or {"runs": []}
    runs["runs"].append(run)
    runs_path.write_text(json.dumps(runs, indent=1), encoding="utf-8")
    run_json.unlink(missing_ok=True)
    left = missing(out, names)
    print(f"{build}: exit {code}, {run.get('total_wall_seconds')} s, "
          f"{len(left)} files without a record", flush=True)


def step_verify(workers: int) -> None:
    """The baseline's verify on every copy of both builds; a missing build is skipped."""
    for build in BUILDS:
        out = build_out(build)
        for stem in stems():
            copy = copy_path(out, stem)
            if copy is None:
                print(f"verify: {build} {stem}: no copy yet")
                continue
            dst = out / "verify" / f"{stem}.json"
            if dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"verify: {build} {stem}", flush=True)
            # Exit 1 is the gate holding the copy back, which is expected here.
            subprocess.call(locked(verify_command(FOOTAGE / f"{stem}.mp4", copy, dst, workers)),
                            cwd=BASELINE_DIR)
            if not dst.is_file():
                print(f"verify: {build} {stem}: no report written", flush=True)


def step_recall() -> None:
    """The eval harness per side, from raw caches built in this worktree.

    The small label files are copied from the main checkout's cache; its raw
    caches are not, because they carry the fingerprint of `conf` 0.6 and
    would not be read.
    """
    WORK_CACHE.mkdir(parents=True, exist_ok=True)
    for stem in stems():
        for kind in ("consensus", "oracle", "hands"):
            src, dst = MAIN_CACHE / f"{stem}.{kind}.json", WORK_CACHE / f"{stem}.{kind}.json"
            if src.is_file() and not dst.exists():
                shutil.copyfile(src, dst)
    env = dict(os.environ, FACEBLUR_EVAL_CACHE=str(WORK_CACHE))
    for side in BUILDS:
        for stem in stems():
            dst = OUT / "recall" / side / f"{stem}.json"
            if recall_current(dst, SIDE_SETTINGS[side]().to_dict()):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not (WORK_CACHE / f"{stem}.consensus.json").is_file():
                dst.write_text(json.dumps({"recall": None, "note": "no consensus labels",
                                           "settings": SIDE_SETTINGS[side]().to_dict()},
                                          indent=1), encoding="utf-8")
                continue
            print(f"recall: {side} {stem}", flush=True)
            # As a module from the worktree root, so `eval/` itself is not on
            # the path and its files cannot shadow a top-level import.
            subprocess.call(locked([sys.executable, "-m", "eval.bench_speed",
                                    "--recall-one", side, stem]), cwd=REPO, env=env)
            if not dst.is_file():
                print(f"recall: {side} {stem}: failed", flush=True)


def recall_current(path: Path, settings: dict) -> bool:
    """A stored recall counts only if it was scored with the side's settings now.

    Later tasks edit `branch_settings`; a result from the old defaults would
    otherwise be reported as the new build's.
    """
    held = load_json(path)
    return held is not None and held.get("settings") == json.loads(json.dumps(settings))


def recall_one(side: str, stem: str) -> int:
    """One side, one video, in its own process under the lock.

    Builds the raw cache and the hands cache when they are missing, and
    refuses a result scored without the zone: every mask that ships has the
    zone subtracted, and a recall measured without it describes no build.
    """
    if "FACEBLUR_EVAL_CACHE" not in os.environ:
        raise SystemExit("--recall-one wants FACEBLUR_EVAL_CACHE set")
    sys.path.insert(0, str(REPO))
    import eval.common as common
    import eval.zone as zone
    from eval.measure import evaluate, load_inputs
    # eval/zone.py keeps its own CACHE_DIR; point it where the override points.
    zone.CACHE_DIR = common.CACHE_DIR
    settings = SIDE_SETTINGS[side]()
    video = FOOTAGE / f"{stem}.mp4"
    if zone.load_hands(video, settings) is None:
        print(f"  building the hands cache for {stem}", flush=True)
        zone.build_hands_cache(video, settings)
    cache, consensus, oracle = load_inputs(video, settings)
    r = evaluate(settings, cache, consensus, oracle, video)
    if r["zone_note"]:
        raise SystemExit(f"recall {side} {stem}: scored without the zone: {r['zone_note']}")
    dst = OUT / "recall" / side / f"{stem}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"  recall {r['recall']}", flush=True)
    return 0


def timed(run_json: Path, cmd: list[str]) -> int:
    """Run `cmd` and write its wall and the GPU state before it."""
    before = nvidia_smi()
    started = time.time()
    code = subprocess.call(cmd)
    Path(run_json).write_text(json.dumps({
        "nvidia_smi_before": before, "total_wall_seconds": round(time.time() - started, 2),
        "exit_code": code, "started": datetime.datetime.fromtimestamp(started).isoformat(
            timespec="seconds")}, indent=1), encoding="utf-8")
    return code


# ---------------------------------------------------------------- report

def faces40(verify: dict | None) -> int | None:
    if not verify:
        return None
    return (verify.get("residual_by_size") or {}).get("40+ px")


def gate(base: dict | None, branch: dict | None) -> dict:
    """The three gates for one file, from the two file entries of the report.

    A side that has not run leaves its gates null rather than failing them.
    Recall is null on both sides for a file without consensus labels, and so
    is its gate: there is nothing to compare.
    """
    base, branch = base or {}, branch or {}
    bw, nw = base.get("wall_seconds"), branch.get("wall_seconds")
    reduction = 1 - nw / bw if bw and nw is not None else None
    f_base, f_branch = faces40(base.get("verify")), faces40(branch.get("verify"))
    r_base, r_branch = base.get("recall"), branch.get("recall")
    return {
        "wall_reduction": None if reduction is None else round(reduction, 4),
        "wall_ok": None if reduction is None else reduction >= WALL_GATE - _EPS,
        "faces40_baseline": f_base,
        "faces40_branch": f_branch,
        "faces40_ok": None if f_base is None or f_branch is None else f_branch <= f_base,
        "recall_baseline": r_base,
        "recall_branch": r_branch,
        "recall_ok": (None if r_base is None or r_branch is None
                      else abs(r_branch - r_base) <= RECALL_TOL + _EPS),
    }


def all_pass(gates: dict) -> bool | None:
    """True when every file passes; None while any wall or faces gate is unmeasured."""
    if not gates:
        return None
    for g in gates.values():
        if g["wall_ok"] is None or g["faces40_ok"] is None:
            return None
    return all(g["wall_ok"] and g["faces40_ok"] and g["recall_ok"] is not False
               for g in gates.values())


RECORD_KEYS = ("wall_seconds", "detect_seconds", "encode_seconds", "check_seconds",
               "second_chance_seconds", "second_chance_rounds", "frames", "quarantined",
               "overlap_check_seconds", "second_chance_reencoded_frames",
               "frames_skipped_adaptive", "second_chance_seeds", "second_chance_seeds_kept")
RECALL_KEYS = ("recall", "recall_by_size", "exposed_40", "exposed_24_40", "continuity",
               "masked_mean")


def file_entry(record: dict | None, verify: dict | None, recall: dict | None) -> dict:
    record, recall = record or {}, recall or {}
    entry = {k: record.get(k) for k in RECORD_KEYS}
    entry["record_residual_faces"] = record.get("residual_faces")
    entry["record_residual_by_size"] = record.get("residual_by_size")
    entry["verify"] = None if verify is None else {
        k: verify.get(k) for k in ("residual_faces", "residual_by_size", "residual_max_px",
                                   "frames_checked")}
    entry.update({k: recall.get(k) for k in RECALL_KEYS})
    return entry


def build_entry(names: list[str], records: dict, verifies: dict, recalls: dict,
                runs: dict | None) -> dict:
    """One build's side of the report, from what is on disk for it."""
    runs = (runs or {}).get("runs", [])
    settings = next((r.get("settings") for r in records.values() if r), None)
    if settings is None:
        settings = next((r.get("settings") for r in recalls.values() if r), None)
    walls = [r.get("total_wall_seconds") for r in runs if r.get("total_wall_seconds") is not None]
    return {
        "settings": settings,
        "nvidia_smi_before": runs[0].get("nvidia_smi_before") if runs else None,
        "total_wall_seconds": round(sum(walls), 2) if walls else None,
        "runs": len(runs),
        "files": {s: file_entry(records.get(s), verifies.get(s), recalls.get(s))
                  for s in names},
    }


def assemble(date: str, branch_commit: str, workers: int, names: list[str],
             builds: dict) -> dict:
    """The whole report. `builds` maps a build to its `build_entry`, or None."""
    gates = {s: gate(((builds.get("baseline") or {}).get("files") or {}).get(s),
                     ((builds.get("branch") or {}).get("files") or {}).get(s))
             for s in names}
    return {
        "date": date,
        "baseline_commit": BASELINE_COMMIT,
        "branch_commit": branch_commit,
        "workers": workers,
        "command": " ".join(cli_command(Path("OUT"), workers, Path("FOOTAGE"))[1:]),
        "verify_code": VERIFY_CODE,
        "builds": {b: builds.get(b) for b in BUILDS},
        "gates": gates,
        "all_pass": all_pass(gates),
    }


def collect(build: str, names: list[str], out_root: Path = OUT) -> dict | None:
    out = build_out(build, out_root)
    records = {s: load_record(out, s) for s in names}
    verifies = {s: load_json(out / "verify" / f"{s}.json") for s in names}
    recalls = {s: load_json(out_root / "recall" / build / f"{s}.json") for s in names}
    runs = load_json(out / "runs.json")
    if not any(records.values()) and not any(recalls.values()):
        return None
    return build_entry(names, records, verifies, recalls, runs)


def step_report(date: str, workers: int) -> Path:
    names = stems()
    builds = {b: collect(b, names) for b in BUILDS}
    report = assemble(date, git("rev-parse", "HEAD"), workers, names, builds)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = BENCH_DIR / f"speed_{date}.json"
    dst.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {dst}  all_pass {report['all_pass']}")
    for s, g in report["gates"].items():
        print(f"  {s}: wall {g['wall_reduction']} faces40 {g['faces40_baseline']}->"
              f"{g['faces40_branch']} recall {g['recall_baseline']}->{g['recall_branch']}")
    return dst


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Internal modes, used by the children this script launches under the lock.
    if argv[:1] == ["--timed"]:
        rest = argv[2:]
        return timed(Path(argv[1]), rest[1:] if rest[:1] == ["--"] else rest)
    if argv[:1] == ["--recall-one"]:
        return recall_one(argv[1], argv[2])

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(STEPS),
                    help=f"comma separated steps out of {', '.join(STEPS)}")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown step {', '.join(unknown)}")
    for step in STEPS:
        if step not in steps:
            continue
        if step == "export":
            step_export()
        elif step in BUILDS:
            step_run(step, args.workers)
        elif step == "verify":
            step_verify(args.workers)
        elif step == "recall":
            step_recall()
        elif step == "report":
            step_report(args.date, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
