"""Where does the time go? A profile of one production run, split by phase.

    python eval/profile_run.py <source.mp4> <out dir>

Runs `faceblur.batch.run_video` in this process (`serial_submit`, one
worker) with the production settings, the ones `cli.py --quarantine
--second-chance 1` builds. A profiler sees one process only, so the run has
to stay in one.

Two instruments, because one is not enough:

- Timers. Each phase entry point (`detect_job`, `Tracker.run`,
  `encode_job`, `residual_job`, ...) and each named part inside it
  (`decode_range`, `DetectorBank.detect_raw`, `redact`, ...) is wrapped in a
  wall clock, and so is `onnxruntime.InferenceSession.run`. The detection
  pass and the check call the same models, so the clock files every call
  under the phase that is running. These are the inclusive numbers.
- cProfile, one profiler per phase, switched as the phases change. Its own
  time per function (tottime) gives the hot spots. Its cumulative times are
  not trusted: on Python 3.12 cProfile loses the call events of frames that
  call into onnxruntime, and those frames show no or nonsense cumtime.

Writes to the out dir: the blurred copy and its record, `profile.prof`
(merged), `profile_<phase>.prof`, `timers.json`, and `profile_tables.md`,
which it also prints.
"""
from __future__ import annotations

import cProfile
import functools
import inspect
import json
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Phase name, module, attribute ("Class.method" is patched on the class).
PHASES = [
    ("detect", "faceblur.batch", "detect_job"),
    ("track", "faceblur.track", "Tracker.run"),
    ("zone", "faceblur.zone", "build"),
    ("zone", "faceblur.zone", "describe"),
    ("encode", "faceblur.batch", "encode_job"),
    ("copy", "faceblur.batch", "copy_job"),
    ("join", "faceblur.segments", "concat"),
    ("join", "faceblur.segments", "verify"),
    ("check_setup", "faceblur.verify", "plan_jobs"),
    ("check_setup", "faceblur.verify", "summarise"),
    ("check", "faceblur.verify", "residual_job"),
]

# Named parts, timed under whichever phase is running when they are called.
PARTS = [
    ("faceblur.segments", "decode_range"),
    ("faceblur.detect", "DetectorBank.detect_raw"),
    ("faceblur.detect", "DetectorBank.detect"),
    ("faceblur.detect", "DetectorBank.confirm_on_crops"),
    ("faceblur.detect", "YuNetOrtBackend.detect_resized"),
    ("faceblur.detect", "CenterFaceBackend.detect_batch"),
    ("faceblur.detect", "UltraFaceBackend.detect_batch"),
    ("faceblur.detect", "filter_candidates"),
    ("faceblur.detect", "weak_candidates"),
    ("faceblur.zone", "HandFinder.detect"),
    ("faceblur.hands", "PalmDetector.detect"),
    ("faceblur.hands", "HandLandmarks.presence"),
    ("faceblur.hands", "HandRule.hand_cover"),
    ("faceblur.motion", "estimate_shift"),
    ("faceblur.motion", "downscale"),
    ("faceblur.batch", "_bank"),
    ("faceblur.batch", "_zone"),
    ("faceblur.track", "Tracker.link"),
    ("faceblur.track", "Tracker._match"),
    ("faceblur.track", "Tracker._predict"),
    ("faceblur.track", "Tracker._stitch"),
    ("faceblur.track", "Tracker._fill_gaps"),
    ("faceblur.track", "Tracker._extend_backwards"),
    ("faceblur.track", "Tracker.render"),
    ("faceblur.redact", "redact"),
    ("faceblur.redact", "build_alpha"),
    ("faceblur.redact", "masked_shares"),
    ("faceblur.segments", "SegmentEncoder.__init__"),
    ("faceblur.segments", "SegmentEncoder.write"),
    ("faceblur.segments", "SegmentEncoder.close"),
    ("faceblur.verify", "applied"),
    ("faceblur.verify", "frame_noise"),
    ("faceblur.verify", "zone_changed"),
    ("faceblur.verify", "zone_crop"),
    ("faceblur.video", "probe"),
    ("faceblur.segments", "frame_times"),
    ("faceblur.segments", "keyframes"),
    ("faceblur.segments", "cut_copy"),
    ("faceblur.segments", "reorder_delay"),
    ("onnxruntime", "InferenceSession.run"),
]

INFERENCE = "InferenceSession.run"
CATEGORIES = ("gpu", "decode", "encode", "subprocess", "model_load", "cv2", "numpy", "python")


# ---------------------------------------------------------------- the instruments

class Instruments:
    """A wall clock per (phase, part) and a cProfile per phase."""

    def __init__(self):
        self.profiles = {"other": cProfile.Profile()}
        self.stack = ["other"]
        self.timers: dict[tuple[str, str], list] = {}   # (phase, part) -> [calls, seconds]

    def _add(self, part: str, seconds: float, calls: int = 1) -> None:
        t = self.timers.setdefault((self.stack[-1], part), [0, 0.0])
        t[0] += calls
        t[1] += seconds

    def phase(self, name: str, func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            self.profiles[self.stack[-1]].disable()
            self.stack.append(name)
            self.profiles.setdefault(name, cProfile.Profile()).enable()
            t = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                seconds = time.perf_counter() - t
                self.profiles[self.stack[-1]].disable()
                self._add("(phase wall)", seconds)
                self.stack.pop()
                self.profiles[self.stack[-1]].enable()
        return inner

    def timed(self, part: str, func):
        if inspect.isgeneratorfunction(func):
            # A generator's time is the time spent producing each item, not
            # the consumer's time between items.
            @functools.wraps(func)
            def gen(*args, **kwargs):
                it = func(*args, **kwargs)
                phase = self.stack[-1]
                calls, spent = 1, 0.0
                try:
                    while True:
                        t = time.perf_counter()
                        try:
                            item = next(it)
                        except StopIteration:
                            spent += time.perf_counter() - t
                            return
                        spent += time.perf_counter() - t
                        yield item
                finally:
                    it.close()
                    t = self.timers.setdefault((phase, part), [0, 0.0])
                    t[0] += calls
                    t[1] += spent
            return gen

        @functools.wraps(func)
        def inner(*args, **kwargs):
            t = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._add(part, time.perf_counter() - t)
        return inner

    def install(self) -> None:
        for name, module, attr in PHASES:
            _replace(module, attr, lambda f, n=name: self.phase(n, f))
        for module, attr in PARTS:
            _replace(module, attr, lambda f, a=attr: self.timed(a, f))


def _replace(module: str, attr: str, wrap) -> None:
    """Wrap `module.attr` and every faceblur module's own name for it."""
    import importlib

    owner = importlib.import_module(module)
    if "." in attr:
        cls, name = attr.split(".")
        owner = getattr(owner, cls)
        setattr(owner, name, wrap(getattr(owner, name)))
        return
    original = getattr(owner, attr)
    wrapped = wrap(original)
    for mod in [m for k, m in list(sys.modules.items()) if k.startswith("faceblur")] + [owner]:
        for key, value in list(vars(mod).items()):
            if value is original:
                setattr(mod, key, wrapped)


# ---------------------------------------------------------------- the tables

@functools.lru_cache(maxsize=1)
def _cv2_names() -> frozenset:
    import cv2

    return frozenset(dir(cv2)) | frozenset(dir(cv2.dnn))


def category(key, caller=None) -> str:
    """What a function's own time is: GPU inference, waiting on ffmpeg, or CPU.

    DirectML's `run` blocks until the GPU is done, and cProfile does not see
    the C call under onnxruntime's Python `run`, so that `run`'s own time is
    the inference. A read from or a write to an ffmpeg pipe is decode or
    encode work done in ffmpeg's process; which one is told by the caller.
    """
    file, _, name = key
    path = file.replace("\\", "/")
    if "/onnxruntime/" in path:
        if name == "run":
            return "gpu"
        return "model_load" if name in ("__init__", "_create_inference_session") else "python"
    if "InferenceSession" in name and "'run'" in name:
        return "gpu"
    if "_io.Buffered" in name:
        cname = caller[2] if caller else ""
        if cname == "decode_range":
            return "decode"
        if cname in ("write", "close") and caller[0].replace("\\", "/").endswith("segments.py"):
            return "encode"
        return "subprocess"
    if ("WaitForSingleObject" in name or "CreateProcess" in name or "_thread.lock" in name
            or "/subprocess.py" in path):
        return "subprocess"
    if "onnx" in name or "/onnx/" in path:
        return "model_load"
    # cProfile names a cv2 function `<resize>`, with no module.
    if "/cv2/" in path or (file == "~" and name[1:-1] in _cv2_names()):
        return "cv2"
    if "numpy" in name or "/numpy/" in path:
        return "numpy"
    return "python"


def split(stats: dict) -> dict[str, float]:
    """Own time by category. Sums to the phase's profiled total."""
    out = {c: 0.0 for c in CATEGORIES}
    for key, (_, _, tt, _, callers) in stats.items():
        if "_io.Buffered" in key[2] and callers:
            # Pipe I/O is split by who asked for it.
            for caller, edge in callers.items():
                out[category(key, caller)] += edge[2]
        else:
            out[category(key)] += tt
    return out


def total(stats: dict) -> float:
    return sum(v[2] for v in stats.values())


def label(key) -> str:
    file, line, name = key
    if file == "~":
        return name
    return f"{Path(file).name}:{line}({name})"


def hot_spots(phase_stats: dict[str, dict], walls: dict[str, float], n: int = 10) -> list[dict]:
    """The n functions with the most own time over every phase. The share is
    of the phase's wall, or of its profiled total where it has no wall."""
    rows = []
    for phase, stats in phase_stats.items():
        whole = walls.get(phase) or total(stats) or 1.0
        for key, (_, nc, tt, _, _) in stats.items():
            rows.append({"phase": phase, "function": label(key), "calls": nc,
                         "tottime": tt, "share": tt / whole, "kind": category(key)})
    rows.sort(key=lambda r: r["tottime"], reverse=True)
    return rows[:n]


def top_cumulative(stats: dict, n: int = 40) -> list[tuple[str, int, float, float]]:
    rows = [(label(k), v[1], v[2], v[3]) for k, v in stats.items()]
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows[:n]


def phase_walls(timers: dict) -> dict[str, float]:
    return {phase: v[1] for (phase, part), v in timers.items() if part == "(phase wall)"}


def tables(phase_stats: dict[str, dict], timers: dict, merged: dict, wall: float) -> str:
    walls = phase_walls(timers)
    walls["other"] = max(0.0, wall - sum(walls.values()))
    lines = [f"Run wall {wall:.1f} s (profiled, one process, serial).", "",
             "## Phases: wall, inference, and what the own time is", "",
             "`inference` is the wall clock around `InferenceSession.run`; the category "
             "columns are cProfile own time.", "",
             "| phase | wall s | inference s | GPU share | " + " | ".join(CATEGORIES) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(CATEGORIES)]
    for phase, stats in phase_stats.items():
        w = walls.get(phase, 0.0)
        inf = timers.get((phase, INFERENCE), [0, 0.0])[1]
        parts = split(stats)
        cells = " | ".join(f"{parts[c]:.1f}" for c in CATEGORIES)
        lines.append(f"| {phase} | {w:.1f} | {inf:.1f} | {inf / w if w else 0:.0%} | {cells} |")
    lines += ["", "## Named parts per phase (wall clock, inclusive)", ""]
    for phase in phase_stats:
        rows = sorted(((part, v) for (ph, part), v in timers.items()
                       if ph == phase and part != "(phase wall)"),
                      key=lambda r: r[1][1], reverse=True)
        if not rows:
            continue
        w = walls.get(phase) or 1.0
        lines += [f"### {phase}", "", "| part | calls | seconds | share of phase |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {part} | {v[0]} | {v[1]:.1f} | {v[1] / w:.0%} |" for part, v in rows]
        lines.append("")
    lines += ["## Top 10 by own time (cProfile tottime)", "",
              "| # | phase | function | calls | tottime s | share of phase | kind |",
              "|---:|---|---|---:|---:|---:|---|"]
    for i, r in enumerate(hot_spots(phase_stats, walls), 1):
        lines.append(f"| {i} | {r['phase']} | {r['function']} | {r['calls']} | "
                     f"{r['tottime']:.1f} | {r['share']:.0%} | {r['kind']} |")
    lines += ["", "## Top 40 by cumulative time (cProfile, all phases merged; see the "
              "module docstring on cumtime)", "",
              "| function | calls | tottime s | cumtime s |", "|---|---:|---:|---:|"]
    lines += [f"| {f} | {nc} | {tt:.1f} | {ct:.1f} |" for f, nc, tt, ct in
              top_cumulative(merged)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def production_settings():
    """What `cli.py --quarantine --second-chance 1` builds, every other flag
    at its default: the CLI turns check_output on for either flag."""
    from faceblur.settings import Settings

    return Settings(check_output=True, quarantine=True, second_chance=1)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    import faceblur.batch
    import faceblur.verify  # noqa: F401  (loaded so its names are wrapped too)

    settings = production_settings()
    dst = out / f"{src.stem}{settings.suffix}{src.suffix}"
    inst = Instruments()
    inst.install()
    started = time.perf_counter()
    inst.profiles["other"].enable()
    try:
        record = faceblur.batch.run_video(src, dst, settings,
                                          submit=faceblur.batch.serial_submit, workers=1)
    finally:
        inst.profiles["other"].disable()
    wall = time.perf_counter() - started
    print(f"status {record.status}  wall {record.wall_seconds}  detect {record.detect_seconds}  "
          f"encode {record.encode_seconds}  check {record.check_seconds}  "
          f"second_chance {record.second_chance_seconds}  frames {record.frames}  "
          f"error {record.error!r}")

    order = ["detect", "track", "zone", "encode", "copy", "join", "check_setup", "check",
             "other"]
    phase_stats = {}
    for phase in order:
        p = inst.profiles.get(phase)
        if p is None:
            continue
        p.dump_stats(str(out / f"profile_{phase}.prof"))
        phase_stats[phase] = pstats.Stats(p).stats
    merged = pstats.Stats(*[str(out / f"profile_{ph}.prof") for ph in phase_stats])
    merged.dump_stats(str(out / "profile.prof"))
    (out / "timers.json").write_text(json.dumps(
        [{"phase": ph, "part": part, "calls": v[0], "seconds": round(v[1], 3)}
         for (ph, part), v in sorted(inst.timers.items())], indent=1), encoding="utf-8")
    text = tables(phase_stats, inst.timers, merged.stats, wall)
    (out / "profile_tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
