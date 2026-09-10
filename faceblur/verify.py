"""What the finished copy still shows.

Every other check in this pipeline reads the source. The detectors say what
they found, the tracker says what it kept, the audit says how much was
destroyed, and the evaluation harness measures all of it against the source
too. None of them can see a face that every one of them missed: a miss is
invisible to the thing that missed it.

This pass reads the output instead. A face the detectors find in the finished
copy, on pixels the pipeline never changed, is a face this run missed, and
saying so needs no labels, no recogniser and nobody's time.

A detector fires on a blurred face as well, so a box in the copy means
nothing by itself. What separates the two is whether anything happened there.
The comparison is made where a mask would land, not over the whole box, and
at the peak rather than the average, against what the encoder moved on its
own. Near one, the mask ran and the detector is firing on its own blur. Near
zero, nothing reached that face.

Some boxes cannot answer: on a flat wall or a dark corner a mask changes
almost nothing, so neither does the ratio. Those are counted and reported
separately rather than being called either way.

One question is asked of every face that survives: is it a hand? The same
detectors that produced the copy call the wearer's hand a face, and the
track-level rules that keep hands out of the mask have no counterpart in a
check that reads one frame at a time. `faceblur/hands.py` answers it with
MediaPipe's palm detector and its landmark model. A box a confirmed hand
covers stays in the record, marked with the coverage that decided it, and is
left out of every count a decision hangs on.

The check covers the kinds in `KINDS_CHECKED`, and only the ones this run was
asked to mask. A run that never asked for screens is not held back for one.

Screens are asked a question faces are not: **how long was it there?** The
screen detector calls a table a laptop for a frame and then stops, and 14 of
the 29 false runs on the sample footage last a single frame while no real
screen does (`docs/report.md` section 15.2). A gate that held a copy back for
one frame of a table would hold every copy back, so screen finds are grouped
into runs by `screens.runs_in` and only a run of `screen_gate_min_run`
checked frames decides anything. Faces need no such rule: one frame of a face
is a face.

    .venv\\Scripts\\python.exe -m faceblur.verify SOURCE OUTPUT [--stride N]
                                                  [--workers N] [--no-hand-rule]
                                                  [--mask face,screen]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .detect import Detection
from .redact import redact, region_for
from .segments import decode_range, frame_times
from .settings import Settings
from .video import VideoError, probe

# The kinds this check can look for in a finished copy. A kind is not ready
# until it is in here: a mask nothing checks is a promise nobody has tested.
# `tests/test_classes.py` fails if a kind is marked ready and is not named
# here, so condition 2 of "ready" is enforced by a test and not by memory.
KINDS_CHECKED: tuple[str, ...] = ("face", "screen")

# Below this share of the change a mask would make, nothing reached the box.
MISSED_APPLIED = 0.3
# A mask that moves the pixels less than this, once the encoder's own noise is
# out of the way, cannot be told from that noise, so the ratio means nothing.
FLAT_DIFF = 4.0
# Every pixel of the copy differs a little from the source: the whole file is
# re-encoded. Its size is measured per frame and taken off both sides of the
# ratio, because on a small dark face it is most of what the ratio would
# otherwise be reading. Sampling every fourth row and column is enough for a
# median and keeps the check off the critical path.
NOISE_SAMPLE = 4


def box_diff(a: np.ndarray, b: np.ndarray, det: Detection) -> float:
    """Mean absolute difference between two frames inside a detection's box."""
    h, w = a.shape[:2]
    x0, y0 = max(0, int(det.x)), max(0, int(det.y))
    x1, y1 = min(w, int(det.x + det.w)), min(h, int(det.y + det.h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.mean(np.abs(a[y0:y1, x0:x1].astype(np.float32)
                                - b[y0:y1, x0:x1].astype(np.float32))))


def frame_noise(before: np.ndarray, after: np.ndarray, step: int = NOISE_SAMPLE) -> float:
    """What re-encoding alone moves the pixels, over the whole frame.

    The median, not the mean: masked faces are a small share of a frame and a
    median ignores them, which is exactly what is wanted here.
    """
    a = before[::step, ::step].astype(np.int16)
    b = after[::step, ::step].astype(np.int16)
    return float(np.median(np.abs(a - b)))


def applied(before: np.ndarray, after: np.ndarray, det: Detection,
            settings: Settings) -> tuple[float, float]:
    """(what the copy changed, what a mask here would change) where a mask lands.

    Not over the whole box. A detector's box around a large face takes in hair,
    neck and background, the mask covers an ellipse over the eyes, nose and
    mouth, and averaging over the box mixes the two: on frame 228 of `004100`
    a 193 px face that was masked properly read 0.29 and was called a miss.
    The control says exactly which pixels a mask would move, and the question
    is only ever about those.

    The region comes from `redact.region_for`, so this asks the same question
    of a screen's quad that it asks of a face's ellipse: the control masks
    whatever shape that kind gets, and the comparison is made inside it.
    """
    control, _ = redact(before, [det], settings)
    e = region_for(det, settings)
    h, w = before.shape[:2]
    x0, y0, x1, y1 = e.bounds(settings.feather * 2 + 2)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    a = before[y0:y1, x0:x1].astype(np.float32)
    b = after[y0:y1, x0:x1].astype(np.float32)
    c = control[y0:y1, x0:x1].astype(np.float32)
    mask_diff = np.abs(a - c).mean(axis=2)
    where = mask_diff > 1.0                     # the mask's own footprint
    if where.sum() < 16:
        return 0.0, 0.0
    changed = np.abs(a - b).mean(axis=2)[where]
    # The peak, not the average. A blurred face on this footage gives the
    # detector a box half again the size of the one the pipeline masked, and
    # averaging over that box mixes a destroyed face with untouched hair: on
    # frame 228 of `004100` a masked 193 px face averaged 0.24 of what a mask
    # would do, in the middle of the misses. Read at the 95th percentile the
    # same case reads 0.53 and the misses 0.11 to 0.19, because a mask that
    # lands anywhere on a face moves some pixels a long way and an untouched
    # face moves none further than the encoder does. The percentile rather
    # than the maximum keeps one hot pixel from deciding.
    return float(np.percentile(changed, 95)), float(np.percentile(mask_diff[where], 95))


def classify(changed: float, would: float, noise: float = 0.0) -> str:
    """missed, masked, or flat: see the module docstring.

    `noise` is what the encoder moved on its own. Both sides of the ratio are
    measured against it, since neither number means anything below it: on a
    dark 40 px face a mask moves the pixels by 15 and the encoder by 3.5, and
    without this a plain miss creeps up towards the line.

    On the eight boxes of `004100` whose answer was read off the difference
    maps by eye, this rule puts the three misses at 0.11 to 0.19 and the five
    masked faces at 0.53 to 1.44.
    """
    changed, would = max(0.0, changed - noise), max(0.0, would - noise)
    if would < FLAT_DIFF:
        return "flat"
    return "missed" if changed / would < MISSED_APPLIED else "masked"


def runs_of(frames: list[int], gap: int = 5) -> list[list[int]]:
    """Frame numbers merged into [first, last] stretches, gaps of `gap` closed."""
    out: list[list[int]] = []
    for i in sorted(set(frames)):
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def kinds_checked(settings: Settings) -> tuple[str, ...]:
    """The kinds this check will look for: what it can do and what was asked.

    A run that never asked for screens must not be held back for one. The
    copy of such a run has every screen in it untouched by design, and
    reporting them as residuals would be reporting the scope, not a miss.
    """
    return tuple(name for name in KINDS_CHECKED if settings.wants(name))


def _row(det: Detection, frame: int, changed: float, would: float,
         noise: float) -> dict:
    """One find, as the record keeps it. Coordinates and counts, never pixels."""
    over = max(0.0, would - noise)
    return {"kind": det.kind, "frame": frame, "px": round(det.long_side),
            "x": round(det.cx), "y": round(det.cy),
            "score": round(det.score, 3),
            "applied": round(max(0.0, changed - noise) / over, 3) if over else None}


def residual_job(job: dict) -> dict:
    """Detect on the copy's frames start to end and say what was never masked."""
    from .batch import _bank, _hand_rule, _screens    # the worker's cached models

    src, out = Path(job["src"]), Path(job["out"])
    settings: Settings = job["settings"]
    start, end = job["start"], job["end"]
    stride = max(1, job.get("stride", 1))
    wanted = kinds_checked(settings)
    bank = _bank(settings) if "face" in wanted else None
    screens = _screens(settings) if "screen" in wanted else None
    rule = _hand_rule(settings) if bank is not None and settings.hand_rule else None
    found, hands, checked = [], 0, 0
    flat = {name: 0 for name in wanted}
    source_frames = decode_range(src, job["times"], start, end, job["info"], settings.hwaccel)
    copy_frames = decode_range(out, job["out_times"], start, end, job["out_info"],
                               settings.hwaccel)
    for offset, (before, after) in enumerate(zip(source_frames, copy_frames)):
        i = start + offset
        if i % stride:
            continue
        checked += 1
        faces = bank.detect(after) if bank is not None else []
        glass = screens.detect(after) if screens is not None else []
        if not faces and not glass:
            continue
        noise = frame_noise(before, after)
        for det in faces:
            changed, would = applied(before, after, det, settings)
            call = classify(changed, would, noise)
            if call == "flat":
                flat["face"] += 1
            elif call == "missed":
                row = _row(det, i, changed, would, noise)
                if rule is not None:
                    # The hand rule reads the source, not the copy. Nothing
                    # masked a hand, so it looks the same in both, and the
                    # source is the sharper of the two.
                    cover = rule.hand_cover(before, det)
                    if cover >= settings.hand_cover:
                        row["hand"] = round(cover, 3)
                        hands += 1
                found.append(row)
        for det in glass:
            # No hand rule here: nothing mistakes a hand for a television, and
            # a hand over a monitor does not stop it being a monitor. What a
            # screen needs instead is persistence, and that is decided in
            # `summarise` over the whole file rather than one frame at a time.
            changed, would = applied(before, after, det, settings)
            call = classify(changed, would, noise)
            if call == "flat":
                flat["screen"] += 1
            elif call == "missed":
                row = _row(det, i, changed, would, noise)
                # The box, so that `summarise` can group these into runs, and
                # the label, so that a reader knows the detector said phone
                # rather than television.
                row.update({"w": round(det.w), "h": round(det.h), "label": det.source})
                found.append(row)
    return {"start": start, "end": end, "checked": checked,
            "flat": sum(flat.values()), "flat_by_kind": flat,
            "hands": hands, "found": found,
            "hand_models": {} if rule is None else rule.model_hashes,
            "hand_compute": {} if rule is None else rule.compute}


def plan_jobs(src: Path, out: Path, settings: Settings, chunk: int,
              stride: int = 1) -> list[dict]:
    """One job per frame range, with both files' timing resolved once."""
    info, out_info = probe(src), probe(out)
    times, out_times = frame_times(src), frame_times(out)
    n = min(len(times), len(out_times))
    if n == 0:
        raise VideoError("Could not read one of these videos.")
    return [{"src": str(src), "out": str(out), "settings": settings, "stride": stride,
             "info": info, "out_info": out_info, "times": times, "out_times": out_times,
             "start": a, "end": min(b, n)}
            for a, b in [(k, k + chunk) for k in range(0, n, max(1, chunk))]]


def not_hands(rows: list[dict]) -> list[dict]:
    """The rows that decide anything: what is left once the hands are out."""
    return [row for row in rows if "hand" not in row]


def of_kind(rows: list[dict], name: str) -> list[dict]:
    """The rows of one kind. A row written before 2026-09-10 has no kind and
    is a face, because a face was all this check could look for."""
    return [row for row in rows if row.get("kind", "face") == name]


def screen_runs(rows: list[dict], settings: Settings) -> list[dict]:
    """Screen finds grouped into runs: same label, overlapping, close in time.

    The same grouping the pipeline uses on the source side, on the same
    settings, so a run means the same thing on both sides of the file. It
    matters here because a run of one frame is what a table looks like and a
    run of three is what a screen looks like, and the gate reads the length.

    `first` and `last` are frame numbers in the video; `hits` counts the
    *checked* frames the run was seen in, which at `check_stride` 2 are two
    source frames apart. `x` and `y` are the centre of the last box, so that
    a reader can find the thing in the frame and a later pass can match a run
    on the copy against one on the source.
    """
    from .screens import runs_in

    per_frame: dict[int, list[Detection]] = {}
    for row in rows:
        w, h = float(row.get("w", row["px"])), float(row.get("h", row["px"]))
        per_frame.setdefault(row["frame"], []).append(
            Detection(row["x"] - w / 2, row["y"] - h / 2, w, h, row["score"],
                      None, row.get("label", "tv"), True, row["score"], "screen"))
    out = []
    for run in runs_in(per_frame, settings):
        frames = [i for i, _ in run.hits]
        last = run.last_box
        out.append({"label": run.label, "first": frames[0], "last": frames[-1],
                    "hits": len(frames),
                    "px": max(round(det.long_side) for _, det in run.hits),
                    "x": round(last.cx), "y": round(last.cy),
                    "w": round(last.w), "h": round(last.h)})
    return sorted(out, key=lambda run: (run["first"], -run["hits"]))


def summarise(results: list[dict], settings: Optional[Settings] = None) -> dict:
    """The audit record's view: what the copy still shows, and where.

    Hands stay in `residual_list`, each marked with the coverage that decided
    it, because a rule that quietly dropped what it disagreed with would be
    worth nothing to an auditor. Every count a decision hangs on leaves them
    out.

    `flat_boxes` counts every box of every kind a mask would barely move, and
    `flat_by_kind` splits it. The single number keeps its name and becomes the
    union, the way the per kind mask shares do in the audit record.
    """
    settings = settings or Settings()
    found = [row for r in results for row in r["found"]]
    found.sort(key=lambda row: (row["frame"], -row["px"]))
    faces = not_hands(of_kind(found, "face"))
    glass = of_kind(found, "screen")
    checked = sum(r["checked"] for r in results)
    by_size = {"40+ px": 0, "24-40 px": 0, "under 24 px": 0}
    for row in faces:
        key = ("40+ px" if row["px"] >= 40 else
               "24-40 px" if row["px"] >= 24 else "under 24 px")
        by_size[key] += 1
    models, compute = {}, {}
    flat_by_kind: dict[str, int] = {}
    for r in results:
        models.update(r.get("hand_models") or {})
        compute.update(r.get("hand_compute") or {})
        for name, count in (r.get("flat_by_kind") or {}).items():
            flat_by_kind[name] = flat_by_kind.get(name, 0) + count
    # Runs are grouped from the finds that clear the size floor, not from all
    # of them. A gate built the other way could be held back by a run of 20 px
    # flickers and one unrelated large find in a single frame.
    big = [row for row in glass if row["px"] >= settings.screen_gate_min_px]
    # 200 rows of each kind rather than 200 rows in total. On the table tennis
    # file the check finds 67 faces and 129 screens, and a shared cap would let
    # a file with hundreds of screen finds crowd the faces out of the record
    # that an auditor reads. Every count above is over every row either way.
    kept = of_kind(found, "face")[:200] + of_kind(found, "screen")[:200]
    kept.sort(key=lambda row: (row["frame"], -row["px"]))
    return {"frames_checked": checked,
            "hand_models": models,
            "hand_compute": compute,
            "flat_boxes": sum(r["flat"] for r in results),
            "flat_by_kind": flat_by_kind,
            "residual_faces": len(faces),
            "residual_hands": sum(r.get("hands", 0) for r in results),
            "residual_frames": len({row["frame"] for row in faces}),
            "residual_by_size": by_size,
            # The largest face left, over every row rather than the 200 the
            # record carries. The gate reads this: a file with more than 200
            # finds would otherwise be judged on the first 200 of them.
            "residual_max_px": max((row["px"] for row in faces), default=0),
            "residual_runs": runs_of([row["frame"] for row in faces]),
            "residual_screens": len(glass),
            "residual_screen_frames": len({row["frame"] for row in glass}),
            "residual_screen_max_px": max((row["px"] for row in glass), default=0),
            "residual_screen_runs": screen_runs(big, settings),
            "residual_list": kept}


def held_for(report: dict, settings: Settings) -> tuple[str, ...]:
    """Which kinds hold this copy back, in listing order. Empty means it ships.

    A face holds a copy back on its own, at `quarantine_min_px` and up: one
    frame of a face is a face. A screen has to have been there, which means a
    run of `screen_gate_min_run` checked frames among the finds that clear
    `screen_gate_min_px`. Hands hold nothing back; they are not in these
    numbers at all.
    """
    kinds = []
    if (settings.wants("face") and report.get("residual_faces")
            and report.get("residual_max_px", 0) >= settings.quarantine_min_px):
        kinds.append("face")
    if settings.wants("screen") and any(
            run["hits"] >= settings.screen_gate_min_run
            for run in report.get("residual_screen_runs") or []):
        kinds.append("screen")
    return tuple(kinds)


def check(src: Path, out: Path, settings: Optional[Settings] = None, stride: int = 1,
          submit=None, workers: int = 1, chunk: int = 300) -> dict:
    """Run the whole check. `submit` maps jobs the way batch.run_video does."""
    from .batch import serial_submit

    settings = settings or Settings()
    submit = submit or serial_submit
    jobs = plan_jobs(Path(src), Path(out), settings, chunk, stride)
    return summarise(list(submit(residual_job, jobs, workers)), settings)


def describe(report: dict, settings: Optional[Settings] = None) -> str:
    settings = settings or Settings()
    by_size = report["residual_by_size"]
    hands = report.get("residual_hands", 0)
    text = ""
    if settings.wants("face"):
        text = (f"{report['residual_faces']} faces still in the copy on untouched pixels "
                f"({by_size['40+ px']} at 40 px and over, {by_size['24-40 px']} at 24 to 40, "
                f"{by_size['under 24 px']} under 24), over {report['residual_frames']} frames "
                f"of {report['frames_checked']} checked; {report['flat_boxes']} boxes too "
                f"flat to say")
        if hands:
            text += f"; {hands} more set aside as the wearer's hands"
    if settings.wants("screen"):
        runs = report.get("residual_screen_runs") or []
        long_enough = [r for r in runs if r["hits"] >= settings.screen_gate_min_run]
        joiner = ". " if text else ""
        text += (f"{joiner}{report.get('residual_screens', 0)} screen boxes still visible "
                 f"over {report.get('residual_screen_frames', 0)} frames, the largest "
                 f"{report.get('residual_screen_max_px', 0)} px; {len(runs)} runs clear the "
                 f"{settings.screen_gate_min_px} px floor and {len(long_enough)} of those "
                 f"last {settings.screen_gate_min_run} checked frames or more")
    return text or f"nothing to check over {report['frames_checked']} frames"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("output", help="the blurred copy of it")
    ap.add_argument("--stride", type=int, default=1, help="check every Nth frame")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-hand-rule", dest="hand_rule", action="store_false",
                    help="report the wearer's hands as missed faces, the way this check "
                         "did before it could tell them apart")
    ap.add_argument("--mask", default="face",
                    help="what the run was asked to mask, comma separated (default: "
                         f"face). This check can look for {', '.join(KINDS_CHECKED)}, and "
                         "looks only for what was asked: a copy nobody asked to have "
                         "screens masked in is not missing one")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()
    from .classes import parse

    settings = Settings(hand_rule=args.hand_rule, mask=parse(args.mask))
    workers = max(1, args.workers)
    if workers == 1:
        report = check(Path(args.source), Path(args.output), settings, args.stride)
    else:
        import multiprocessing

        from .batch import PoolSubmit, init_pool_worker

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=init_pool_worker, initargs=(workers,)) as pool:
            report = check(Path(args.source), Path(args.output), settings, args.stride,
                           submit=PoolSubmit(pool), workers=workers)
    print(describe(report, settings))
    for row in report["residual_list"][:20]:
        mark = f"  hand {row['hand']}" if "hand" in row else ""
        print(f"  {row.get('kind', 'face'):6s} frame {row['frame']:6d}  {row['px']:4d} px "
              f"at ({row['x']},{row['y']})  score {row['score']:.2f}  "
              f"applied {row['applied']}{mark}")
    held = held_for(report, settings)
    if held:
        print(f"  this copy would be held back for: {', '.join(held)}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 1 if report["residual_faces"] or report.get("residual_screens") else 0


if __name__ == "__main__":
    sys.exit(main())
