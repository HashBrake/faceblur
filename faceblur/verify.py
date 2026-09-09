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

One question is asked of everything that survives: is it a hand? The same
detectors that produced the copy call the wearer's hand a face, and the
track-level rules that keep hands out of the mask have no counterpart in a
check that reads one frame at a time. `faceblur/hands.py` answers it with
MediaPipe's palm detector and its landmark model. A box a confirmed hand
covers stays in the record, marked with the coverage that decided it, and is
left out of every count a decision hangs on.

    .venv\\Scripts\\python.exe -m faceblur.verify SOURCE OUTPUT [--stride N]
                                                  [--workers N] [--no-hand-rule]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .detect import Detection
from .redact import ellipse_for, redact
from .segments import decode_range, frame_times
from .settings import Settings
from .video import VideoError, probe

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
    """
    control, _ = redact(before, [det], settings)
    e = ellipse_for(det, settings)
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


def residual_job(job: dict) -> dict:
    """Detect on the copy's frames start to end and say what was never masked."""
    from .batch import _bank, _hand_rule    # the worker's cached models

    src, out = Path(job["src"]), Path(job["out"])
    settings: Settings = job["settings"]
    start, end = job["start"], job["end"]
    stride = max(1, job.get("stride", 1))
    bank = _bank(settings)
    rule = _hand_rule(settings) if settings.hand_rule else None
    found, hands, flat, checked = [], 0, 0, 0
    source_frames = decode_range(src, job["times"], start, end, job["info"], settings.hwaccel)
    copy_frames = decode_range(out, job["out_times"], start, end, job["out_info"],
                               settings.hwaccel)
    for offset, (before, after) in enumerate(zip(source_frames, copy_frames)):
        i = start + offset
        if i % stride:
            continue
        checked += 1
        dets = bank.detect(after)
        noise = frame_noise(before, after) if dets else 0.0
        for det in dets:
            changed, would = applied(before, after, det, settings)
            call = classify(changed, would, noise)
            if call == "flat":
                flat += 1
            elif call == "missed":
                over = max(0.0, would - noise)
                row = {"frame": i, "px": round(det.long_side),
                       "x": round(det.cx), "y": round(det.cy),
                       "score": round(det.score, 3),
                       "applied": round(max(0.0, changed - noise) / over, 3) if over else None}
                if rule is not None:
                    # The hand rule reads the source, not the copy. Nothing
                    # masked a hand, so it looks the same in both, and the
                    # source is the sharper of the two.
                    cover = rule.hand_cover(before, det)
                    if cover >= settings.hand_cover:
                        row["hand"] = round(cover, 3)
                        hands += 1
                found.append(row)
    return {"start": start, "end": end, "checked": checked, "flat": flat,
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


def summarise(results: list[dict]) -> dict:
    """The audit record's view: how many faces the copy still shows, and where.

    Hands stay in `residual_list`, each marked with the coverage that decided
    it, because a rule that quietly dropped what it disagreed with would be
    worth nothing to an auditor. Every count a decision hangs on leaves them
    out.
    """
    found = [row for r in results for row in r["found"]]
    found.sort(key=lambda row: (row["frame"], -row["px"]))
    faces = not_hands(found)
    checked = sum(r["checked"] for r in results)
    by_size = {"40+ px": 0, "24-40 px": 0, "under 24 px": 0}
    for row in faces:
        key = ("40+ px" if row["px"] >= 40 else
               "24-40 px" if row["px"] >= 24 else "under 24 px")
        by_size[key] += 1
    models, compute = {}, {}
    for r in results:
        models.update(r.get("hand_models") or {})
        compute.update(r.get("hand_compute") or {})
    return {"frames_checked": checked,
            "hand_models": models,
            "hand_compute": compute,
            "flat_boxes": sum(r["flat"] for r in results),
            "residual_faces": len(faces),
            "residual_hands": sum(r.get("hands", 0) for r in results),
            "residual_frames": len({row["frame"] for row in faces}),
            "residual_by_size": by_size,
            # The largest face left, over every row rather than the 200 the
            # record carries. The gate reads this: a file with more than 200
            # finds would otherwise be judged on the first 200 of them.
            "residual_max_px": max((row["px"] for row in faces), default=0),
            "residual_runs": runs_of([row["frame"] for row in faces]),
            "residual_list": found[:200]}


def check(src: Path, out: Path, settings: Optional[Settings] = None, stride: int = 1,
          submit=None, workers: int = 1, chunk: int = 300) -> dict:
    """Run the whole check. `submit` maps jobs the way batch.run_video does."""
    from .batch import serial_submit

    settings = settings or Settings()
    submit = submit or serial_submit
    jobs = plan_jobs(Path(src), Path(out), settings, chunk, stride)
    return summarise(list(submit(residual_job, jobs, workers)))


def describe(report: dict) -> str:
    by_size = report["residual_by_size"]
    hands = report.get("residual_hands", 0)
    return (f"{report['residual_faces']} faces still in the copy on untouched pixels "
            f"({by_size['40+ px']} at 40 px and over, {by_size['24-40 px']} at 24 to 40, "
            f"{by_size['under 24 px']} under 24), over {report['residual_frames']} frames "
            f"of {report['frames_checked']} checked; {report['flat_boxes']} boxes too flat "
            f"to say" + (f"; {hands} more set aside as the wearer's hands" if hands else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("output", help="the blurred copy of it")
    ap.add_argument("--stride", type=int, default=1, help="check every Nth frame")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-hand-rule", dest="hand_rule", action="store_false",
                    help="report the wearer's hands as missed faces, the way this check "
                         "did before it could tell them apart")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()
    settings = Settings(hand_rule=args.hand_rule)
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
    print(describe(report))
    for row in report["residual_list"][:20]:
        mark = f"  hand {row['hand']}" if "hand" in row else ""
        print(f"  frame {row['frame']:6d}  {row['px']:4d} px at ({row['x']},{row['y']})  "
              f"score {row['score']:.2f}  applied {row['applied']}{mark}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 1 if report["residual_faces"] else 0


if __name__ == "__main__":
    sys.exit(main())
