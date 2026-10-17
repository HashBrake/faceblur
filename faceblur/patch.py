"""Blur what the check still finds in a held back copy, and nothing else.

On the production run the gate held back 289 of 321 copies for faces left in
them. A spot check found most of those finds are false alarms and some are
real bystanders, and the owner approved a patch pass rather than a second
full run: for each held back copy, run the check again on (source, copy) and
mask every face row it reports that is not set aside as the wearer's hand or
as inside the handled zone, whatever its size.

The check is the one in `verify.py`, run over every frame by the same jobs
and the same worker pool the pipeline uses. Only the collection differs: the
record keeps the first 200 rows of each kind, and a patch that masked only
those would leave the rest, so the rows are taken straight from the jobs.

The mask is drawn on the **copy**, not the source. Everything the first run
masked stays masked, the frames nobody touches come through as they were,
and the new masks go on with the pipeline's own `redact`, so a patched face
looks like every other face in the file. The write is the pipeline's own
write pass (`batch._write_and_join`): the same encoder and its fallback, the
same timing, stretches with no new mask copied from the copy byte for byte,
the same join and the same verification, with no audio track.

    .venv\\Scripts\\python.exe -m faceblur.patch SOURCE COPY OUTPUT [--workers N]
                                                 [--hold 2] [--mask face]

The record goes to OUTPUT.json. A copy with nothing to mask is copied to
OUTPUT byte for byte, so whatever reads the output always finds a file.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from . import __version__
from .detect import Detection, iou
from .segments import frame_times, keyframes, plan_segments, reorder_delay, split_long
from .settings import Settings
from .timestamps import SourceTiming, retime, same_timing, source_timing
from .verify import detection_from_row, not_hands, of_kind, plan_jobs, residual_job, runs_of
from .video import VideoError, has_audio, probe

# Frames a patch mask is held either side of the frame its row was found on.
HOLD = 2


def find_rows(src: Path, copy: Path, settings: Settings, submit, workers: int,
              chunk: int) -> tuple[list[dict], dict]:
    """Every row the check finds in the copy, over every frame, uncapped.

    The same jobs `verify.check` submits, at stride 1, with no hands handed
    in: the detection pass that found them is long gone, so each job finds
    the hands itself, which is what the check did on any run whose zone
    stride was not 1. `summarise` is not called because it is what caps the
    list; the counts this module needs are taken from the rows themselves.
    """
    jobs = plan_jobs(Path(src), Path(copy), settings, chunk, stride=1)
    results = list(submit(residual_job, jobs, workers))
    rows = [row for r in results for row in r["found"]]
    rows.sort(key=lambda row: (row["frame"], -row["px"]))
    extra = {"frames_checked": sum(r["checked"] for r in results),
             "flat_boxes": sum(r["flat"] for r in results),
             "hand_models": {}, "hand_compute": {}}
    for r in results:
        extra["hand_models"].update(r.get("hand_models") or {})
        extra["hand_compute"].update(r.get("hand_compute") or {})
    return rows, extra


def to_mask(rows: list[dict]) -> list[dict]:
    """The rows a patch masks: every face the check did not set aside.

    The same two exclusions the gate makes (`verify.not_hands`): a box
    MediaPipe says is the wearer's hand is not a face, and a find inside the
    handled zone was left there on purpose, because nothing the wearer
    handles is ever masked. No size floor: the owner asked for every face
    left, and a box too small to hold a copy back can still be a face.
    Screens and other kinds are not patched; this pass is about faces.
    """
    return not_hands(of_kind(rows, "face"))


def held(rows: list[dict], n: int, hold: int = HOLD) -> dict[int, list[Detection]]:
    """The boxes to mask, by frame, each held `hold` frames either side.

    The detector blinks: a face the check caught on frame N it may have
    passed over on N-1 and N+1, where it is just as visible. So each box is
    also put on the frames around it, the same box, unmoved, because nothing
    here knows which way the face went; the second chance in `batch.run_video`
    holds its seeds the same way for the same reason. A frame that already
    has a box overlapping the held one (IoU 0.3, the tracker's own test) does
    not get a second, so a face found on consecutive frames is masked once
    per frame, not two or three times over.

    The hold does not ask whether the frames it lands on are inside the
    handled zone. A row in the zone is never masked, but a row outside it
    could, two frames later, have a hand move over it. Keeping the rule here
    would mean rebuilding the zone for every held frame for a case that lasts
    at most `hold` frames and only ever masks too much rather than too little.
    """
    out: dict[int, list[Detection]] = {}
    found = [(int(row["frame"]), detection_from_row(row)) for row in rows]
    # Each row's own frame first, then the holds, so a box held onto a frame
    # never stands in for the box that frame was actually found with.
    for i, det in found:
        if 0 <= i < n:
            out.setdefault(i, []).append(det)
    for i, det in found:
        for k in range(i - hold, i + hold + 1):
            if k == i or not 0 <= k < n:
                continue
            here = out.setdefault(k, [])
            if not any(iou(det, other) >= 0.3 for other in here):
                here.append(det)
    return {k: v for k, v in out.items() if v}


def by_size(rows: list[dict]) -> dict[str, int]:
    """The size buckets `verify.summarise` uses, so the two records read alike."""
    out = {"40+ px": 0, "24-40 px": 0, "under 24 px": 0}
    for row in rows:
        key = ("40+ px" if row["px"] >= 40 else
               "24-40 px" if row["px"] >= 24 else "under 24 px")
        out[key] += 1
    return out


def write(copy: Path, output: Path, per_frame: list[list[Detection]], settings: Settings,
          submit, workers: int, timing: Optional[Path] = None) -> dict:
    """Mask `per_frame` on the copy and write `output`: the pipeline's write pass.

    The same steps `batch.run_video` takes, in the same order, with the copy
    standing where the source stands there: the rate the timestamps show
    rather than the container's, segments cut at the copy's keyframes,
    clean stretches copied when the copy has no reordering, and the whole
    file encoded if the mixed join does not verify. `zones` is not passed:
    rows in the zone were left out already (see `to_mask` and `held`).

    `timing` is the original source: the output carries its frame
    timestamps, not the copy's (which may be on the encoder's constant
    rate). None keeps the copy's.
    """
    from .batch import _write_and_join

    info, times = probe(copy), frame_times(copy)
    n = len(times)
    if n > 1 and times[-1] > times[0]:
        info = replace(info, fps=(n - 1) / (times[-1] - times[0]))
    chunk = max(1, int(round(settings.chunk_seconds * info.fps))) if settings.chunk_seconds else n
    keys = keyframes(copy, times)
    min_copy = max(1, int(round(settings.min_copy_seconds * info.fps)))
    piece = max(1, int(round(settings.encode_seconds * info.fps))) if settings.encode_seconds else chunk
    want_copy = settings.copy_clean and reorder_delay(copy) == 0
    flags = [bool(f) for f in per_frame]
    out = {"frames_copied": 0, "join_attempts": 0, "error": ""}
    for attempt, copy_clean in enumerate((want_copy, False)):
        if attempt and not want_copy:
            break
        segments = split_long(plan_segments(flags, keys, n, min_copy, copy_clean), keys, piece)
        work = tempfile.mkdtemp(prefix="faceblur_", dir=str(output.parent))
        try:
            result = _write_and_join(copy, output, info, times, settings, segments, per_frame,
                                     submit, workers, Path(work), lambda *a: None,
                                     lambda: False, None, audio=False, timing=timing)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        _, _, _, problem, copied = result
        out["join_attempts"] = attempt + 1
        out["error"] = problem
        if not problem:
            out["frames_copied"] = copied
            out["retimed"] = timing is not None
            break
        output.unlink(missing_ok=True)
    return out


def patch(src: Path, copy: Path, output: Path, settings: Optional[Settings] = None,
          hold: int = HOLD, submit=None, workers: int = 1,
          rows: Optional[list[dict]] = None) -> dict:
    """Check the copy, mask what it still shows, write OUTPUT and its record.

    `rows` skips the check and masks these instead; the tests use it, so they
    need no detector. Returns the record, which is also written beside OUTPUT.
    """
    from .batch import serial_submit

    settings = settings or Settings()
    submit = submit or serial_submit
    src, copy, output = Path(src), Path(copy), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {"source": src.name, "copy": copy.name, "output": output.name,
              "status": "failed", "frames": 0, "hold": hold,
              "mask": list(settings.mask),
              "frames_checked": 0, "rows_found": 0, "rows_face": 0, "masked": 0,
              "set_aside_hand": 0, "set_aside_zone": 0, "rows_other_kinds": 0,
              "flat_boxes": 0, "frames_found": 0, "frames_touched": 0,
              "masked_by_size": {}, "masked_max_px": 0, "masked_runs": [],
              "masked_list": [], "frames_copied": 0, "join_attempts": 0,
              "copy_audio": False, "output_audio": False, "retimed": False,
              "hand_models": {}, "compute": {},
              "check_seconds": 0.0, "encode_seconds": 0.0, "wall_seconds": 0.0,
              "settings": settings.to_dict(), "faceblur_version": __version__,
              "error": ""}
    try:
        times = frame_times(copy)
        n = len(times)
        if n == 0:
            raise VideoError("Could not read the copy.")
        record["frames"] = n
        record["copy_audio"] = has_audio(copy)
        t0 = time.time()
        if rows is None:
            fps = (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else 30.0
            chunk = (max(1, int(round(settings.chunk_seconds * fps)))
                     if settings.chunk_seconds else n)
            rows, extra = find_rows(src, copy, settings, submit, workers, chunk)
            record["frames_checked"] = extra["frames_checked"]
            record["flat_boxes"] = extra["flat_boxes"]
            record["hand_models"] = extra["hand_models"]
            record["compute"] = extra["hand_compute"]
        record["check_seconds"] = round(time.time() - t0, 2)
        faces = of_kind(rows, "face")
        chosen = to_mask(rows)
        record["rows_found"] = len(rows)
        record["rows_face"] = len(faces)
        record["rows_other_kinds"] = len(rows) - len(faces)
        record["set_aside_hand"] = sum(1 for row in faces if "hand" in row)
        # A row can carry both marks; it is counted once, as a hand.
        record["set_aside_zone"] = sum(1 for row in faces if "zone" in row and "hand" not in row)
        record["masked"] = len(chosen)
        record["frames_found"] = len({row["frame"] for row in chosen})
        record["masked_by_size"] = by_size(chosen)
        record["masked_max_px"] = max((row["px"] for row in chosen), default=0)
        record["masked_runs"] = runs_of([row["frame"] for row in chosen])
        # Coordinates and counts, never pixels, and capped the way the check's
        # own list is; the counts above are over every row.
        record["masked_list"] = chosen[:200]
        t1 = time.time()
        if not chosen:
            # Nothing to mask: the copy goes through as it is, byte for byte,
            # so the output always exists and is exactly what was checked.
            # Unless it is not on the source's clock: then its packets go
            # through unchanged, remuxed with the source's timestamps.
            ref = source_timing(src)
            if len(ref) >= n and same_timing(copy, SourceTiming(ref.time_base, ref.pts[:n],
                                                                 ref.last_duration)):
                part = Path(str(output) + ".part")
                shutil.copyfile(copy, part)
                part.replace(output)
            else:
                retime(copy, src, output, frames=n)
                record["retimed"] = True
        else:
            boxes = held(chosen, n, hold)
            record["frames_touched"] = len(boxes)
            per_frame = [boxes.get(i, []) for i in range(n)]
            got = write(copy, output, per_frame, settings, submit, workers, timing=src)
            record.update(got)
            if got["error"]:
                raise VideoError(got["error"])
        record["encode_seconds"] = round(time.time() - t1, 2)
        record["output_audio"] = has_audio(output)
        record["status"] = "done"
    except Exception as exc:          # WorkerLost too: the record says why, always
        record["error"] = record["error"] or str(exc) or type(exc).__name__
        record["status"] = "failed"
        output.unlink(missing_ok=True)
    record["wall_seconds"] = round(time.time() - started, 2)
    Path(str(output) + ".json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def describe(record: dict) -> str:
    if record["error"]:
        return f"failed: {record['error']}"
    sizes = record["masked_by_size"] or by_size([])
    return (f"{record['masked']} face rows masked on {record['frames_found']} frames, "
            f"{record['frames_touched']} frames touched with a hold of {record['hold']} "
            f"({sizes['40+ px']} at 40 px and over, {sizes['24-40 px']} at 24 to 40, "
            f"{sizes['under 24 px']} under 24); set aside {record['set_aside_hand']} as hands "
            f"and {record['set_aside_zone']} inside the zone, of {record['rows_found']} rows "
            f"over {record['frames']} frames")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="the video the copy was made from")
    ap.add_argument("copy", help="the blurred copy the gate held back")
    ap.add_argument("output", help="where the patched copy goes; its record goes beside it")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--hold", type=int, default=HOLD,
                    help=f"frames each mask is held before and after its row (default {HOLD})")
    ap.add_argument("--mask", default="face",
                    help="what the run was asked to mask, comma separated (default: face); "
                         "the check looks for those, and only faces are patched")
    args = ap.parse_args()
    from .classes import parse

    settings = Settings(mask=parse(args.mask))
    workers = max(1, args.workers)
    hold = max(0, args.hold)
    if workers == 1:
        record = patch(Path(args.source), Path(args.copy), Path(args.output), settings, hold)
    else:
        import multiprocessing

        from .batch import PoolSubmit, init_pool_worker

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=init_pool_worker, initargs=(workers,)) as pool:
            record = patch(Path(args.source), Path(args.copy), Path(args.output), settings,
                           hold, submit=PoolSubmit(pool), workers=workers)
    print(describe(record))
    return 1 if record["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
