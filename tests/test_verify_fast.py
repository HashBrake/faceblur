"""The check's CPU side on less of the frame, with the same answers (T03).

`residual_job` used to draw its zone masks on the whole frame, three times a
frame, and work out the encoder's noise once per detection. It now draws them
over the hands' box with `redact._zone_in` and takes the noise once a frame.
The reference copies below are the code as it was before that change, copied
verbatim from `faceblur/verify.py` at 95dccb9, and every answer is compared
with theirs.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import run_ffmpeg
from faceblur import batch, verify
from faceblur.detect import Detection
from faceblur.hands import Hand
from faceblur.settings import Settings
from faceblur.verify import (FLAT_DIFF, _row, applied, classify, covered_by_zone, frame_noise,
                             kinds_checked, plan_jobs, residual_job, zone_changed, zone_crop)
from faceblur.segments import decode_range
from faceblur.zone import alpha_for


# ------------------------------------------------------- the code as it was

def _ref_covered_by_zone(det: Detection, zone_mask) -> float:
    if zone_mask is None:
        return 0.0
    h, w = zone_mask.shape[:2]
    x0, y0 = max(0, int(det.x)), max(0, int(det.y))
    x1, y1 = min(w, int(det.x + det.w)), min(h, int(det.y + det.h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(zone_mask[y0:y1, x0:x1].mean())


def _ref_zone_changed(before: np.ndarray, after: np.ndarray, polygons,
                      noise: float) -> tuple[float, int]:
    mask = alpha_for(before.shape[:2], polygons, 0.0) >= 0.5
    total = int(mask.sum())
    if not total:
        return 0.0, 0
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    diff = np.abs(before[y0:y1, x0:x1].astype(np.float32)
                  - after[y0:y1, x0:x1].astype(np.float32)).mean(axis=2)
    moved = ((diff > noise + FLAT_DIFF) & mask[y0:y1, x0:x1]).sum()
    return float(moved) / total, total


def _reference_frame(before, after, live, quads, dets, needs_hand=True):
    """One frame's zone work as `residual_job` did it before: whole-frame masks."""
    zone_mask = face_zone = None
    moved = None
    noise = frame_noise(before, after)
    if live:
        zone_mask = alpha_for(before.shape[:2], live, 0.0) >= 0.5
        face_zone = ((alpha_for(before.shape[:2], quads, 0.0) >= 0.5)
                     if needs_hand and quads else zone_mask)
        strict = quads if needs_hand else live
        if strict:
            moved = _ref_zone_changed(before, after, strict, noise)
    return ([_ref_covered_by_zone(d, face_zone) for d in dets],
            [_ref_covered_by_zone(d, zone_mask) for d in dets], moved, noise)


def _new_frame(before, after, live, quads, dets, needs_hand=True):
    """The same frame the way `residual_job` does it now."""
    zone_mask = face_zone = None
    moved = None
    noise = frame_noise(before, after)
    if live:
        zone_mask = zone_crop(before.shape[:2], live)
        face_zone = (zone_crop(before.shape[:2], quads)
                     if needs_hand and quads else zone_mask)
        strict = quads if needs_hand else live
        if strict:
            moved = zone_changed(before, after, strict, noise,
                                 face_zone if strict is quads else zone_mask)
    return ([covered_by_zone(d, face_zone) for d in dets],
            [covered_by_zone(d, zone_mask) for d in dets], moved, noise)


def _reference_residual_job(job: dict) -> dict:
    """Detect on the copy's frames start to end and say what was never masked.

    Rows always carry source frame numbers. `out` may be the whole copy or one
    piece of it, written before the join: `out_start` is the piece's own index
    of source frame `start` (the same frame when `out` is the whole copy),
    because the join is a concat of the pieces without re-encoding and the
    check reads the same pixels either way. A `copied` job is a stretch stream
    copied from the source, which decodes to the source's pixels bit for bit,
    so the source frame is both sides and the copy is not decoded at all.
    `seconds` is the job's own time, for the record's `overlap_check_seconds`.
    """
    from faceblur.batch import _bank, _hand_rule, _screens, _zone   # the worker's cached models

    t0 = time.time()
    src = Path(job["src"])
    settings: Settings = job["settings"]
    start, end = job["start"], job["end"]
    stride = max(1, job.get("stride", 1))
    wanted = kinds_checked(settings)
    bank = _bank(settings) if "face" in wanted else None
    screens = _screens(settings) if "screen" in wanted else None
    rule = _hand_rule(settings) if bank is not None and settings.hand_rule else None
    # The detection pass already found every source frame's hands with the
    # same models and settings, so they come with the job when it has them
    # and are not found a second time. That was about half of the check.
    known = job.get("hands")
    finder = _zone(settings) if settings.zone and known is None else None
    zone_rows: list[dict] = []
    found, hands, checked, in_zone = [], 0, 0, 0
    flat = {name: 0 for name in wanted}
    # The flat boxes by frame as well as in total, so that a later check over
    # part of the file can take a frame's count back out (`merge_checks`).
    # Every other count here is either one per checked frame or one per row.
    flat_frames: dict[int, dict[str, int]] = {}
    source_frames = decode_range(src, job["times"], start, end, job["info"], settings.hwaccel)
    if job.get("copied"):
        pairs = ((frame, frame) for frame in source_frames)
    else:
        out_start = job.get("out_start", start)
        copy_frames = decode_range(Path(job["out"]), job["out_times"], out_start,
                                   out_start + (end - start), job["out_info"], settings.hwaccel)
        pairs = zip(source_frames, copy_frames)
    for offset, (before, after) in enumerate(pairs):
        i = start + offset
        if i % stride:
            continue
        checked += 1
        faces = bank.detect(after) if bank is not None else []
        glass = screens.detect(after) if screens is not None else []
        zone_mask = face_zone = None
        if settings.zone:
            # The zone is rebuilt from the source: hands are unmasked in both
            # and the source is sharper. Live polygons only; see the module
            # docstring for why the remembered ones are not checked here.
            from faceblur.zone import polygon_for, qualifies

            source_hands = known.get(i, []) if known is not None else finder.detect(before)
            mine = [hand for hand in source_hands
                    if qualifies(hand, before.shape[:2], settings)]
            live = [polygon_for(hand, settings, before.shape[:2]) for hand in mine]
            quads = [tuple(hand.grown(settings.zone_face_scale).quad) for hand in mine]
            if live:
                zone_mask = alpha_for(before.shape[:2], live, 0.0) >= 0.5
                # Faces are asked about the hands themselves, not the reach.
                face_zone = ((alpha_for(before.shape[:2], quads, 0.0) >= 0.5)
                             if settings.zone_face_needs_hand and quads
                             else zone_mask)
                # The gate measures where the promise is **absolute**, which
                # since `zone_face_needs_hand` is the hand itself rather than
                # its reach. A face may now be masked inside the reach, on
                # purpose, so measuring the reach would report the rule working
                # as the rule broken. What the reach still protects is text and
                # screens, and the check cannot rebuild the remembered half of
                # that, so it is reported by the pipeline instead as
                # `masked_in_zone_prevented`. Section 17 says so.
                strict = quads if settings.zone_face_needs_hand else live
                if strict:
                    moved, pixels = _ref_zone_changed(before, after, strict,
                                                 verify.frame_noise(before, after))
                    if moved:
                        zone_rows.append({"frame": i, "changed": round(moved, 5),
                                          "zone_px": pixels})
        if not faces and not glass:
            continue
        noise = verify.frame_noise(before, after)
        for det in faces:
            changed, would = applied(before, after, det, settings)
            call = classify(changed, would, noise)
            if call == "flat":
                flat["face"] += 1
                counts = flat_frames.setdefault(i, {})
                counts["face"] = counts.get("face", 0) + 1
            elif call == "missed":
                row = _row(det, i, changed, would, noise)
                cover = _ref_covered_by_zone(det, face_zone)
                if cover >= settings.zone_cover:
                    # Not a miss. The owner's rule says nothing handled is ever
                    # masked, so a face under the wearer's own hand is left
                    # there on purpose.
                    row["zone"] = round(cover, 3)
                    in_zone += 1
                # The hand rule is asked as well, not instead. They are two
                # mechanisms answering two questions, and how often they agree
                # is worth knowing: on the sample footage the zone contains 30
                # of the 31 boxes the hand rule calls hands, which is the zone
                # marked by somebody else's homework. Asking only one would
                # throw that away and would leave `residual_hands` meaning
                # something different from what section 14 measured.
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
                counts = flat_frames.setdefault(i, {})
                counts["screen"] = counts.get("screen", 0) + 1
            elif call == "missed":
                row = _row(det, i, changed, would, noise)
                # The label as well, so that a reader knows the detector said
                # phone rather than television. The box itself is in the row
                # already, and `summarise` groups the runs from it.
                row["label"] = det.source
                cover = _ref_covered_by_zone(det, zone_mask)
                if cover >= settings.zone_cover:
                    row["zone"] = round(cover, 3)
                    in_zone += 1
                found.append(row)
    return {"start": start, "end": end, "stride": stride, "checked": checked,
            "flat": sum(flat.values()), "flat_by_kind": flat, "flat_frames": flat_frames,
            "hands": hands, "in_zone": in_zone, "found": found, "zone": zone_rows,
            "hand_models": {} if rule is None else rule.model_hashes,
            "hand_compute": {} if rule is None else rule.compute,
            "seconds": round(time.time() - t0, 3)}


# ------------------------------------------------------------ the helpers

def square(cx, cy, half, angle=17.0):
    """A rotated square, the shape a hand's quad has."""
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    return tuple((cx + c * x - s * y, cy + s * x + c * y)
                 for x, y in ((-half, -half), (half, -half), (half, half), (-half, half)))


def clamp(poly, shape):
    """What `zone.polygon_for` does to a polygon: each corner into the frame."""
    h, w = shape
    return tuple((min(max(x, 0.0), float(w)), min(max(y, 0.0), float(h))) for x, y in poly)


def same(a, b, path="result"):
    """Deep equality, floats to 1e-9."""
    if isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1e-9), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{k}]")
    else:
        assert a == b, path


def frames(shape, seed):
    """`before` and an `after` that differs inside and outside any zone."""
    h, w = shape
    rng = np.random.default_rng(seed)
    before = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    after = before.copy()
    step = rng.integers(-30, 31, (h, w // 2, 3))
    after[:, : w // 2] = np.clip(before[:, : w // 2].astype(np.int16) + step, 0, 255)
    after[h // 3: h // 2] = 255 - after[h // 3: h // 2]
    return before, after


def centres(shape):
    """The interior, every corner and the middle of every edge."""
    h, w = shape
    return [(w / 2, h / 2), (0, 0), (w, 0), (0, h), (w, h),
            (w / 2, 0), (0, h / 2), (w, h / 2), (w / 2, h)]


def dets_around(cx, cy, shape):
    """Inside the polygons, partly outside their box, wholly outside it, and
    partly off the frame."""
    h, w = shape
    return [Detection(cx - 40.3, cy - 40.7, 80.5, 80.2, 0.9),
            Detection(cx + 60.0, cy + 60.0, 400.0, 300.0, 0.9),
            Detection(cx - 900.0, cy - 700.0, 120.0, 120.0, 0.9),
            Detection(-30.5, -20.2, 90.0, 90.0, 0.9),
            Detection(w - 50.0, h - 40.0, 120.0, 120.0, 0.9),
            Detection(cx + 2000.0, cy, 50.0, 50.0, 0.9)]


SHAPES = [(1080, 1920), (723, 1081)]


# -------------------------------------------------- the masks on a box

@pytest.mark.parametrize("shape", SHAPES)
def test_one_hand_anywhere_gives_the_whole_frame_answers(shape):
    before, after = frames(shape, 3)
    for k, (cx, cy) in enumerate(centres(shape)):
        live = [clamp(square(cx, cy, 260.0, 11.0 * k), shape)]
        quads = [square(cx, cy, 160.0, 11.0 * k)]
        dets = dets_around(cx, cy, shape)
        for needs_hand in (True, False):
            ref = _reference_frame(before, after, live, quads, dets, needs_hand)
            new = _new_frame(before, after, live, quads, dets, needs_hand)
            same(new, ref, f"centre {k}, needs_hand {needs_hand}")
        assert max(ref[0]) > 0.5, "the test must put a box inside the zone"


@pytest.mark.parametrize("shape", SHAPES)
def test_hands_far_apart_give_the_whole_frame_answers(shape):
    h, w = shape
    before, after = frames(shape, 5)
    spots = [(120.0, 140.0), (w - 100.0, h - 90.0), (w / 2, h / 2)]
    live = [clamp(square(x, y, 180.0, 30.0), shape) for x, y in spots]
    quads = [square(x, y, 110.0, 30.0) for x, y in spots]
    dets = [d for x, y in spots for d in dets_around(x, y, shape)]
    same(_new_frame(before, after, live, quads, dets),
         _reference_frame(before, after, live, quads, dets))


@pytest.mark.parametrize("shape", SHAPES)
def test_no_hands_build_nothing(shape, monkeypatch):
    before, after = frames(shape, 9)
    dets = dets_around(shape[1] / 2, shape[0] / 2, shape)
    assert zone_crop(shape, []) is None
    assert all(covered_by_zone(d, None) == 0.0 for d in dets)
    calls = []
    monkeypatch.setattr(verify, "_zone_in", lambda *a: calls.append(a))
    same(_new_frame(before, after, [], [], dets), _reference_frame(before, after, [], [], dets))
    assert not calls, "a frame with no live polygons draws nothing"


@pytest.mark.parametrize("shape", SHAPES)
def test_a_hand_wholly_off_the_frame_covers_nothing(shape):
    h, w = shape
    before, after = frames(shape, 11)
    quads = [square(w + 400.0, h + 400.0, 100.0)]
    live = [clamp(square(w + 400.0, h + 400.0, 200.0), shape)]
    dets = dets_around(w - 10.0, h - 10.0, shape)
    same(_new_frame(before, after, live, quads, dets),
         _reference_frame(before, after, live, quads, dets))
    assert zone_crop(shape, quads) is None


@pytest.mark.parametrize("shape", SHAPES)
def test_zone_changed_on_its_own_gives_the_whole_frame_answer(shape):
    before, after = frames(shape, 13)
    for k, (cx, cy) in enumerate(centres(shape)):
        polygons = [square(cx, cy, 150.0 + 7 * k, 9.0 * k), square(cx + 300, cy - 200, 60.0)]
        for noise in (0.0, 2.5, 40.0):
            same(zone_changed(before, after, polygons, noise),
                 _ref_zone_changed(before, after, polygons, noise), f"centre {k}")


# --------------------------------------------------- the job end to end

def _hands_on(i: int, shape) -> list:
    """Hands the way the detection pass hands them over, varied by frame."""
    h, w = shape

    def hand(cx, cy, palm, angle=15.0):
        return Hand(square(cx, cy, 1.3 * palm, angle), 0.9,
                    (cx - palm / 2, cy - palm / 2, palm, palm))

    kind = i % 5
    if kind == 0:
        return []
    if kind == 1:                                   # on the face
        return [hand(315.0, 345.0, 130.0)]
    if kind == 2:                                   # a corner, partly off the frame
        return [hand(20.0, h - 10.0, 140.0, 40.0)]
    if kind == 3:                                   # two, far apart
        return [hand(w - 10.0, 10.0, 130.0), hand(w / 2, h - 5.0, 150.0, 70.0)]
    # off the frame entirely, and one too small to be the wearer's
    return [hand(w + 300.0, h + 300.0, 130.0), hand(100.0, 60.0, 30.0)]


@pytest.fixture(scope="module")
def marked(tmp_path_factory, video_with_face):
    """The face clip re-encoded, not masked, and painted where a hand is."""
    out = tmp_path_factory.mktemp("t03") / "marked.mp4"
    result = run_ffmpeg(["-y", "-loglevel", "error", "-i", str(video_with_face),
                         "-vf", "drawbox=x=400:y=400:w=150:h=150:color=red:t=fill",
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                         "-pix_fmt", "yuv420p", str(out)])
    assert result.returncode == 0, result.stderr[-400:]
    return out


def _job(src, out, settings):
    shape = (640, 640)
    hands = {i: _hands_on(i, shape) for i in range(10) if _hands_on(i, shape)}
    [job] = plan_jobs(Path(src), Path(out), settings, chunk=1000, hands=hands)
    return job


def _compare(job):
    new, ref = residual_job(job), _reference_residual_job(job)
    new.pop("seconds"), ref.pop("seconds")
    same(new, ref)
    return new


def test_the_job_gives_what_it_gave_before(video_with_face, marked):
    for needs_hand in (True, False):
        settings = Settings(device="cpu", zone_face_needs_hand=needs_hand)
        result = _compare(_job(video_with_face, marked, settings))
        assert result["checked"] == 10 and result["found"], "the face must be found"
        assert any("zone" in row for row in result["found"]), "a find inside the zone"
        assert result["zone"], "the painted box must move the zone"


class _Bank:
    """Three finds a frame: on the face, partly off the frame, and far from any hand."""

    def detect(self, frame):
        return [Detection(240.0, 240.0, 160.0, 160.0, 0.9),
                Detection(-40.0, 560.0, 120.0, 120.0, 0.9),
                Detection(420.0, 20.0, 90.0, 90.0, 0.9)]


class _Screens:
    def detect(self, frame):
        return [Detection(250.0, 260.0, 140.0, 120.0, 0.8, None, "tv", True, 0.8, "screen"),
                Detection(560.0, -30.0, 200.0, 120.0, 0.8, None, "laptop", True, 0.8, "screen")]


def test_the_job_with_many_finds_and_screens_gives_what_it_gave_before(video_with_face,
                                                                        marked, monkeypatch):
    monkeypatch.setattr(batch, "_bank", lambda settings: _Bank())
    monkeypatch.setattr(batch, "_screens", lambda settings: _Screens())
    settings = Settings(device="cpu", mask=("face", "screen"), hand_rule=False)
    result = _compare(_job(video_with_face, marked, settings))
    assert any(row["kind"] == "screen" and "zone" in row for row in result["found"])


def test_the_noise_is_taken_once_a_frame(video_with_face, marked, monkeypatch):
    monkeypatch.setattr(batch, "_bank", lambda settings: _Bank())
    calls = []
    real = verify.frame_noise

    def counting(before, after, *a):
        calls.append(1)
        return real(before, after, *a)

    monkeypatch.setattr(verify, "frame_noise", counting)
    settings = Settings(device="cpu", hand_rule=False)
    result = residual_job(_job(video_with_face, marked, settings))
    assert result["checked"] == 10
    assert len(calls) == 10, "three finds and a zone on a frame, one noise"


def test_zone_changed_reads_the_floor_as_the_float_mean_did_at_every_level():
    # The floor is read off the integer channel sums. Every sum there is,
    # against floors on, just below and just above each mean there can be.
    rng = np.random.default_rng(17)
    before = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    after = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    after[0, :3] = before[0, :3]                       # a sum of zero
    after[1, 0], before[1, 0] = (255, 255, 255), (0, 0, 0)   # and of 765
    polygons = [((0, 0), (96, 0), (96, 96), (0, 96))]
    means = np.arange(766, dtype=np.float32) / np.float32(3)
    floors = [float(m) for m in means[::7]] + [float(np.nextafter(m, 0)) for m in means[::11]]
    floors += [float(np.nextafter(m, 1000)) for m in means[::13]] + [-1.0, 300.0]
    for floor in floors:
        noise = floor - FLAT_DIFF
        same(zone_changed(before, after, polygons, noise),
             _ref_zone_changed(before, after, polygons, noise), f"floor {floor!r}")
