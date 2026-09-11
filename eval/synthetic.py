"""Recall on faces whose position is known by construction. No labels.

Face crops come from the footage itself: pseudo-faces both detectors scored at
0.8 or more. Each crop is pasted into other frames of the same video at a range
of sizes, moving across a short sequence, with optional motion blur. The true
box and landmarks follow the paste, so recall is exact.

Three kinds of sequence:

- interior: faces drifting inside the picture (the original set)
- edge: faces that start 30 to 70 percent outside the picture and come in
  at 10 to 40 pixels a frame, the way a person enters the field of view
- pan: the whole picture slides 8 to 25 pixels a frame, faces with it, the
  way a wearable camera turns

    .venv\\Scripts\\python.exe -m eval.synthetic VIDEO
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval.common import (CACHE_DIR, REFERENCE, det_from_dict, ellipse_mask,
                         fingerprint, load_json, read_frames, size_bucket)
from faceblur.detect import Detection, DetectorBank, filter_candidates, weak_candidates
from faceblur.motion import downscale, estimate_shift
from faceblur.pipeline import tracker_for
from faceblur.redact import build_alpha, ellipse_for
from faceblur.settings import Settings

SIZES = [24, 32, 48, 64, 96, 128]
BLURS = [0, 0, 7, 13]
SEQ_LEN = 15
KINDS = [("interior", 36), ("edge", 18), ("pan", 18)]
# 3 carries the stamp of the pseudo-labels the faces were cut from, so that a
# set built on labels that have since moved is rebuilt instead of read. Same
# reason as the raw cache's fingerprint in eval/common.py.
VERSION = 3


def data_path(video: Path, settings: Settings | None = None) -> Path:
    """Where the pasted face set for these detection settings lives.

    The name carries the detection fingerprint because the set holds the
    detector's answers on every pasted frame and not only the pastes. A set
    built with `engine yunet` at 1280 and 1920 cannot score `engine both` or
    a 2560 scan: it would quietly report the answer of the detector it was
    built with, and package F2 sweeps exactly those settings.
    """
    stamp = fingerprint(settings or Settings())
    return CACHE_DIR / f"{video.stem}.synthetic.v{VERSION}.{stamp}.pkl"


def stamp_of(consensus: dict) -> str:
    return hashlib.sha256(json.dumps(consensus, sort_keys=True).encode()).hexdigest()[:16]


def load(video: Path, settings: Settings | None = None) -> list:
    """The sequences on disk, whatever they were built from."""
    return pickle.loads(data_path(video, settings).read_bytes())["data"]


def face_bank(video: Path, consensus: dict, limit: int = 24) -> list[dict]:
    """Crops of confident pseudo-faces, with box and landmarks relative to the crop."""
    picks = []
    for key, faces in sorted(consensus["frames"].items(), key=lambda kv: int(kv[0])):
        for f in faces:
            if f.get("source") == "yunet" and f["score"] >= 0.8 and f.get("landmarks") \
                    and max(f["w"], f["h"]) >= 48 and f.get("support", 0) >= 2:
                picks.append((int(key), f))
    # Spread picks across the video rather than taking one face many times.
    picks = picks[:: max(1, len(picks) // limit)][:limit]
    wanted = {i for i, _ in picks}
    frames = {i: fr for i, fr in read_frames(video, wanted)}
    bank = []
    for i, f in picks:
        fr = frames[i]
        m = 0.35
        x0 = int(max(0, f["x"] - m * f["w"]))
        y0 = int(max(0, f["y"] - m * f["h"]))
        x1 = int(min(fr.shape[1], f["x"] + f["w"] * (1 + m)))
        y1 = int(min(fr.shape[0], f["y"] + f["h"] * (1 + m)))
        crop = fr[y0:y1, x0:x1].copy()
        bank.append({
            "crop": crop,
            "box": [f["x"] - x0, f["y"] - y0, f["w"], f["h"]],
            "landmarks": [[px - x0, py - y0] for px, py in f["landmarks"]],
        })
    return bank


def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    if length < 3:
        return img
    k = np.zeros((length, length), np.float32)
    c = length // 2
    dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for t in np.linspace(-c, c, length * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1
    k /= max(1.0, k.sum())
    return cv2.filter2D(img, -1, k)


def paste(frame: np.ndarray, face: dict, long_side: float, cx: float, cy: float,
          blur: int, angle: float) -> Detection:
    """Paste a face so its box long side is `long_side`, centred at (cx, cy).
    The face may lie partly outside the picture; the truth box keeps its
    full extent. Returns None when nothing of it lands in the picture."""
    bx, by, bw, bh = face["box"]
    s = long_side / max(bw, bh)
    crop = cv2.resize(face["crop"], None, fx=s, fy=s,
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ch, cw = crop.shape[:2]
    # Where the box lands in the frame.
    box_cx, box_cy = (bx + bw / 2) * s, (by + bh / 2) * s
    ox, oy = cx - box_cx, cy - box_cy
    # Blend through a soft ellipse a little larger than the face box.
    alpha = np.zeros((ch, cw), np.float32)
    cv2.ellipse(alpha, (int(box_cx), int(box_cy)),
                (int(0.75 * bw * s), int(0.8 * bh * s)), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.08 * long_side))
    if blur:
        crop = motion_blur(crop, blur, angle)
        alpha = motion_blur(alpha, blur, angle)
    H, W = frame.shape[:2]
    x0, y0 = int(round(ox)), int(round(oy))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + cw), min(H, y0 + ch)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    cx0, cy0 = fx0 - x0, fy0 - y0
    a = alpha[cy0:cy0 + (fy1 - fy0), cx0:cx0 + (fx1 - fx0)][:, :, None]
    src = crop[cy0:cy0 + (fy1 - fy0), cx0:cx0 + (fx1 - fx0)].astype(np.float32)
    dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (src * a + dst * (1 - a)).astype(np.uint8)
    lm = tuple((px * s + ox, py * s + oy) for px, py in face["landmarks"])
    return Detection(bx * s + ox, by * s + oy, bw * s, bh * s, 1.0, lm, "truth")


def translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The picture slid by (dx, dy), the uncovered strip filled with the edge."""
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32)
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_BANK = None
_FACES = None
_VIDEO = None


def _init(video, faces, settings=None):
    global _BANK, _FACES, _VIDEO
    _BANK = DetectorBank(settings or Settings())
    _FACES = faces
    _VIDEO = video


def _sequence(task):
    seed, background, long_side, blur, kind = task
    rng = random.Random(seed)
    frames = {i: f for i, f in read_frames(Path(_VIDEO), {background})}
    base = frames[background]
    H, W = base.shape[:2]
    faces = rng.sample(_FACES, k=min(2, len(_FACES)))
    angle = rng.uniform(0, 180)
    pan = (0.0, 0.0)
    if kind == "interior":
        starts = [(rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.7) * H) for _ in faces]
        vel = [(rng.uniform(-5, 5), rng.uniform(-3, 3)) for _ in faces]
    elif kind == "edge":
        starts, vel = [], []
        for _ in faces:
            side = rng.choice(["left", "right", "top", "bottom"])
            outside = rng.uniform(0.3, 0.7) * long_side      # of the box lies outside at first
            speed = rng.uniform(10, 40)
            drift = rng.uniform(-3, 3)
            along = rng.uniform(0.2, 0.8)
            if side == "left":
                starts.append((-outside + long_side / 2, along * H))
                vel.append((speed, drift))
            elif side == "right":
                starts.append((W + outside - long_side / 2, along * H))
                vel.append((-speed, drift))
            elif side == "top":
                starts.append((along * W, -outside + long_side / 2))
                vel.append((drift, speed))
            else:
                starts.append((along * W, H + outside - long_side / 2))
                vel.append((drift, -speed))
    else:   # pan: the scene slides, the faces sit still in it
        pan = (rng.choice([-1, 1]) * rng.uniform(8, 25), rng.uniform(-6, 6))
        starts = [(rng.uniform(0.25, 0.75) * W, rng.uniform(0.2, 0.7) * H) for _ in faces]
        vel = [(pan[0], pan[1]) for _ in faces]
    out = []
    shifts = []
    prev_small = None
    for t in range(SEQ_LEN):
        frame = translate(base, pan[0] * t, pan[1] * t) if kind == "pan" else base.copy()
        truth = []
        for face, (sx, sy), (vx, vy) in zip(faces, starts, vel):
            d = paste(frame, face, long_side, sx + vx * t, sy + vy * t, blur, angle)
            if d is not None:
                truth.append(d)
        small = downscale(frame)
        shifts.append(estimate_shift(prev_small, small, max(H, W)))
        prev_small = small
        out.append((_BANK.detect_raw(frame), truth))
    return {"seed": seed, "long_side": long_side, "blur": blur, "kind": kind,
            "shape": (H, W), "frames": out, "shifts": shifts, "pan": pan}


def task_list(consensus: dict) -> list[tuple]:
    """One task per sequence: seed, background frame, size, blur, kind.

    Its own function because the sequences on disk carry their seed and not
    the frame they were pasted into, and `zone_bias` needs that frame to ask
    where the wearer's hands were. The generator is seeded, so replaying this
    recovers the mapping exactly rather than approximately.
    """
    consensus_frames = [int(k) for k in consensus["frames"]]
    rng = random.Random(7)
    tasks = []
    seed = 0
    for kind, count in KINDS:
        for s in range(count):
            tasks.append((seed, rng.choice(consensus_frames), SIZES[s % len(SIZES)],
                          BLURS[(s // len(SIZES)) % len(BLURS)], kind))
            seed += 1
    return tasks


def generate(video: Path, consensus: dict, workers: int | None = None,
             settings: Settings | None = None) -> list:
    from faceblur.detect import default_workers
    workers = workers or default_workers()
    path = data_path(video, settings)
    stamp = stamp_of(consensus)
    if path.exists():
        held = pickle.loads(path.read_bytes())
        if held.get("stamp") == stamp:
            return held["data"]
        path.unlink()          # the pseudo-labels these were cut from have moved
    faces = face_bank(video, consensus)
    if not faces:
        raise SystemExit("no confident faces to build a bank from")
    tasks = task_list(consensus)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(str(video), faces, settings)) as pool:
        result = list(pool.imap_unordered(_sequence, tasks))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(pickle.dumps({"version": VERSION, "stamp": stamp, "data": result}))
    os.replace(part, path)
    return result


def _inside_share(shape, d: Detection) -> float:
    """How much of the identity ellipse lies inside the picture."""
    e = ellipse_for(d, REFERENCE)
    full = math.pi * max(1.0, e.ax) * max(1.0, e.ay)
    return float(ellipse_mask(shape, d, REFERENCE).sum()) / full


def evaluate(settings: Settings, data: list, cover: float = 0.9) -> dict:
    """Recall per size, blur and kind, the entry delay at the frame edge, and
    how much larger the mask is than the target.

    A face counts as covered when the mask's alpha averages at least `cover`
    over its identity ellipse. A face counts at all only when at least half
    of that ellipse is inside the picture. The entry delay is how many frames
    after a face became half visible its mask arrived; SEQ_LEN if never.
    """
    hit = defaultdict(lambda: [0, 0])
    area_ratio = []
    coverage = []
    delays = []
    for seq in data:
        shape = seq["shape"]
        H, W = shape
        per = [filter_candidates(raw, settings, shape) for raw, _ in seq["frames"]]
        weak = [weak_candidates(raw, settings, shape) for raw, _ in seq["frames"]]
        shifts = seq.get("shifts") if settings.camera_comp else None
        tracked, _ = tracker_for(settings).run(per, weak, shifts, shape)
        kind = seq.get("kind", "interior")
        first_visible: dict[int, int] = {}
        first_covered: dict[int, int] = {}
        for i, (raw, truth) in enumerate(seq["frames"]):
            alpha = build_alpha(shape, tracked[i], settings) if tracked[i] else None
            for j, d in enumerate(truth):
                ident = ellipse_mask(shape, d, REFERENCE)
                if not ident.any() or _inside_share(shape, d) < 0.5:
                    continue
                first_visible.setdefault(j, i)
                c = float(alpha[ident].mean()) if alpha is not None else 0.0
                coverage.append(c)
                ok = c >= cover
                if ok:
                    first_covered.setdefault(j, i)
                for k in (size_bucket(d.long_side), "blurred" if seq["blur"] else "sharp",
                          kind, "40+" if d.long_side >= 40 else "under 40", "all"):
                    hit[k][1] += 1
                    hit[k][0] += ok
                if ok:
                    x0, y0 = max(0, int(d.x - d.w)), max(0, int(d.y - d.h))
                    x1, y1 = min(W, int(d.x + 2 * d.w)), min(H, int(d.y + 2 * d.h))
                    near = (alpha[y0:y1, x0:x1] >= 0.5).sum()
                    area_ratio.append(float(near) / max(1, ident.sum()))
        if kind == "edge":
            for j, seen in first_visible.items():
                delays.append(first_covered.get(j, seen + SEQ_LEN) - seen)
    return {
        "recall": {k: [v[0], v[1], v[0] / v[1]] for k, v in sorted(hit.items())},
        "mean_coverage": float(np.mean(coverage)) if coverage else None,
        "mask_over_target_area": float(np.median(area_ratio)) if area_ratio else None,
        "entry_delay_mean": float(np.mean(delays)) if delays else None,
        "entry_covered_at_once": float(np.mean([d == 0 for d in delays])) if delays else None,
    }


def zone_bias(video: Path, data: list, consensus: dict,
              settings: Settings | None = None) -> dict:
    """How much of the pasted face recall the handled zone would take back.

    This set pastes faces wherever the sequence puts them, and it builds its
    masks without the zone, so its recall is the recall of a build that does
    not subtract the wearer's hands. Every other number in package F2 does
    subtract them. The difference cannot be assumed small, so it is measured:
    for every pasted face that counts, how much of its identity ellipse lies
    under the polygons that protect a face, which are the strict hand quads
    when `zone_face_needs_hand` is on.

    A face is counted as taken back when more than a tenth of its ellipse is
    protected, because recall wants the mask's alpha to average 0.9 over the
    ellipse and the zone multiplies that alpha by what it protects.

    The number is an upper bound on an artefact rather than a property of the
    footage. A pasted face lands on the wearer's hands as often as chance puts
    it there; a real one rarely does.
    """
    from eval.zone import load_hands
    from faceblur.hands import Hand
    from faceblur.zone import alpha_for, build, moved

    settings = settings or Settings()
    held = load_hands(Path(video), settings)
    if held is None:
        return {"faces": 0, "taken_back": 0, "share": None,
                "note": "no hands cache; run eval.zone --cache-hands"}
    backgrounds = {t[0]: t[1] for t in task_list(consensus)}
    rows_for = held.get("frames", {})
    faces = taken = touched = 0
    for seq in data:
        shape = tuple(seq["shape"])
        rows = rows_for.get(str(backgrounds.get(seq["seed"])), [])
        found = [Hand(tuple(tuple(q) for q in r["quad"]), r["score"],
                      tuple(r["palm"]), r.get("presence", 1.0)) for r in rows]
        zone = build({0: found}, None, settings, 1, 30.0, shape)[0] if found else None
        protects = tuple(zone.hands) if zone and settings.zone_face_needs_hand else (
            tuple(zone.live) if zone else ())
        pan = seq.get("pan") or (0.0, 0.0)
        for i, (_, truth) in enumerate(seq["frames"]):
            here = tuple(moved(p, pan[0] * i, pan[1] * i) for p in protects) \
                if seq.get("kind") == "pan" else protects
            zalpha = alpha_for(shape, here, settings.feather) if here else None
            for d in truth:
                ident = ellipse_mask(shape, d, REFERENCE)
                if not ident.any() or _inside_share(shape, d) < 0.5:
                    continue
                faces += 1
                if zalpha is None:
                    continue
                share = float(zalpha[ident].mean())
                touched += share > 0.0
                taken += share > 0.1
    return {"faces": faces, "taken_back": taken, "touched": touched,
            "share": taken / faces if faces else None,
            "touched_share": touched / faces if faces else None, "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None)
    ap.add_argument("--zone-bias", action="store_true",
                    help="how much of the pasted face recall the zone takes back")
    args = ap.parse_args()
    video = Path(args.video)
    consensus = load_json(CACHE_DIR / f"{video.stem}.consensus.json")
    settings_in = Settings(**(json.loads(args.json) if args.json else {}))
    data = generate(video, consensus, settings=settings_in)
    if args.zone_bias:
        b = zone_bias(video, data, consensus, settings_in)
        if b["share"] is None:
            print(b["note"])
            return 1
        print(f"{video.name}: {b['faces']} pasted faces counted, "
              f"{b['touched']} touch the zone ({100*b['touched_share']:.2f}%), "
              f"{b['taken_back']} taken back ({100*b['share']:.2f}%)")
        return 0
    r = evaluate(settings_in, data)
    for k, (h, t, rate) in r["recall"].items():
        print(f"  {k:14s} {h:4d}/{t:4d} = {100*rate:5.1f}%")
    print(f"  mask area / identity ellipse area: {r['mask_over_target_area']:.2f}")
    if r["entry_delay_mean"] is not None:
        print(f"  entry delay {r['entry_delay_mean']:.2f} frames, covered at once "
              f"{100*r['entry_covered_at_once']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
