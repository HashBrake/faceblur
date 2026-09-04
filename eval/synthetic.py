"""Recall on faces whose position is known by construction. No labels.

Face crops come from the footage itself: pseudo-faces both detectors scored at
0.8 or more. Each crop is pasted into other frames of the same video at a range
of sizes, moving across a short sequence, with optional motion blur. The true
box and landmarks follow the paste, so recall is exact.

    .venv\\Scripts\\python.exe -m eval.synthetic VIDEO
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eval.common import (CACHE_DIR, REFERENCE, build_cache, det_from_dict,
                         ellipse_mask, load_json, read_frames, save_json, size_bucket)
from faceblur.detect import Detection, DetectorBank, filter_candidates, weak_candidates
from faceblur.pipeline import tracker_for
from faceblur.redact import build_alpha, ellipse_for
from faceblur.settings import Settings

SIZES = [24, 32, 48, 64, 96, 128]
BLURS = [0, 0, 7, 13]
SEQ_LEN = 15


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
    """Paste a face so its box long side is `long_side`, centred at (cx, cy)."""
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


_BANK = None
_FACES = None
_VIDEO = None


def _init(video, faces):
    global _BANK, _FACES, _VIDEO
    _BANK = DetectorBank(Settings())
    _FACES = faces
    _VIDEO = video


def _sequence(task):
    seed, background, long_side, blur = task
    rng = random.Random(seed)
    frames = {i: f for i, f in read_frames(Path(_VIDEO), {background})}
    base = frames[background]
    H, W = base.shape[:2]
    faces = rng.sample(_FACES, k=min(2, len(_FACES)))
    starts = [(rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.7) * H) for _ in faces]
    vel = [(rng.uniform(-5, 5), rng.uniform(-3, 3)) for _ in faces]
    angle = rng.uniform(0, 180)
    out = []
    for t in range(SEQ_LEN):
        frame = base.copy()
        truth = []
        for face, (sx, sy), (vx, vy) in zip(faces, starts, vel):
            d = paste(frame, face, long_side, sx + vx * t, sy + vy * t, blur, angle)
            if d is not None:
                truth.append(d)
        out.append((_BANK.detect_raw(frame), truth))
    return {"seed": seed, "long_side": long_side, "blur": blur, "shape": (H, W), "frames": out}


def generate(video: Path, consensus: dict, sequences: int = 36, workers: int = 10) -> list:
    path = CACHE_DIR / f"{video.stem}.synthetic.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    faces = face_bank(video, consensus)
    if not faces:
        raise SystemExit("no confident faces to build a bank from")
    n = max(consensus_frames := [int(k) for k in consensus["frames"]]) + 1
    rng = random.Random(7)
    tasks = []
    for s in range(sequences):
        tasks.append((s, rng.choice(consensus_frames), SIZES[s % len(SIZES)],
                      BLURS[(s // len(SIZES)) % len(BLURS)]))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(str(video), faces)) as pool:
        result = list(pool.imap_unordered(_sequence, tasks))
    path.write_bytes(pickle.dumps(result))
    return result


def evaluate(settings: Settings, data: list, cover: float = 0.9) -> dict:
    """Recall per size and blur, plus how much larger the mask is than the target.

    A face counts as covered when the mask's alpha averages at least `cover`
    over its identity ellipse. The area ratio compares the mask near that face
    (inside three times its box) with the identity ellipse.
    """
    hit = defaultdict(lambda: [0, 0])
    area_ratio = []
    coverage = []
    for seq in data:
        shape = seq["shape"]
        H, W = shape
        per = [filter_candidates(raw, settings, shape) for raw, _ in seq["frames"]]
        weak = [weak_candidates(raw, settings, shape) for raw, _ in seq["frames"]]
        tracked, _ = tracker_for(settings).run(per, weak)
        for i, (raw, truth) in enumerate(seq["frames"]):
            alpha = build_alpha(shape, tracked[i], settings) if tracked[i] else None
            for d in truth:
                key_size = size_bucket(d.long_side)
                key_blur = "blurred" if seq["blur"] else "sharp"
                ident = ellipse_mask(shape, d, REFERENCE)
                if not ident.any():
                    continue
                c = float(alpha[ident].mean()) if alpha is not None else 0.0
                coverage.append(c)
                ok = c >= cover
                for k in (key_size, key_blur, "all"):
                    hit[k][1] += 1
                    hit[k][0] += ok
                if ok:
                    x0, y0 = max(0, int(d.x - d.w)), max(0, int(d.y - d.h))
                    x1, y1 = min(W, int(d.x + 2 * d.w)), min(H, int(d.y + 2 * d.h))
                    near = (alpha[y0:y1, x0:x1] >= 0.5).sum()
                    area_ratio.append(float(near) / max(1, ident.sum()))
    return {
        "recall": {k: [v[0], v[1], v[0] / v[1]] for k, v in sorted(hit.items())},
        "mean_coverage": float(np.mean(coverage)) if coverage else None,
        "mask_over_target_area": float(np.median(area_ratio)) if area_ratio else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    video = Path(args.video)
    consensus = load_json(CACHE_DIR / f"{video.stem}.consensus.json")
    data = generate(video, consensus)
    settings = Settings(**(json.loads(args.json) if args.json else {}))
    r = evaluate(settings, data)
    for k, (h, t, rate) in r["recall"].items():
        print(f"  {k:14s} {h:4d}/{t:4d} = {100*rate:5.1f}%")
    print(f"  mask area / identity ellipse area: {r['mask_over_target_area']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
