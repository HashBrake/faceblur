"""Measure a settings choice on real footage, against consensus pseudo-labels.

Numbers per settings:

- recall: share of pseudo-faces whose identity ellipse the mask covers
- off_face: share of the frame masked where no detector sees a face. The zone
  a mask may sit in is every pseudo-face, plus every box any of the three
  detectors scored at 0.5 or more. The gate uses this number.
- off_face_strict: the same, counting only pseudo-faces as face. A real face
  the pseudo-label set missed counts against the pipeline here, so this is an
  upper bound. It stays in the report.
- off_face_detections: the part of off_face that comes from boxes a detector
  produced, as against tails and gap fills the tracker drew. A tail before a
  face enters the picture is off-face by construction; a detection box off a
  face is the thing the gate exists to catch.
- hand_damage: share of hand pixels the mask touches
- masked: share of the frame masked, mean, p95, max

    .venv\\Scripts\\python.exe -m eval.measure VIDEO
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.common import (CACHE_DIR, REFERENCE, build_cache, det_from_dict,
                         ellipse_mask, load_json, polygon_mask, shifts_list, size_bucket)
from faceblur.detect import filter_candidates, iou, weak_candidates
from faceblur.pipeline import tracker_for
from faceblur.redact import build_alpha
from faceblur.settings import Settings


def tracked(cache: dict, settings: Settings):
    n = max(cache["frames"]) + 1
    shape = tuple(cache["shape"])
    have = [i in cache["frames"] and i % settings.stride == 0 for i in range(n)]
    per = [filter_candidates(cache["frames"][i], settings, shape) if have[i] else []
           for i in range(n)]
    weak = [weak_candidates(cache["frames"][i], settings, shape) if have[i] else []
            for i in range(n)]
    shifts = shifts_list(cache, n) if settings.camera_comp else None
    return tracker_for(settings).run(per, weak, shifts, shape)


def dilated_box(d, factor=0.5):
    return (d.x - factor * d.w, d.y - factor * d.h, d.w * (1 + 2 * factor), d.h * (1 + 2 * factor))


def _paint(zone, box, W, H):
    x, y, w, h = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 > x0 and y1 > y0:
        zone[y0:y1, x0:x1] = True


def continuity(cache: dict, settings: Settings, per_frame, tracks, reach: int = 15) -> float:
    """Share of frames with a confirmed face still visible that carry a mask.

    For every confirmed track, look up to `reach` frames past each end. A frame
    counts as "face still visible" when YuNet fired at 0.3 or more on a box
    that overlaps the track's last box. The consensus recall cannot see these
    frames, because they are where the detectors disagree.
    """
    n = len(per_frame)
    seen = masked = 0
    for t in tracks:
        first, last = t.first, t.last
        head, foot = t.dets[first], t.dets[last]
        for i in range(first, last + 1):
            seen += 1
            masked += any(iou(d, t.dets.get(i) or head) >= 0.3 for d in per_frame[i]) \
                if i in t.dets else bool(per_frame[i])
        for anchor, rng in ((head, range(first - 1, max(-1, first - reach - 1), -1)),
                            (foot, range(last + 1, min(n, last + reach + 1)))):
            for i in rng:
                raw = cache["frames"].get(i)
                if raw is None:
                    break
                near = [d for d in raw.get("yunet", []) if d.score >= 0.3 and iou(d, anchor) >= 0.3]
                if not near:
                    break
                seen += 1
                masked += any(iou(d, near[0]) >= 0.3 for d in per_frame[i])
    return masked / seen if seen else 1.0


def evaluate(settings: Settings, cache: dict, consensus: dict, oracle: dict) -> dict:
    shape = tuple(cache["shape"])
    H, W = shape
    per_frame, tracks = tracked(cache, settings)

    covered = total = 0
    by_size = defaultdict(lambda: [0, 0])
    off, off_loose, off_det, masked = [], [], [], []
    hand_hit = hand_area = 0.0
    for key, faces in consensus["frames"].items():
        i = int(key)
        alpha = build_alpha(shape, per_frame[i], settings) if per_frame[i] else None
        hard = (alpha >= 0.5) if alpha is not None else np.zeros(shape, bool)
        masked.append(float(hard.mean()))

        face_zone = np.zeros(shape, bool)
        for f in faces:
            d = det_from_dict(f)
            total += 1
            bucket = size_bucket(d.long_side)
            by_size[bucket][1] += 1
            ident = ellipse_mask(shape, d, REFERENCE)
            hit = alpha is not None and ident.any() and float(alpha[ident].mean()) >= 0.9
            covered += hit
            by_size[bucket][0] += hit
            _paint(face_zone, dilated_box(d), W, H)
        off.append(float((hard & ~face_zone).mean()))

        # Loose zone: anything any detector scored at 0.5 or more, any size.
        # Masked pixels outside this are on something no detector calls a face.
        loose_zone = face_zone.copy()
        raw = cache["frames"].get(i, {})
        for name in ("yunet", "centerface"):
            for d in raw.get(name, []):
                if d.score >= 0.5:
                    _paint(loose_zone, dilated_box(d, 0.25), W, H)
        for b in oracle["frames"].get(key, {}).get("faces", []):
            if b[4] >= 0.5:
                _paint(loose_zone, (b[0] - 0.25 * b[2], b[1] - 0.25 * b[3],
                                    1.5 * b[2], 1.5 * b[3]), W, H)
        off_loose.append(float((hard & ~loose_zone).mean()))
        dets_only = [d for d in per_frame[i] if d.source != "track"]
        hard_det = (build_alpha(shape, dets_only, settings) >= 0.5) if dets_only \
            else np.zeros(shape, bool)
        off_det.append(float((hard_det & ~loose_zone).mean()))

        hands = oracle["frames"].get(key, {}).get("hands", [])
        if hands:
            hm = polygon_mask(shape, hands)
            hand_area += float(hm.sum())
            hand_hit += float((hm & hard).sum())

    off_a, off_l, m_a = np.asarray(off), np.asarray(off_loose), np.asarray(masked)
    off_d = np.asarray(off_det)
    return {
        "settings": settings.to_dict(),
        "pseudo_faces": total,
        "recall": covered / total if total else None,
        "recall_by_size": {k: [v[0], v[1]] for k, v in sorted(by_size.items())},
        "off_face_strict_mean": float(off_a.mean()),
        "off_face_strict_max": float(off_a.max()),
        "off_face_mean": float(off_l.mean()),
        "off_face_max": float(off_l.max()),
        "off_face_detections_mean": float(off_d.mean()),
        "hand_damage": hand_hit / hand_area if hand_area else None,
        "masked_mean": float(m_a.mean()),
        "masked_p95": float(np.percentile(m_a, 95)),
        "masked_max": float(m_a.max()),
        "tracks": len(tracks),
        "track_len_mean": float(np.mean([len(t) for t in tracks])) if tracks else 0.0,
        "continuity": continuity(cache, settings, per_frame, tracks),
        "frames_evaluated": len(masked),
    }


def load_inputs(video: Path):
    cache = build_cache(video)
    consensus = load_json(CACHE_DIR / f"{video.stem}.consensus.json")
    oracle = load_json(CACHE_DIR / f"{video.stem}.oracle.json")
    return cache, consensus, oracle


def describe(r: dict) -> str:
    rb = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in r["recall_by_size"].items())
    hd = "n/a" if r["hand_damage"] is None else f"{100*r['hand_damage']:.2f}%"
    return (f"recall {100*(r['recall'] or 0):5.1f}%  off-face mean {100*r['off_face_mean']:.2f}% "
            f"max {100*r['off_face_max']:.2f}% (strict {100*r['off_face_strict_mean']:.2f}%, "
            f"detections {100*r['off_face_detections_mean']:.2f}%)  "
            f"hands {hd}  continuity {100*r['continuity']:.1f}%  masked mean {100*r['masked_mean']:.2f}% "
            f"p95 {100*r['masked_p95']:.2f}% max {100*r['masked_max']:.2f}%  "
            f"tracks {r['tracks']}  [{rb}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--json", default=None, help="settings overrides as JSON")
    args = ap.parse_args()
    video = Path(args.video)
    settings = Settings(**(json.loads(args.json) if args.json else {}))
    cache, consensus, oracle = load_inputs(video)
    r = evaluate(settings, cache, consensus, oracle)
    print(describe(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
